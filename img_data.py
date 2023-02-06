import numpy as np

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import torch
from PIL import Image

from tqdm import tqdm

def datainfo(dataset):
    print({"nimg": dataset.nimg, "augmentation": dataset.augmentation, "device": dataset.device, "resize": dataset.resize,
           "imagemode": dataset.imgmode, "scaling": dataset.scaling, "repeat_ch": dataset.repeat_ch})


##img should be a numpy array with channel as last dimension
def partscrop(img, hparts, vparts):
    rows = np.split(img, vparts, axis=0)
    parts = np.array([np.split(row, hparts, axis=1) for row in rows])
    return parts


##img should be a torch tensor with channel as FIRST dimension
def torchcrop(img, vparts, hparts):
    vstep = img.shape[1]//vparts
    hstep = img.shape[2]//hparts
    parts = img.unfold(1, vstep, vstep).unfold(2, hstep, hstep) #arguments to unfold are dimension, size and step.
    return parts


##assumes parts is indexed like (c,hpatch,vpatch,row,col)
def torchpaste(parts):
    pasted = []
    for c in range(parts.shape[0]):
        ##parts[c,i,j] has 2 dimensions (row,col), first cat horizontally then vertically for each channel
        cpaste = torch.cat([torch.cat([parts[c, i, j] for j in range(parts.shape[2])], dim=1) for i in range(parts.shape[1])], dim=0)
        pasted.append(cpaste.unsqueeze(0))
    img = torch.cat(pasted, dim=0)
    return img


class ImageDataSet(torch.utils.data.Dataset):
    """
    Class for dealing with image loading, processing and augmenting.
    It provides a `__getitem__` method that returns the  **possibly augmented** image as a pytorch tensor with the channel dimension first.
    The preprocessed images are stored in `self.imagelist`, which is indexed by [sample] and has numpy arrays[width,height,channel] as elements.
    Input arguments:
    imagelist should be a list of PIL image objects or numpy arrays indexed by [width,height,channel]
    Resize is a tuple or None.
    Imgmode refers to PIL mode,
    Augmentation is a function working on numpy arrays.
    Scaling is "standard" (i.e. mean zero std 1) or "minmax" (scaling to interval (0,1)).
    Repeat_ch is an optional argument that makes "getitem" repeat the image the given number of times in the channel dimension.
    """
    def __init__(self, imagelist, targetlist, resize, imgmode="L", augmentation=None, scaling="standard", device=torch.device("cpu"), repeat_ch=0):
        super().__init__()
        assert (targetlist is None) or (len(imagelist) == len(targetlist))
        self.nimg = len(imagelist)
        self.targetlist = targetlist
        self.resize = resize
        self.augmentation = augmentation
        self.device = device
        self.imgmode = imgmode
        self.repeat_ch = repeat_ch
        self.scaling = scaling
        # do preprocessing first here for all images, augmentation is done on single image in the getitem function
        self.imagelist = self.nimg * [None]
        print("preprocessing images")
        for i in tqdm(range(self.nimg)):
            self.imagelist[i] = self.preprocess(imagelist[i])

    @classmethod
    def initfromfiles(cls, imagepathlist, targetlist, resize, repeat_ch=0, augmentation=None, scaling="standard", device=torch.device("cpu")):
        ##resize is a tuple or None, augmentation is a function working on numpy arrays
        imgshape = np.array(Image.open(imagepathlist[0])).shape
        print("first input image shape", imgshape)
        imagelist = len(imagepathlist) * [None]
        print("reading images from file")
        for i in tqdm(range(len(imagepathlist))):
            image = Image.open(imagepathlist[i])
            #imagelist[i] = np.array(image, dtype=np.float32)
            imagelist[i] = image
        return cls(imagelist, targetlist, resize, imgmode=image.mode, repeat_ch=repeat_ch, augmentation=augmentation,
                   scaling=scaling, device=device)

    def __len__(self):
        return self.nimg

    # gets the image indexed by item and does optional augmentation,
    # returns torch Tensor object
    def __getitem__(self, item):
        image = self.imagelist[item]
        if self.augmentation is not None:
            augmented = self.augmentation(image=image)
            image = augmented
        # convert to tensor and transpose so channel is first for training
        image = torch.tensor(image, device=self.device,dtype=torch.float32)
        image = torch.permute(image, (2, 0, 1))
        if self.repeat_ch > 0:
            ##duplicate each image in colour channel dimension
            image = image.repeat(self.repeat_ch, 1, 1)
        return image

    def gettarget(self, item):
        target = self.targetlist[item]
        return torch.tensor(np.array(target), device=self.device)

    def showimage(self, i):
        if self.imgmode == "RGB":
            ##note: imshow needs integer input for RGB image, otherwise clips floats to [0,1] and interprets in [0,255] scale.
            plt.imshow(self.convertint8(self.imagelist[i]))
            # plt.show()
        if self.imgmode == "L":
            ##only one channel
            plt.imshow(self.imagelist[i].squeeze(), cmap="gray")
            # plt.show()

    ##creates PIL Image object from numpy array, does optional resize and scales values
    def preprocess(self, image):
        if isinstance(image, np.ndarray):
            ##NB: Convert numpy float array to uint8 (0-255) format because pillow library is annoying
            intimg = self.convertint8(image, axis=(0, 1))
            ##Note the squeeze to get rid of possible single channel
            image = Image.fromarray(np.squeeze(intimg), mode=self.imgmode)
        if self.resize is not None:
            image = image.resize(self.resize, resample=Image.Resampling.BILINEAR)
        ##convert png palette to 4 channels and drop transparency channel
        if hasattr(image, "mode") and image.mode == "P":
            image = np.array(image.convert())[:, :, :3]
        ##convert to normal float
        img = np.array(image, dtype=np.float32)
        img = self.fixchannelshape(img, self.imgmode)
        scaledimage = self.scale(img)
        return scaledimage

    def scale(self, img):
        if self.scaling == "standard":
            scaledimage = ImageDataSet.standardscale(img)
        elif self.scaling == "minmax":
            scaledimage = ImageDataSet.minmaxscale(img)
        else:
            NotImplementedError("Scaling" + self.scaling + "Not implemented")
        return scaledimage

    @staticmethod
    def fixchannelshape(image, imgmode):
        ##add dummy channel in third dimension for grayscale image
        if image.ndim == 2:
            image = image.reshape(*image.shape[:2], -1)
        chdict = {"RGB": 3, "L": 1}
        assert (chdict[imgmode] == image.shape[-1])
        return image

    @staticmethod
    ##convert a float image to int8, i.e. range [0,255]
    def convertint8(img):
        if img.dtype == np.uint8:
            return img
        scaled = ImageDataSet.minmaxscale(img)
        return (255 * scaled).astype(np.uint8)

    @staticmethod
    ##scales into [0,1], channels with constant values across image are set to 1
    def minmaxscale(img):
        num = img - np.min(img)
        scale = np.max(img) - np.min(img)
        scaledimg = np.divide(num, scale, out=np.ones(img.shape,dtype=np.float32), where=(np.abs(scale) > 1e-8))
        return scaledimg

    @staticmethod
    ##channels with constant values are set to zero
    def standardscale(img):
        std = np.std(img)
        scaledimage = np.divide(img - np.mean(img), std, out=np.zeros(img.shape, dtype=np.float32), where=(np.abs(std) > 1e-8))
        return scaledimage


#################################################################
class NoiseAugment:
    def __init__(self, noise_type="add", sigma=0.1, device="cpu"):
        self.noise_type = noise_type
        self.sigma = sigma
        self.device = device

    def noisetransform(self, img):
        if self.noise_type == "add":
            randvec = np.random.randn(*img.shape)
            img = img + self.sigma * randvec
            return img
        if self.noise_type == "multiply":
            randvec = np.random.randn(*img.shape)
            img = img * (1 + self.sigma * randvec)
            scaledimage = (img - np.min(img, axis=(0, 1), keepdims=True)) / (np.max(img, axis=(0, 1), keepdims=True) - np.min(img, axis=(0, 1), keepdims=True))
            return scaledimage
        raise NotImplemented("Undefined noise type" + str(self.noise_type))
