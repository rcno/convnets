import numpy as np
import torch
import torch.nn as nn
import math
from tqdm import tqdm
import os

#from img_ae.img_data import *


def create_and_train(imgpath, ntrain):
    """Example function"""
    spec = basicspec()
    spec.update(insize=299, layer_ch=[1,4,8,16,32,64], ksizes=[3]*5, fcsizes=[], hdim=2048, maxpool=[1, 3, 4, 5])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    convae = ConvAE(spec).to(device)
    traindat = create_dataset(imgpath, eind=ntrain)
    testdat = create_dataset(imgpath, sind=ntrain)
    hist, opt = trainmod(convae, traindat, testdat, niter=40)
    return convae, hist, opt


def create_dataset(imgpath, sind=None, eind=None, repeat_ch=0):
    pathlist = imagepathlist(imgpath)
    sind = sind if sind else 0
    eind = eind if eind else len(pathlist)
    pathlist = pathlist[sind:eind]
    targetlist = np.zeros(len(pathlist))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ##scales images to meaqn zero, std 1.
    return img_data.ImageDataSet.initfromfiles(pathlist, targetlist, resize=(299, 299), repeat_ch=repeat_ch, augmentation=None,
                                               scaling="standard", device=device)


def imagepathlist(imgpath):
    return [os.path.join(imgpath, p) for p in list(os.listdir(imgpath))]


def pairplot(first,second, figsize=(7,7)):
    omin = torch.min(torch.cat((first.squeeze(), second.squeeze())))
    omax = torch.max(torch.cat((first.squeeze(), second.squeeze())))
    fig, ax = plt.subplots(ncols=2, figsize=figsize)
    ax[0].imshow(first.squeeze().detach().cpu(), vmin=omin, vmax=omax)
    ax[1].imshow(second.squeeze().detach().cpu(), vmin=omin, vmax=omax)
    plt.show()


def trainmod(model, traindat, testdat, testsplit=None, niter=30, bs=32, initlr=1e-2, lam=1e-2, opt=None, hist=None):
    """
    returns hist, opt
    traindat and testdat should be of type ImageDataSet,
    testsplit is None or a tuple indicating number of vertical and horizontal parts to split the testimage into before feeding into the model
    """
    trainsamp = torch.utils.data.DataLoader(traindat, batch_size=bs, shuffle=True)
    testarr = torch.vstack([img.unsqueeze(0) for img in testdat])

    optimizer = opt if opt else torch.optim.Adam(model.parameters(), lr=initlr, weight_decay=lam)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=15)
    criterion = nn.MSELoss()
    history = hist if hist else {"trainloss": [], "testloss": []}

    progbar = tqdm(range(niter*len(trainsamp)), total=niter*len(trainsamp))
    for e in range(niter):
        #print(e)
        model.train()
        tr_loss = 0
        for b, trainbatch in enumerate(trainsamp):
            optimizer.zero_grad()
            outputs = model(trainbatch)
            target = trainbatch if (traindat.repeat_ch == 0) else trainbatch[:, 0, :, :].unsqueeze(1) ##only look at one channel when training images have repeated channels
            loss = criterion(outputs, target)
            tr_loss += loss.item()
            loss.backward()
            optimizer.step()
            progbar.update()
        ##todo: change this to be unbiased if last batch is smaller than bs. This is the average batch loss, not that this is averaged over different weights, so will be higher than loss on train set after last batch.
        tr_loss /= len(trainsamp)
        scheduler.step(tr_loss)
        with torch.no_grad():
            model.eval()
            if testsplit:
                testloss = split_testloss(criterion, model, testdat, testsplit)
            else:
                testout = model(testarr) ##only look at one channel when test images have repeated channels
                targetarr = testarr if (testdat.repeat_ch == 0) else testarr[:, 0, :, :].unsqueeze(1) ##only look at one channel when test images have repeated channels
                testloss = criterion(testout, targetarr).item()
        update_metrics(history, progbar, testloss, tr_loss)
        if "cuda" in traindat.device.type:
            torch.cuda.empty_cache()

    return history, optimizer


def split_testloss(criterion, model, testdat, testsplit):
    testloss = 0
    for img in testdat:
        patches = torchcrop(img, testsplit[0], testsplit[1]) ##get (channel,nh,nv,row,col)
        ##dimensions 1 and 2 index the patches, combine these into one dimension, so get (channel,patch,row,col)
        patches = patches.reshape(patches.shape[0], -1, patches.shape[3], patches.shape[4])
        patchlosses = []
        for i in range(patches.shape[1]):
            patch = patches[:, i, :, :] ##get (channel,row,col)
            partout = model(patch.unsqueeze(0))
            target = patch.unsqueeze(0) if (testdat.repeat_ch == 0) else patch[0, :, :].unsqueeze(0).unsqueeze(0)  ##only look at one channel when test images have repeated channels
            patchlosses.append(criterion(partout, target).item())
        testloss += np.mean(patchlosses)
    testloss /= len(testdat)
    return testloss


def update_metrics(history, progbar, testloss, tr_loss):
    progbar.update()
    progbar.set_postfix(test_loss=f"{testloss:.4f}", train_loss=f"{tr_loss:.4f}")
    # print(tr_loss)
    history["trainloss"].append(round(tr_loss, 5))
    history["testloss"].append(round(testloss, 5))


def basicspec():
    return {"insize": 64, "layer_ch": [1, 8, 16, 32, 64], "fcsizes": [2048, 1024], "hdim": 512, "ksizes": [2, 2, 2, 2],
            "maxpool": [1, 3]}


def largespec():
    spec = basicspec()
    spec.update(insize=299, layer_ch=[1, 4, 8, 16, 32, 64, 128], fcsizes=[2048, 1024], hdim=512,
                ksizes=[3, 3, 3, 3, 3, 3], maxpool=[1, 2, 4], pad=1)
    return spec


class ConvAE(nn.Module):
    """
    Creates encoder and decoder modules designed to mirror each other.

    The encoder is based on convolution blocks and optional maxpool layers,
    where a convolution block consists of a convolution, optional batch-norm and then relu activation.
    Note that if there are no maxpool layers then the convolution blocks get stride 2.

    The decoder uses transpose convolution layers, optional batchnorm and relu activations, where
    the encoder maxpool layers correspond to transpose convolutions with kernel size 2 and stride 2.

    The spec should contain the following parameters:
        - layer_ch gives the number of channels in each layer, **starting with the number of channels in the input image**.
        - fcsizes gives the size of the fully connected layers between the code layer and the convolutions, can be empty.
        - hdim gives the size of the code layer. 
        - ksizes gives the sizes of the kernels in the convolution layers.
        - maxpool is a list of conv layers that have maxpool at the **end**, counting from zero
        - p1 and p2 are padding for even and odd **convolution** layers respectively
        - Note that the parameters layer_ch, maxpool and ksizes have indexes corresponding to the convolution layers,
          so not counting the maxpool layers.
    """

    def __init__(self, spec):
        super().__init__()
        self.imgsizes = None
        self.fcsizes = None
        print("Constructing encoder")
        self.enc = self.make_enc(spec)
        print(self.imgsizes)
        print("Constructing decoder")
        self.dec = self.make_dec(spec, self.imgsizes)
        print(self.fcsizes)

    def make_enc(self, spec):
        insize = spec.get("insize")
        layer_ch = spec.get("layer_ch")
        fcsizes = spec.get("fcsizes", [])
        hdim = spec.get("hdim", 100)
        ksizes = spec.get("ksizes")
        maxpool = spec.get("maxpool", [])
        batchnorm = spec.get("batchnorm", True)
        p1 = spec.get("pad", 0)
        bias = not batchnorm

        assert len(layer_ch) - 1 == len(ksizes), "See documentation of class"

        nconvs = len(layer_ch) - 1  ##first item is image channels
        stride = 1 if maxpool else 2
        currentsize = insize
        self.imgsizes = [currentsize]

        mlist = []
        for i in range(nconvs):
            pad = p1
            mlist.append(
                #ResBlock(layer_ch[i], layer_ch[i + 1], kernel_size=ksizes[i], stride=stride, bias=bias, batchnorm=batchnorm))
                ##A block is a convolutional layer, optional batchnorm and ReLU activation
                Conv2dBlock(layer_ch[i], layer_ch[i + 1], kernel_size=ksizes[i], stride=stride, bias=bias,
                            batchnorm=batchnorm, padding=pad))
            currentsize = math.floor((currentsize - ksizes[i] + 2 * pad) / stride + 1)
            self.imgsizes.append(currentsize)
            if maxpool and (i in maxpool):
                mlist.append(nn.MaxPool2d(kernel_size=2, stride=2))
                currentsize = currentsize // 2
                self.imgsizes.append(currentsize)
        ##todo: some check that imgsize is not zero
        conv = nn.Sequential(*mlist)

        ##Encoder fully connected layers
        flatdim = (layer_ch[-1] * currentsize ** 2)
        self.fcsizes = [flatdim]
        fcenc = nn.Sequential()
        fcenc.append(nn.Flatten(start_dim=1))
        if len(fcsizes) > 0:
            for i, _ in enumerate(fcsizes):
                inch = flatdim if i == 0 else fcsizes[i-1]
                fc = FcBlock(inch, fcsizes[i], bias=bias, batchnorm=batchnorm)
                fcenc.append(fc)
                self.fcsizes.append(fcsizes[i])
        ##final layer connecting to hdim
        if self.fcsizes[-1] != hdim:
            fcenc.append(FcBlock(self.fcsizes[-1], hdim, bias=bias, batchnorm=batchnorm))

        ##initialise layers
        for m in [conv, fcenc]:
            m.apply(self.initlayer)

        return nn.Sequential(conv, fcenc)

###########################################################################################################
    def make_dec(self, spec, imgsizes):
        assert imgsizes

        insize = spec.get("insize")
        layer_ch = spec.get("layer_ch")
        fcsizes = spec.get("fcsizes", [])
        hdim = spec.get("hdim", 100)
        ksizes = spec.get("ksizes")
        maxpool = spec.get("maxpool", [])
        batchnorm = spec.get("batchnorm", True)
        p1 = spec.get("pad", 0)
        sigact = spec.get("sigact", False)
        bias = not batchnorm

        ##Decoder fully connected layers
        flatdim = (layer_ch[-1] * imgsizes[-1] ** 2)
        fcdec = nn.Sequential()
        ##first layer connecting to hdim
        if hdim != self.fcsizes[-1]:
            fcdec.append(FcBlock(hdim, self.fcsizes[-1], bias=bias, batchnorm=batchnorm))
        if len(fcsizes) > 0:
            for i in range(1, len(fcsizes)+1):
                outch = flatdim if i == len(fcsizes) else fcsizes[-i-1]
                fc = FcBlock(fcsizes[-i], outch, bias=bias, batchnorm=batchnorm)
                fcdec.append(fc)
        fcdec.append(nn.Unflatten(dim=1, unflattened_size=(layer_ch[-1], imgsizes[-1], imgsizes[-1])))

        # Decoder convolutions
        assert len(layer_ch) - 1 == len(ksizes)
        nconvs = len(layer_ch) - 1

        currentsize = imgsizes[-1]
        ix = 1
        mlist = []
        for i in range(1, nconvs+1):
            ##j = nconvs-i, -i = j-nconvs; j: nconvs-1 --->0
            ##if we have conv+maxpool in encoder, use a doubling (stride 2) convtrans in decoder, may get one too small output, so use final upscaling
            ##if we have only conv in encoder, use an inverse convtrans
            if maxpool and ((nconvs - i) in maxpool):
                ##old version: usibng 2 invconvblocks (1 for inverse of maxpool), this trains badly without batchnorm
                pad = math.floor((ksizes[-i]-1)/2) ##adapt so we get doubling convtrans
                mlist.append(Inv_Convblock(currentsize, layer_ch[-i], layer_ch[-i - 1], ks=ksizes[-i],
                                           stride=2, bias=bias, batchnorm=batchnorm, pad=pad, outpad=1))
                currentsize = mlist[-1].outsize # should be currentsize*2
                ix += 1
            else:
                stride = 2 if ((imgsizes[-ix - 1] + 2 * p1) // currentsize) == 2 else 1
                outpad = 1 if (stride * (currentsize - 1) + ksizes[-i] - 2 * p1) < imgsizes[-ix - 1] else 0
                mlist.append(Inv_Convblock(currentsize, layer_ch[-i], layer_ch[-i - 1], ks=ksizes[-i],
                                           stride=stride, bias=bias, batchnorm=batchnorm, pad=p1, outpad=outpad))
                currentsize = mlist[-1].outsize
                ix += 1
            print("i {}, currentsize {}".format(i, currentsize))
        # last conv transpose activation should be something other than relu to give real-valued output
        mlist[-1].act = nn.Sigmoid() if sigact else nn.Identity()
        ##final upsample
        mlist.append(nn.Upsample(size=imgsizes[0], mode="bilinear"))
        currentsize = imgsizes[0]
        #print("i {} currentsize {}".format(i, currentsize))

        assert currentsize == insize, "currentsize is {}, input size is {}, sizelist is {}".format(currentsize, insize, imgsizes)
        conv = nn.Sequential(*mlist)

        ##initialise layers
        for m in [fcdec, conv]:
            m.apply(self.initlayer)

        return nn.Sequential(fcdec, conv)

    def forward(self, x):
        h = self.enc(x)
        xrec = self.dec(h)
        return xrec

    @staticmethod
    def initlayer(m, orthinit=False):
        if (isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d)):
            if orthinit:
                print("Doing orthogonal init")
                torch.nn.init.orthogonal_(m.weight.data)
            else:
                print("Doing Kaiming normal init")
                torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')


class Conv2dBlock(nn.Module):
    """
    A convolution, optional batchnorm and relu activation.
    Note the high value for the batch-norm momentum, this is to make training and test more similar.
    Also note that pytorch defines momentum in an unintuitive way, see https://github.com/pytorch/pytorch/issues/41559
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=True, batchnorm=True, mom=0.8, padding=0):
        super().__init__()
        self.batchnorm = batchnorm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if self.batchnorm:
            self.bn = nn.BatchNorm2d(out_channels, momentum=mom)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        xa = self.relu(x)
        return xa


class Inv_Convblock(nn.Module):
    """
    layer corresponding to convolution in encoder
    ##NB: conv operation can have input size 2n or 2n+1 for output size n,
    ##so might need to add output padding to get corresponding size here
    """
    def __init__(self, insize, inch, outch, ks, stride, bias=True, batchnorm=True, pad=0, outpad=0, mom=0.8):
        super().__init__()
        #newsize = stride * (insize - 1) + ks - 2 * pad
        #if not (outsize in [newsize, newsize + 1]):
        ##raise AssertionError("target size is {}, but previous size is {}, new is {}".format(outsize, insize,newsize))
        #outpad = 0 if outsize == newsize else 1
        self.convtrans = nn.ConvTranspose2d(inch, outch, kernel_size=ks, stride=stride,
                                            output_padding=outpad, bias=bias, padding=pad)
        #self.outsize = newsize + outpad
        self.outsize = stride * (insize - 1) + ks - 2 * pad + outpad
        self.batchnorm = batchnorm
        if batchnorm:
            self.bn = nn.BatchNorm2d(outch, momentum=mom)
        self.act = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.convtrans(x)
        if self.batchnorm:
            x = self.bn(x)
        xa = self.act(x)
        return xa


class FcBlock(nn.Module):
    """
    A fully connected layer, optional batchnorm and relu activation
    Note the high value for the batch-norm momentum, this is to make training and test more similar.
    Also note that pytorch defines momentum in an unintuitive way, see https://github.com/pytorch/pytorch/issues/41559
    """
    def __init__(self, in_channels, out_channels, bias=True, batchnorm=True, mom=0.8):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_channels, momentum=mom)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        if self.batchnorm:
            x = self.bn(x)
        xa = self.relu(x)
        return xa



class InvResBlock(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=True, batchnorm=True, mom=0.8):
        super().__init__()
        assert (kernel_size % 2) == 1, "Need odd kernel to make same conv"
        pad = (kernel_size - 1) / 2
        self.batchnorm = batchnorm
        ##the first conv can change the image size by using stride!=1
        self.conv1 = Conv2dBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, batchnorm=batchnorm,
                                 padding=pad, bias=bias, mom=mom)
        ##the second conv should preserve the image size
        self.conv2 = Conv2dBlock(out_channels, out_channels, kernel_size=kernel_size, stride=1, batchnorm=batchnorm,
                                 padding=pad, bias=bias, mom=mom)
        self.conv2.relu = nn.Identity()  ##NB: remove relu in last block
        ###convskip has kernel size 1 and transforms #channels and image size like the first conv
        ###kernel 1 and stride 2 is just subsampling every other pixel across all channels.
        self.convskip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) \
            if stride != 1 or in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        xskip = self.convskip(x)
        xa = self.relu(x2 + xskip)
        return xa



class ResBlock(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=True, batchnorm=True, mom=0.8):
        super().__init__()
        assert (kernel_size % 2) == 1, "Need odd kernel to make same conv"
        pad = (kernel_size - 1) / 2
        self.batchnorm = batchnorm
        ##the first conv can change the image size by using stride!=1
        self.conv1 = Conv2dBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, batchnorm=batchnorm,
                                 padding=pad, bias=bias, mom=mom)
        ##the second conv should preserve the image size
        self.conv2 = Conv2dBlock(out_channels, out_channels, kernel_size=kernel_size, stride=1, batchnorm=batchnorm,
                                 padding=pad, bias=bias, mom=mom)
        self.conv2.relu = nn.Identity()  ##NB: remove relu in last block
        ###convskip has kernel size 1 and transforms #channels and image size like the first conv
        ###kernel 1 and stride 2 is just subsampling every other pixel across all channels.
        self.convskip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) \
            if stride != 1 or in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        xskip = self.convskip(x)
        xa = self.relu(x2 + xskip)
        return xa


####################################################################


