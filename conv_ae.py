import numpy as np
import torch
import torch.nn as nn
import math
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

#from img_ae.img_data import *


def create_and_train(imgpath, ntrain):
    """Example function"""
    spec = basicspec()
    #spec.update(insize=299, res=False, layer_ch=[1, 4, 8, 16, 32, 64], ksizes=[3] * 5, fcsizes=[], hdim=2048, maxpool=[1, 3, 4])
    spec.update(insize=299, res=True, layer_ch=[1,4,8,16,32,64], ksizes=[3]*5, fcsizes=[], hdim=2048, maxpool=[1, 3, 4])
    #spec.update(insize=299, res=True, layer_ch=[1, 4, 8, 16, 32, 64], ksizes=[3] * 5, fcsizes=[], hdim=2048, maxpool=False)
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


def plot_rand_channels(firstimg,outchannels,nrows,ncols):
    fig, ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(25,8))
    nch = outchannels.shape[0]
    randinds = np.random.choice(nch,size=nrows*ncols)
    ax[0,0].imshow(firstimg)
    for i in range(nrows):
        for j in range(ncols):
            if i == 0 and j == 0:
                continue 
            ax[i,j].imshow(outchannels[randinds[ncols*i + j - 1]])
            ax[i,j].set_title(randinds[ncols*i + j - 1])
    plt.subplots_adjust(left=0.3,right=0.7,top=0.95,bottom=0.05, hspace=0.1,wspace=0.3)

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
    the encoder maxpool layers correspond to transpose convolutions with stride 2.
    The decoder has a final upsampling layer because the transpose convolution may not
    exactly match the sizes of the conv/maxpool layers in the encoder.

    The spec should contain the following parameters:
        - layer_ch gives the number of channels in each layer, **starting with the number of channels in the input image**.
        - fcsizes gives the size of the fully connected layers between the code layer and the convolutions, can be empty.
        - hdim gives the size of the code layer. 
        - ksizes gives the sizes of the kernels in the convolution layers.
        - maxpool is a list of conv layers that have maxpool at the **end**, counting from zero
        - pad is the padding for the convolution layers, ignored for residual version as the skip connection needs different pad.
        - Note that the parameters layer_ch, maxpool and ksizes have indexes corresponding to the convolution layers,
          so not counting the maxpool layers.
    """

    def __init__(self, spec):
        super().__init__()
        self.insize = spec.get("insize")
        self.layer_ch = spec.get("layer_ch")
        self.fcsizes = spec.get("fcsizes", [])
        self.hdim = spec.get("hdim", 100)
        self.ksizes = spec.get("ksizes")
        self.maxpool = spec.get("maxpool", [])
        self.batchnorm = spec.get("batchnorm", True)
        self.p1 = spec.get("pad", 0)
        self.sigact = spec.get("sigact", False)
        self.res = spec.get("res", False)
        self.longskip = spec.get("longskip", [])
        ##########################################
        self.encimgsizes = None
        self.decimgsizes = None
        self.flatdim = None

        print("Constructing encoder")
        self.enc = self.make_enc()
        self.test_sizes(encoder=True)
        print("Encoder sizes", self.encimgsizes)

        print("Constructing decoder")
        self.dec = self.make_dec()
        self.test_sizes(encoder=False)
        print("Longskip", self.longskip)
        print("Fully connected sizes", self.fcsizes)
        print("Decoder sizes", self.decimgsizes)

    def test_sizes(self, encoder=True):
        test = torch.ones(1, self.layer_ch[0], self.insize, self.insize) if encoder \
            else torch.ones(1, self.layer_ch[-1], self.decimgsizes[0], self.decimgsizes[0])
        convs = self.enc[0] if encoder else self.dec[1]
        for i, layer in enumerate(convs):
            test = layer(test)
            esize = self.encimgsizes[i + 1] if encoder else self.decimgsizes[i+1]
            assert test.shape[-1] == esize, "output shape is {}, stored size is {}".format(test.shape[-1],esize)

    def make_enc(self):
        assert len(self.layer_ch) - 1 == len(self.ksizes), "See documentation of class"

        conv = self.make_enc_conv()
        currentsize = self.encimgsizes[-1]

        ##Encoder fully connected layers
        self.flatdim = (self.layer_ch[-1] * currentsize ** 2)
        currentsize = self.flatdim
        fcenc = nn.Sequential()
        fcenc.append(nn.Flatten(start_dim=1))
        if len(self.fcsizes) > 0:
            for i, _ in enumerate(self.fcsizes):
                fc = FcBlock(currentsize, self.fcsizes[i], batchnorm=self.batchnorm)
                fcenc.append(fc)
                currentsize = self.fcsizes[i]
        ##final layer connecting to hdim
        if currentsize != self.hdim:
            fcenc.append(FcBlock(currentsize, self.hdim, batchnorm=self.batchnorm))

        ##initialise layers
        # print("Doing orthogonal init")
        print("Doing Kaiming normal init")
        for m in [conv, fcenc]:
            #m.apply(self.initlayer, orthinit=True)
            m.apply(self.initlayer)

        return nn.Sequential(conv, fcenc)

    def make_enc_conv(self):
        nconvs = len(self.layer_ch) - 1  ##first item is image channels
        stride = 1 if self.maxpool else 2  ##decoder uses this
        currentsize = self.insize
        self.encimgsizes = [currentsize]
        mlist = []
        for i in range(nconvs):
            if self.res:
                ##Residual block with 2 convs, last is same, adjusts pad such that output size is (n (-1))//stride
                ci = ResBlock(currentsize, self.layer_ch[i], self.layer_ch[i + 1], kernel_size=self.ksizes[i], stride=stride,
                              batchnorm=self.batchnorm)
                currentsize = ci.outsize
            else:
                ##A block is a convolutional layer, optional batchnorm and ReLU activation
                ci = Conv2dBlock(self.layer_ch[i], self.layer_ch[i + 1], kernel_size=self.ksizes[i], stride=stride,
                                 batchnorm=self.batchnorm, padding=self.p1)
                currentsize = math.floor((currentsize - self.ksizes[i] + 2 * self.p1) / stride) + 1
            mlist.append(ci)
            self.encimgsizes.append(currentsize)
            if self.maxpool and (i in self.maxpool):
                mlist.append(nn.MaxPool2d(kernel_size=2, stride=2))
                currentsize = currentsize // 2
                self.encimgsizes.append(currentsize)
        ##todo: some check that imgsize is not zero
        conv = nn.Sequential(*mlist)
        return conv

    ###########################################################################################################
    def make_dec(self):
        assert self.encimgsizes

        ##Decoder fully connected layers
        fcdec = nn.Sequential()
        currentsize = self.hdim
        if len(self.fcsizes) > 0:
            for i in range(len(self.fcsizes)):
                fc = FcBlock(currentsize, self.fcsizes[-i-1], batchnorm=self.batchnorm)
                fcdec.append(fc)
                currentsize = self.fcsizes[-i-1]
        ##final layer connecting to flatdim
        fcdec.append(FcBlock(currentsize, self.flatdim, batchnorm=self.batchnorm))
        fcdec.append(nn.Unflatten(dim=1, unflattened_size=(self.layer_ch[-1], self.encimgsizes[-1], self.encimgsizes[-1])))

        conv = self.make_dec_conv()

        ##initialise layers
        # print("Doing orthogonal init")
        print("Doing Kaiming normal init")
        for m in [fcdec, conv]:
            # m.apply(self.initlayer, orthinit=True)
            m.apply(self.initlayer)

        return nn.Sequential(fcdec, conv)

    def make_dec_conv(self):
        assert len(self.layer_ch) - 1 == len(self.ksizes)
        nconvs = len(self.layer_ch) - 1
        currentsize = self.encimgsizes[-1]
        ix = 1
        self.decimgsizes = [currentsize]
        mlist = []
        for i in range(1, nconvs + 1):
            ##j = nconvs-i, -i = j-nconvs; j: nconvs-1 --->0
            ismaxpool = self.maxpool and ((nconvs - i) in self.maxpool)
            if self.res:
                ##note: may get somewhat too small output as invresblock uses padding based on
                #reversing convolutions, not maxpool, so use final upscaling
                ci = self.make_resinvblock(currentsize, maxoutsize=self.encimgsizes[-ix-1], i=i, domaxpool=ismaxpool)
            else:
                ##if we have conv(stride 1) followed by maxpool in encoder, use a doubling (stride 2) convtrans in decoder,
                # may get too small output due to initial conv, so use final upscaling
                ci = self.make_invblock(currentsize, maxoutsize=self.encimgsizes[-ix-1], i=i, domaxpool=ismaxpool)
            ix = ix + 2 if ismaxpool else ix + 1
            mlist.append(ci)
            currentsize = ci.outsize
            self.decimgsizes.append(currentsize)
            #print("i {}, currentsize {}".format(i, currentsize))
        # last conv transpose activation should be something other than relu to give real-valued output
        mlist[-1].act = nn.Sigmoid() if self.sigact else nn.Identity()
        ##final upsample
        mlist.append(nn.Upsample(size=self.encimgsizes[0], mode="bilinear"))
        self.decimgsizes.append(self.encimgsizes[0])
        conv = nn.Sequential(*mlist)
        return conv

    def make_resinvblock(self, currentsize, maxoutsize, i, domaxpool):
        stride = 2 if domaxpool else (1 if self.maxpool else 2)
        pad = self.ksizes[-i] // 2 ##this is the padding used in the inverse resblock convolution
        realoutsize = calc_invconvoutsize(currentsize, self.ksizes[-i], maxoutsize, pad, stride)
        ci = InvResBlock(currentsize, realoutsize, self.layer_ch[-i], self.layer_ch[-i - 1], kernel_size=self.ksizes[-i],
                         stride=stride, batchnorm=self.batchnorm)
        return ci

    def make_invblock(self, currentsize, maxoutsize, i, domaxpool):
        if domaxpool:
            stride = 2
            pad = calc_invmpoolpad(self.ksizes[-i], maxoutsize)
            realoutsize = calc_invconvoutsize(currentsize, self.ksizes[-i], maxoutsize, pad, stride)
            ci = InvConvblock(currentsize, realoutsize, self.layer_ch[-i], self.layer_ch[-i - 1], kernel_size=self.ksizes[-i],
                              stride=stride, batchnorm=self.batchnorm, pad=pad)
        else:
            stride = 1 if self.maxpool else 2
            pad = self.p1
            realoutsize = calc_invconvoutsize(currentsize, self.ksizes[-i], maxoutsize, pad, stride)
            ci = InvConvblock(currentsize, realoutsize, self.layer_ch[-i], self.layer_ch[-i - 1], kernel_size=self.ksizes[-i],
                              stride=stride, batchnorm=self.batchnorm, pad=pad)
        return ci


    def make_invresblock(self, currentsize, outsize, i, stride):
        ci = InvResBlock(currentsize, outsize, self.layer_ch[-i], self.layer_ch[-i - 1], kernel_size=self.ksizes[-i],
                         stride=stride, batchnorm=self.batchnorm)
        return ci

    def forward(self, x):
        #h = self.enc(x)
        # store results on convolutions
        inter = [] 
        nconvs = len(self.enc[0])
        for i, layer in enumerate(self.enc[0]):
            x = layer(x)
            if self.longskip and (i in self.longskip): 
                inter.append(x)
        h = self.enc[1](x)
        xrec = self.dec[0](h)
        for i, layer in enumerate(self.dec[1]):
            if self.longskip and (nconvs - 1 - i in self.longskip) and (not isinstance(layer, torch.nn.modules.upsampling.Upsample)):
                skipcon = inter.pop()
                xrec = layer(skipcon + xrec)
            else:
                xrec = layer(xrec)
        #xrec = self.dec(h)
        return xrec

    @staticmethod
    def initlayer(m, orthinit=False):
        if (isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d)):
            if orthinit:
                torch.nn.init.orthogonal_(m.weight.data)
            else:
                torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')


class Conv2dBlock(nn.Module):
    """
    A convolution, optional batchnorm and relu activation.
    Note the high value for the batch-norm momentum, this is to make training and test more similar.
    Also note that pytorch defines momentum in an unintuitive way, see https://github.com/pytorch/pytorch/issues/41559
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, batchnorm=True, mom=0.8, padding=0):
        super().__init__()
        self.batchnorm = batchnorm
        bias = not batchnorm
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


class InvConvblock(nn.Module):
    """
    layer corresponding to convolution in encoder
    ##NB: conv operation can have input size 2n or 2n+1 for output size n,
    ##so might need to add output padding to get corresponding size here
    """
    def __init__(self, insize, outsize, inch, outch, kernel_size, stride, batchnorm=True, pad=0, mom=0.8):
        super().__init__()
        bias = not batchnorm
        outpad = calc_outpad(kernel_size, pad, stride, outsize)
        self.convtrans = nn.ConvTranspose2d(inch, outch, kernel_size=kernel_size, stride=stride,
                                            output_padding=outpad, bias=bias, padding=pad)
        self.outsize = stride * (insize - 1) + kernel_size - 2 * pad + outpad
        assert self.outsize == outsize, print("invconv outsize{}, required outsize {}".format(self.outsize, outsize))
        self.batchnorm = batchnorm
        if batchnorm:
            self.bn = nn.BatchNorm2d(outch, momentum=mom)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
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
    def __init__(self, in_channels, out_channels, batchnorm=True, mom=0.8):
        super().__init__()
        bias = not batchnorm
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
    Inverse of ResBlock
    """
    def __init__(self, insize, outsize, in_channels, out_channels, kernel_size, stride, batchnorm=True, mom=0.8):
        super().__init__()
        assert (kernel_size % 2) == 1, "Need odd kernel to make same conv"
        pad = kernel_size // 2  ##this reduces kernel_size f in the output formula to (f mod 2) = 1 for f odd
        self.batchnorm = batchnorm ##store for clarity
        ##this conv should preserve the image size
        self.convtrans1 = InvConvblock(insize, insize, in_channels, in_channels, kernel_size=kernel_size, stride=1,
                                       batchnorm=batchnorm, pad=pad, mom=mom)
        self.convtrans2 = InvConvblock(insize, outsize, in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                       batchnorm=batchnorm, pad=pad, mom=mom)
        self.convtrans2.relu = nn.Identity() ##NB: remove relu in last block
        self.outsize = self.convtrans2.outsize
        ##this should be the same outpad as that used in the second InvConvblock, as we have set the
        ##padding here and above to be floor(kernel/2), then the conv output only depends on kernel mod 2.
        skipoutpad = calc_outpad(ks=1, pad=0, stride=stride, target=outsize)
        ###convskip has kernel size 1 and transforms #channels and image size like the second conv
        self.convskip = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride,
                                           padding=0, output_padding=skipoutpad) \
            if stride != 1 or in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.convtrans1(x)
        x2 = self.convtrans2(x1)
        xskip = self.convskip(x)
        xa = self.relu(x2 + xskip)
        return xa


class ResBlock(nn.Module):
    """
    Block has a conv part and a skip connection.
    The conv part has two convolutions, the first outputs image of size n//stride or (n+1)//stride and out_channels channels, the second keeps the image size and
    channels.
    The skip connection uses a 1*1 kernel and changes the image size and channels like the first convolution.
    """
    def __init__(self, insize, in_channels, out_channels, kernel_size, stride, bias=True, batchnorm=True, mom=0.8):
        super().__init__()
        assert (kernel_size % 2) == 1, "Need odd kernel for res block"
        pad = kernel_size // 2 ##this reduces kernel_size f in the output formula to (f mod 2) = 1 for f odd
        self.batchnorm = batchnorm ##store for clarity
        ##the first conv can change the image size to (n (+1))//stride by using stride!=1
        self.conv1 = Conv2dBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, batchnorm=batchnorm,
                                 padding=pad, mom=mom)
        self.outsize = math.floor((insize - kernel_size + 2 * pad) / stride) + 1
        ##the second conv should preserve the image size, so use stride 1
        self.conv2 = Conv2dBlock(out_channels, out_channels, kernel_size=kernel_size, stride=1, batchnorm=batchnorm,
                                 padding=pad, mom=mom)
        self.conv2.relu = nn.Identity()  ##NB: remove relu in last block
        ###convskip has kernel size 1 and transforms #channels and image size like the first conv
        ###kernel 1 and stride 2 is just subsampling every other pixel across all channels.
        self.convskip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0) \
            if stride != 1 or in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        xskip = self.convskip(x)
        xa = self.relu(x2 + xskip)
        return xa


## try to do outpad such that inverse convolution outputs maxoutsize, return closest output size
def calc_invconvoutsize(currentsize, ks, maxoutsize, pad, stride):
    outpad = calc_outpad(ks, pad, stride, maxoutsize)
    realoutsize = calc_invsize(currentsize, ks, outpad, pad, stride=stride)
    if realoutsize < maxoutsize:
        realoutsize = calc_invsize(currentsize, ks, outpad=stride - 1, pad=pad, stride=stride)
    return realoutsize

def calc_invsize(insize, kernel_size, outpad, pad, stride):
    return stride * (insize - 1) + kernel_size - 2 * pad + outpad

def calc_outpad(ks, pad, stride, target):
    q = (target - ks + 2 * pad) % stride
    return q

def calc_invmpoolpad(kernel_size, outsize):
    pad = kernel_size // 2 if (outsize % 2 == 0) else (kernel_size - 1) // 2
    return pad
####################################################################


