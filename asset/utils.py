from IPython.display import Markdown
import json
import tqdm
import math
import time
import queue as Q
import threading as T
import os.path
from collections import defaultdict
from scipy import stats
from scipy.ndimage.interpolation import zoom
import scipy.misc
from shutil import copyfile
from IPython.display import Javascript
from IPython.display import clear_output
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import numbers
import numpy as np
import random as rand
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad
import cv2
import numpy as np
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from torch.functional import F
from hyperopt import hp, tpe, fmin
import hyperopt
import pandas as pd
import datetime
import requests
import ipykernel
import re
from notebook.notebookapp import list_running_servers
import pprint
import ipywidgets as widgets
from ipywidgets import Layout, Button, Box,VBox,HBox, IntSlider, Label, HTML, IntText
from IPython.display import display
import pdb
import traceback
from colorama import Fore, Back, Style
from math import exp

try:
    TOKEN = "mytoken"

    base_url = next(list_running_servers())['url']
    r = requests.get(
        url=base_url + 'api/sessions',
        headers={'Authorization': 'token {}'.format(TOKEN),})

    r.raise_for_status()
    response = r.json()

    kernel_id = re.search('kernel-(.*).json', ipykernel.connect.get_connection_file()).group(1)
    theNotebook = ({r['kernel']['id']: r['notebook']['path'] for r in response}[kernel_id]).split("/")[-1].replace(".ipynb","")
except:
    theNotebook="Untitled"
dimention=2
seedNumber=10
li=0;ui=li+1
EPS = 0.000000001
normalizeFFT=False
experiment=None
class MovingFGOnBGDataset(Dataset):
    """Moving foreground dataset on background."""
    def __init__(self, infrencePhase, seqLength, shape, scale=2,foregroundScale=0.75, blurIt=True, subpixel=True,minResultSpeed=0, maxResultSpeed=2, square=True,background="random",foreground="MNIST"):
        super(MovingFGOnBGDataset).__init__()
        self.background=background
        self.foregroundScale=foregroundScale
        self.shapeOrig=shape
        self.seqLength=seqLength
        self.blurIt=blurIt
        self.subpixel=subpixel
        self.minResultSpeed=minResultSpeed
        self.maxResultSpeed=maxResultSpeed
        self.square=square
        self.scale=int(scale)
        self.shape=int(shape*scale)
        self.foreground = foreground
        
        if self.foreground=="MNIST":
            self.MNIST=datasets.MNIST('data', train=not infrencePhase, download=True)
        elif self.foreground=="FMNIST":
            self.FMNIST=datasets.FashionMNIST('data', train=not infrencePhase, download=True)
        elif self.foreground=="grid":
            pass
        else:
            raise Exception("Wrong foregorund")
            
        if self.background=="STL10":
            self.STL10=datasets.STL10('data', split='train' if not infrencePhase else 'test', download=True,transform=transforms.Compose([transforms.Grayscale(1),transforms.Resize(self.shape)]))
            self.STL10Size=len(self.STL10)
        elif self.background=="random":
            pass
        else:
            raise Ecxeption("Wrong background")
    def _scaleBlur(self,arry):
        if(self.blurIt):
            arry=cv2.blur(arry[0,:,:], (self.scale,self.scale))[np.newaxis,:,:]
            
        if self.scale!=1:
            arry=cv2.resize(arry[0,:,:],(self.shapeOrig,self.shapeOrig),interpolation=cv2.INTER_NEAREST)[np.newaxis,:,:]
        return np.clip(arry, a_min = 0, a_max = 1)
    
    def _cImg(self,image,scale,original=False,square=True):
        if original == True:
            return image
        
        o = image.max(axis=0) > 0
        o_columns = np.where(np.any(o, axis=0))[0]
        o_rows = np.where(np.any(o, axis=1))[0]
        if not square:
            res= image[:,min(o_rows):max(o_rows) + 1,
                         min(o_columns):max(o_columns) + 1]
        else:
            maximum=max((max(o_rows) + 1-min(o_rows)),(max(o_columns) + 1-min(o_columns)))
            mino_row=max(min(o_rows)-((maximum-(max(o_rows)-min(o_rows)+ 1))//2),0)
            mino_col=max(min(o_columns)-((maximum-(max(o_columns)-min(o_columns)+ 1))//2),0)
            res= image[:,mino_row:mino_row+ maximum,
                 mino_col:mino_col + maximum]
        res=cv2.resize(res[0,:,:],(int(res.shape[2]*scale),int(res.shape[1]*scale)),interpolation=cv2.INTER_AREA)[np.newaxis,:,:]
        res=np.pad(res,((0,0),(4,4),(4,4)),"constant",constant_values=0)
        res=np.pad(res,((0,0),(1,1),(1,1)),"constant",constant_values=0.5)
        res=np.pad(res,((0,0),(1,1),(1,1)),"constant",constant_values=0.75)
        res=np.pad(res,((0,0),(1,1),(1,1)),"constant",constant_values=0.5)
        return res
    
    def _occluded(self,img,val=1,stride=8.,space=5):
        dim=len(img.shape)
        w=img.shape[dim-2]
        maxRange=int(w/stride)
        res=img.copy()
        for i in range(0,maxRange):
            maxStep=int(i*stride+stride+3)
            minStep=int(i*stride+3)
            if i%space==0 and maxStep<w:
                if(dim==5):
                    res[:,:,:,minStep:maxStep,:]=res[:,:,minStep:maxStep,:,:]=val
                elif (dim==4):
                    res[:,:,:,minStep:maxStep]=res[:,:,minStep:maxStep,:]=val
                elif (dim==3):
                    res[:,:,minStep:maxStep]=res[:,minStep:maxStep,:]=val
                elif (dim==2):
                    res[:,minStep:maxStep]=res[minStep:maxStep,:]=val
                else:
                    raise(BaseException("ERROR"))
        return res

    def _addGrid(self,img):
        img=self._occluded(img)
        return img
        
    def __len__(self):
        if self.foreground == "MNIST":
            return len(self.MNIST)
        elif self.foreground == "FMNIST":
            return len(self.FMNIST)
        else:
            return 60000

    def __getitem__(self, idx):
        
        if self.foreground == "MNIST":
            foreground_obj=self.MNIST.__getitem__(idx)
            img = np.array(foreground_obj[0])[np.newaxis,:,:]/255.
            img=self._cImg(img,self.scale*self.foregroundScale,False,square=self.square)
        elif self.foreground == "FMNIST":
            foreground_obj=self.FMNIST.__getitem__(idx)
            img = np.array(foreground_obj[0])[np.newaxis,:,:]/255.
            img=self._cImg(img,self.scale*self.foregroundScale,False,square=self.square)
        elif self.foreground == "grid":
            gridSize=74*self.scale*self.foregroundScale
            img=np.zeros((1,gridSize,gridSize), dtype=np.float32)
            foreground_obj=[img,'grid']
            img=self._addGrid(img)

        sign=np.random.choice([-1,1],size=(1,2))
        if self.subpixel:
            velocities=(np.random.randint(low=int(self.minResultSpeed*self.scale),high=(self.maxResultSpeed*self.scale)+1,size=(1, 2))* sign)
        else:
            velocities=(np.random.randint(low=int(self.minResultSpeed*self.scale),high=self.maxResultSpeed+1,size=(1, 2))* sign*self.scale)
        shape2=int(self.shape/2)
        positions = np.array([[shape2+(np.sign(-velocities[0,0])*(shape2-1-(img.shape[1]*(-velocities[0,0] > 0)))),
                               shape2+(np.sign(-velocities[0,1])*(shape2-1-(img.shape[2]*(-velocities[0,1] > 0))))]])
        if self.background=="random":
            bg=np.random.rand(1,self.shape,self.shape)
        elif self.background=="STL10":
            stl=self.STL10.__getitem__(idx%self.STL10Size)
            bg=1-(np.array(stl[0])[np.newaxis,:,:]/255.)
            # bg*=0.9
            # bg+=0.05
        else:
            bg=np.zeros(1,self.shape,self.shape)
        ResFrame = np.empty((1,self.seqLength,self.shapeOrig, self.shapeOrig), dtype=np.float32)
        ResFrameFG = np.empty((1,self.seqLength,self.shapeOrig, self.shapeOrig), dtype=np.float32)
        ResFrameAlpha = np.empty((1,self.seqLength,self.shapeOrig, self.shapeOrig), dtype=np.float32)
        ResFrameBG = np.empty((1,self.seqLength,self.shapeOrig, self.shapeOrig), dtype=np.float32)
        
        for frame_idx in range(self.seqLength):
            frame = np.zeros((1,self.shape, self.shape), dtype=np.float32)
            frameFG = np.zeros((1,self.shape, self.shape), dtype=np.float32)
            frameAlpha = np.zeros((1,self.shape, self.shape), dtype=np.float32)
            frameBG = np.zeros((1,self.shape, self.shape), dtype=np.float32)
            frameBG=bg
            frame+=bg
            ptmp = positions.copy()
            
            ptmp[0] += velocities[0]
            for dimen in range(2):
                if ptmp[0, dimen] < 0:
                    velocities[0, dimen] *= -1
                    raise Exception("Bounced")
                if ptmp[0, dimen] > self.shape - img.shape[dimen+1]:
                    velocities[0, dimen] *= -1
                    raise Exception("Bounced")

            positions[0] += velocities[0]
            digit_mat = np.zeros((self.shape, self.shape, 1))
            IN=[positions[0][0],
                      positions[0][0]
                      + img.shape[1],
                      positions[0][1],
                      positions[0][1]
                      + img.shape[2]]
            if self.foreground == "grid":
                mask=img>0
                np.place(frame[0,IN[0]:IN[1],IN[2]:IN[3]], mask, img[mask])
            else:
                frame[0,IN[0]:IN[1],IN[2]:IN[3]] = img
            frameFG[0,IN[0]:IN[1],IN[2]:IN[3]] = img
            frameAlpha[0,IN[0]:IN[1],IN[2]:IN[3]] = np.ones_like(img)
            
            ResFrame[0,frame_idx] = self._scaleBlur(frame)
            ResFrameFG[0,frame_idx] = self._scaleBlur(frameFG)
            ResFrameAlpha[0,frame_idx] = self._scaleBlur(frameAlpha)
            ResFrameBG[0,frame_idx] = self._scaleBlur(frameBG)
            del frame,frameFG,frameAlpha,frameBG
            
        #[batch * channel(# of channels of each image) * depth(# of frames) * height * width]   
        result = {'GT': ResFrame,'A': ResFrameAlpha,'BG': ResFrameBG,'FG': ResFrameFG, 'foreground': foreground_obj[1],"velocity":velocities/self.scale}
        return result

class timeit():
    from datetime import datetime
    def __enter__(self):
        self.tic = self.datetime.now()
    def __exit__(self, *args, **kwargs):
        print(Fore.GREEN+'Runtime: {}'.format(self.datetime.now() - self.tic)+Fore.RESET)

    
#([batch, colorchannel, seq, H, W])
def showSeq(normalize, step, caption, data, relDataArray=[], revert=False, oneD=False, dpi=1, save="", sideView=False,vmin=None, vmax=None,normType=matplotlib.colors.NoNorm(),verbose=True):
    '''
    Data should be: [batch, colorchannel, seq, H, W]
    '''
    if type(data) is torch.Tensor:
        data = data.detach().cpu().numpy()
    for i in range(len(relDataArray)):
        if type(relDataArray[i]) is torch.Tensor:
            relDataArray[i]=relDataArray[i].detach().cpu().numpy()
    
    if type(data) == np.ndarray:
        data=np.moveaxis(data, 1, -1)
        for i in range(len(relDataArray)):
            if type(relDataArray[i]) != np.ndarray:
                print("Not consistent data type")
                return
            else:
                relDataArray[i]=np.moveaxis(relDataArray[i], 1, -1)
    else:
        print("Undefined Type Error")
        return

    if verbose:
        print("Data: min and max",data.min(),data.max())
        for i in range(len(relDataArray)):
            print("Data[",i,"]: min and max",relDataArray[i].min(),relDataArray[i].max())
        
    if data.shape[3] == 1 and not oneD:
        dimsqrt = int(math.sqrt(data.shape[2]))
        if(dimsqrt*dimsqrt == data.shape[2]):
            data = data.reshape(
                (data.shape[0], data.shape[1], dimsqrt, dimsqrt, data.shape[4]))
        else:
            print("Error while reshaping")
            return

    #Normilize
    if(vmax==None and vmin==None):
        if normalize:
            maxAbsVal=max(abs(data.max()),abs(data.min()))+0.00000001
            if(len(relDataArray)>0):
                for i in range(len(relDataArray)):
                    maxAbsVal=max(maxAbsVal,max(abs(relDataArray[i].max()),abs(relDataArray[i].min())))
                    
            data=((data/maxAbsVal)/2)+0.5
            if(len(relDataArray)>0):
                for i in range(len(relDataArray)):
                    relDataArray[i]=((relDataArray[i]/maxAbsVal)/2)+0.5
            
        else:
            maxAbsVal=max(abs(data.max()),abs(data.min()))+0.00000001
            if(len(relDataArray)>0):
                for i in range(len(relDataArray)):
                    maxAbsVal=max(maxAbsVal,max(abs(relDataArray[i].max()),abs(relDataArray[i].min())))
            data=((data/maxAbsVal))
            if(len(relDataArray)>0):
                for i in range(len(relDataArray)):
                    relDataArray[i]=((relDataArray[i]/maxAbsVal))
                
    # Check validity of sideview
    sideView = sideView and oneD and len(relDataArray) > 0
    for i in range(len(relDataArray)):
        sideView = sideView and (data.shape == relDataArray[i].shape)

    mData = data

    if sideView:
        caption = caption+"(SideView)"
        div = 2+len(relDataArray)
        imgConcat = torch.ones(data.shape[0], (2+len(relDataArray))
                               * data.shape[1], data.shape[2], data.shape[3], data.shape[4])
        for cic in range(imgConcat.shape[1]):
            if cic % div == 0:
                imgConcat[:, cic, :, :, :] = data[:, cic//div, :, :, :]
            elif cic % div < div-1:
                imgConcat[:, cic, :, :, :] = relDataArray[(
                    cic % div)-1][:, cic//div, :, :, :]
        mData = imgConcat
    else:
        if(len(relDataArray) > 0):
            space = np.ones((data[0].shape[0], max(
                1, data[0].shape[1]//20), data[0].shape[2], data[0].shape[3]))
            for i in range(len(relDataArray)):
                if relDataArray[i].shape[3] == 1 and not oneD:
                    dimsqrt = int(math.sqrt(relDataArray[i].shape[2]))
                    if(dimsqrt*dimsqrt == relDataArray[i].shape[2]):
                        relDataArray[i] = relDataArray[i].reshape(
                            (relDataArray[i].shape[0], relDataArray[i].shape[1], dimsqrt, dimsqrt, relDataArray[i].shape[4]))
                    else:
                        print("Error while reshaping")
                        return
                if relDataArray[i].shape == data.shape:
                    mData = np.array(
                        [np.hstack([mData[j], space, relDataArray[i][j]]) for j in range(len(data))])

    data = mData
    display (Markdown('<b><span style="color: #ff0000">'+caption + (" (N)" if normalize else "") +
          (" (r)" if revert else "")+"</span></b>  "+(str(data.shape) if verbose else "")+''))

    cmap = 'gray'
    imgs_combArrey = []
    for b in range(data.shape[0]):
        if oneD:
            imgs_combArrey.append(np.vstack([data[b, i, :, :, 0].reshape(
                data.shape[2]*data.shape[3]) for i in range(data.shape[1])]))
        else:
            imgs_combArrey.append(np.hstack([data[b, i//2, :, :, 0] if i % 2 == 0 else np.ones(
                (data.shape[2], 1)) for i in range(data.shape[1]*2-1)]))
        if(b < data.shape[0]-1):
            imgs_combArrey.append(np.ones((1, imgs_combArrey[-1].shape[1])))

    finalImg = np.vstack(imgs_combArrey)

    if revert:
        if(vmax==None and vmin==None):
            if normalize: 
                mid = 0.5
            else:
                mid = (finalImg.max()-finalImg.min())/2
            finalImg = ((finalImg-mid)*-1)+mid
        else:
            mid = (vmax-vmin)/2
            finalImg = ((finalImg-mid)*-1)+mid
    dpi = 240.*dpi
    if verbose:
        print("Image: min and max",finalImg.min(),finalImg.max())
    tImg = torch.from_numpy(finalImg)
    plt.figure(figsize=None, dpi=dpi)
    plt.imshow(finalImg, cmap=cmap, norm=normType,vmin=vmin, vmax=vmax,aspect='equal')
    plt.axis('off')
#     plt.colorbar()
    plt.show()
    if save!="":
        plt.imsave(save+"/"+caption+str(step)+'.png',finalImg, cmap=cmap)
    return {"name":caption,"image_data":tImg}

def gaussian_noise(ins, is_training, std=0.005):
    if is_training and std > 0:
        noise = Variable(ins.data.new(ins.size()).normal_(mean=0, std=std))
        return ins + noise
    return ins


def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    inds = tuple(slice(None, None) if i != dim
                 else x.new(torch.arange(x.size(i)-1, -1, -1).tolist()).long()
                 for i in range(x.dim()))
    return x[inds]

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0,n,None) 
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n,None,None)
                  for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front],axis)


def complex_div(t1, t2, eps=1e-8):
    assert t1.size() == t2.size()
    assert t1.size()[-1] == 2
    assert t1.device==t2.device
    t1re = torch.index_select(t1, -1, torch.tensor([0], device=t1.device))
    t1im = torch.index_select(t1, -1, torch.tensor([1], device=t1.device))
    t2re = torch.index_select(t2, -1, torch.tensor([0], device=t1.device))
    t2im = torch.index_select(t2, -1, torch.tensor([1], device=t1.device))
    denominator = t2re**2 + t2im**2 + eps
    numeratorRe = t1re*t2re + t1im*t2im
    numeratorIm = t1im*t2re - t1re*t2im
    return torch.cat([numeratorRe/denominator, numeratorIm/denominator], -1)

def complex_mul(t1, t2):
    assert t1.size() == t2.size()
    assert t1.size()[-1] == 2
    assert t1.device==t2.device
    t1re = torch.index_select(t1, -1, torch.tensor([0], device=t1.device))
    t1im = torch.index_select(t1, -1, torch.tensor([1], device=t1.device))
    t2re = torch.index_select(t2, -1, torch.tensor([0], device=t1.device))
    t2im = torch.index_select(t2, -1, torch.tensor([1], device=t1.device))
    return torch.cat([t1re*t2re - t1im*t2im, t1re*t2im + t1im*t2re], -1)

def complex_conj(iT):
    assert iT.size()[-1] == 2
    iTre = torch.index_select(iT, -1, torch.tensor([0], device=iT.device))
    iTim = torch.index_select(iT, -1, torch.tensor([1], device=iT.device))
    return torch.cat([iTre, -iTim], -1)

def complex_abs(iT):
    assert iT.size()[-1] == 2
    iTre = torch.index_select(iT, -1, torch.tensor([0], device=iT.device))
    iTim = torch.index_select(iT, -1, torch.tensor([1], device=iT.device))
    outR = torch.sqrt(iTre**2 + iTim**2+ 1e-8)
    return torch.cat([outR, torch.zeros_like(outR)], -1)


def listToTensor(inp):
    return torch.stack(inp,dim=1).unsqueeze(1)

def manyListToTensor(inp):
    res=[]
    for i in inp:
        res.append(listToTensor(i))
    return res

    
def createarray(a1,a2,b1,b2,sp):
    return np.hstack((np.linspace(a1,a2,seq_length-sp,endpoint=False),np.linspace(b1,b2,sp,endpoint=False)))


def getItemIfList(inp,indx):
    if type(inp)==np.ndarray:
        return inp[indx]
    else:
        return inp

'''
B,H,W
'''
def fillWithGaussian(inp,sigma,offset=0):
    center = ((inp.shape[1]-2)+offset)/2.
    inp[:,:,:].fill_(0)
    for i_ in range(inp.shape[1]):
        for j_ in range(inp.shape[2]):
            inp[:,i_, j_]+= 1. / 2. / np.pi / (sigma ** 2.) * torch.exp(
            -1. / 2. * ((i_ - center - 0.5) ** 2. + (j_ - center - 0.5) ** 2.) / (sigma ** 2.))

def return2DGaussian(resolution,sigma,offset=0,normalized=False):
    kernel_size = resolution

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size).to(sigma.device).float()
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size).float()
    y_grid = x_grid.t().float()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = ((kernel_size - 1)+offset)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2.*variance)
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    if normalized:
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    return gaussian_kernel.view( 1, 1, kernel_size, kernel_size)

#[colorchannel,H, W]
def showSingleImg(img,colorbar=True):
    if type(img) is torch.Tensor:
        img = img.detach().cpu().numpy()
        
    if type(img) is np.ndarray:
        if len(img.shape)>3:
            raise Exception("showSingleImage is only for one image not batch!")
        elif len(img.shape)<2:
            raise Exception("showSingleImage is for show 2D images!")
        elif len(img.shape)==3:
            if img.shape[0]==1:
                img = img[0,:,:]
            elif img.shape[0]==3 or img.shape==4:
                img = np.moveaxis(img, 0, -1)
            else:
                raise Exception("wrong number of channels!")
        else:#img is 2D
            pass
    else:
        raise Exception("Unknown input type!")
    plt.imshow(img)
    if colorbar:
        plt.colorbar()
    plt.show()

def generateAllRot(dsAlpha,dsFG):
    return (getRot(dsAlpha[:,0,0,:,:],dsAlpha[:,0,1,:,:])
            ,getRot(dsFG[:,0,0,:,:],dsFG[:,0,1,:,:])
            ,getRot(dsAlpha[:,0,0,:,:]*dsFG[:,0,0,:,:],dsAlpha[:,0,1,:,:]*dsFG[:,0,1,:,:]))

def generateAllRotwithBG(dsAlpha,dsFG,dsBG):
    return (getRot(dsAlpha[:,0,0,:,:],dsAlpha[:,0,1,:,:])
            ,getRot(dsFG[:,0,0,:,:],dsFG[:,0,1,:,:])
            ,getRot(dsAlpha[:,0,0,:,:]*dsFG[:,0,0,:,:],dsAlpha[:,0,1,:,:]*dsFG[:,0,1,:,:]),getRot(dsBG[:,0,0,:,:],dsBG[:,0,1,:,:]))

def angleToRotM(inp,rotTFG):
    res= torch.zeros_like(rotTFG)
    res[:,0]=torch.cos(inp[:,0]).clamp(-1,1)
    res[:,1]=torch.sin(inp[:,1]).clamp(-1,1)
    return res
    
def getEnergy(inp0,inp1):
    s=torch.zeros_like(inp0,dtype=torch.float)
    hf0 = torch.stack((inp0,s),dim=3)
    hf0 = fft(hf0,2,normalized=normalizeFFT)
    hf1 = torch.stack((inp1,s),dim=3)
    hf1 = fft(hf1,2,normalized=normalizeFFT)
    R = complex_mul(hf0,complex_conj(hf1))
    return complex_abs(R)[:,:,:,0:1].repeat(1,1,1,2)

def getRot(inp0,inp1):
    hfRel1OrigAgri=getRotComplete(inp0,inp1)
    return hfRel1OrigAgri

def rotIt(inp,rotInp):
    hfP_fft = torch.stack((inp,torch.zeros_like(inp,dtype=torch.float)),dim=3)
    hfP_fft = fft(hfP_fft,2,normalized=normalizeFFT)
    rotInp_fft = fft(rotInp,2,normalized=normalizeFFT)
    hfN_fft=complex_mul(hfP_fft,complex_conj(rotInp_fft))
    return ifft(hfN_fft,2,normalized=normalizeFFT)[:,:,:,0]


def getRotComplete(inp0,inp1):
    s=torch.zeros_like(inp0,dtype=torch.float)
    hf0 = torch.stack((inp0,s),dim=3)
    hf0 = fft(hf0,2,normalized=normalizeFFT)
    hf1 = torch.stack((inp1,s),dim=3)
    hf1 = fft(hf1,2,normalized=normalizeFFT)
    R = complex_mul(hf0,complex_conj(hf1))
    R = complex_div(R,complex_abs(R))
    return R


def visshift(data):
    for dim in range(1, len(data.size())-1):
        data = roll_n(data, axis=dim, n=data.size(dim)//2)
    return data

def justshift(data):
    for dim in range(1, len(data.size())-1):
        data = roll_n(data, axis=dim, n=data.size(dim)//2)
    return data

def fftshift(data,dim,normalized):
    return justshift(fft(data,dim,normalized))

def ifftshift(data,dim,normalized):
    return ifft(justshift(data),dim,normalized)

def fft(data,dim,normalized):
    data=torch.fft(data,dim,normalized=normalized)
    return (data)

def ifft(data,dim,normalized):
    data=torch.ifft((data),dim,normalized=normalized)
    return data

def clmp(a,soft=False):
    if soft:
        return torch.sigmoid(-5+(a)*10)
    return a.clamp(0,1)

def showComplex(norm,step,name,a,justRI=False):
    tmp=visshift(a)[li:ui].permute(0,3,1,2).unsqueeze(1).cpu().detach()
    showSeq(norm,step,name+" R&I "+str(step),tmp[:,:,0:1,:,:],[tmp[:,:,1:2,:,:]],oneD=dimention==1,revert=True,dpi=0.6)
    if justRI:
        return
    angle=torch.atan2(tmp[:,:,1:2,:,:],tmp[:,:,0:1,:,:])
    absol=torch.sqrt(tmp[:,:,1:2,:,:]*tmp[:,:,1:2,:,:]+tmp[:,:,0:1,:,:]*tmp[:,:,0:1,:,:]+ 1e-8)
    showSeq(norm,step,name+" Abs "+str(step),absol,oneD=dimention==1,revert=True,dpi=0.3)
    showSeq(norm,step,name+" Angl "+str(step),angle,oneD=dimention==1,revert=True,dpi=0.3)
    

def showReal(norm,step,name,a,vmin=None,vmax=None):
    showSeq(norm,step,name+" "+str(step),(a)[li:ui].unsqueeze(1).unsqueeze(1).cpu().detach(),oneD=dimention==1,dpi=0.3,revert=True,vmin=vmin,vmax=vmax)

def getPhaseDiff(hf0,hf1):
    R = complex_mul(hf0,complex_conj(hf1))
    R = complex_div(R,complex_abs(R))
    return R

def smoothIt(rot,energy=None,gainEst=0.2,step=1,axis=0,show=False,dims=(1,2),direction=0):
    resolution=rot.shape[1]
    oAxis=1 if axis==0 else 0
    if direction<=0:
        a=getPhaseDiff(rot,rot.roll(-step,dims=dims[axis]))
    if direction>=0:
        ar=getPhaseDiff(rot,rot.roll(step,dims=dims[axis]))
    if(show):
        if direction<=0:
            showComplex(True,1,"a",a,True)
        if direction>=0:
            showComplex(True,1,"ar",ar,True)
    if energy is not None:
        if direction<=0:
            am=(a*energy.abs()).sum(dim=dims[oAxis])/(energy.abs().sum(dim=dims[oAxis])+1e-8)
        if direction>=0:
            amr=(ar*energy.abs()).sum(dim=dims[oAxis])/(energy.abs().sum(dim=dims[oAxis])+1e-8)
    else:
        if direction<=0:
            am=a.mean(dim=dims[oAxis])
        if direction>=0:
            amr=ar.mean(dim=dims[oAxis])
    if(show):
        if direction<=0:
            amshow=am.unsqueeze(2).repeat(1,1,resolution,1) if axis==0 else am.unsqueeze(1).repeat(1,resolution,1,1)
            showComplex(True,1,"am",amshow,True)
        if direction>=0:
            amrshow=amr.unsqueeze(2).repeat(1,1,resolution,1) if axis==0 else am.unsqueeze(1).repeat(1,resolution,1,1)
            showComplex(True,1,"amr",amrshow,True)
    if direction<=0:    
        amm=am.mean(dim=1).unsqueeze(1).unsqueeze(2).repeat(1,resolution,resolution,1)
    if direction>=0:
        ammr=amr.mean(dim=1).unsqueeze(1).unsqueeze(2).repeat(1,resolution,resolution,1)
    if(show):
        if direction<=0:
            showComplex(True,1,"amm",amm,True)
        if direction>=0:
            showComplex(True,1,"ammr",ammr,True)
    if direction<=0:
        newEs=complex_mul(rot.roll(-step,dims=dims[axis]),amm)
    if direction>=0:
        newEsr=complex_mul(rot.roll(step,dims=dims[axis]),ammr)
    assert not torch.isnan(rot).any()
    if direction<=0:
        assert not torch.isnan(newEs).any()
    if direction>=0:
        assert not torch.isnan(newEsr).any()
    if direction<0:
        rot=(1-gainEst)*rot+(gainEst)*newEs#+1e-8
    elif direction>0:
        rot=(1-gainEst)*rot+(gainEst)*newEsr#+1e-8
    else:
        rot=(1-gainEst)*rot+(gainEst/2.)*newEs+(gainEst/2.)*newEsr#+1e-8
    assert not torch.isnan(rot).any()
    rot = complex_div(rot,complex_abs(rot))
    assert not torch.isnan(rot).any()
    if(show):
        if direction<=0:
            showComplex(True,1,"newEs",newEs,True)
        if direction>=0:
            showComplex(True,1,"newEsr",newEsr,True)
    return rot




def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)