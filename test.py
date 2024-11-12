import matplotlib.animation as animation
import numpy as np
from pylab import plt
import numpy as np
from math import cos, pi, sqrt, exp, log, pi
import random
import matplotlib.pyplot as plt
import scipy

import torch
from torch import nn

#from ABC import ABC, abstractmethod


import os
import sys
import itertools
from itertools import chain
import argparse
from collections import OrderedDict, defaultdict, namedtuple, deque
from functools import total_ordering, partial
import re
from copy import copy, deepcopy
import types
import math
import operator
import numpy as np
import subprocess
import glob
import shutil
import json
import csv

from armsim import ArmSim, ArmSimReturn

#------------------------------------
"""
# Project level Todo:

## General improvements
- Have Armsim start a litle early and end a litle late
- Try different timesteps/rk method and test against gold standard to see how far we can push it




"""


#-------------------------------------

class Dataframe(object):
    def __init__(self, cols=None, n=None):
        object.__setattr__(self, '_df_n', n)
        object.__setattr__(self, '_df_cols', OrderedDict())
        if cols is not None:
            for k,v in cols.items():
                self._addCol(k, v)

    @classmethod
    def fromRows(cls, rows):
        cols = [(k, np.array([x[k] for x in rows])) for k in funnel(x.keys() for x in rows)]
        return cls(dCreate(cols, t=OrderedDict), n=len(rows))

    ##################

    def __setattr__(self, k, v): self._addCol(k, v)
    def __getattr__(self, k): return self._getCol(k)
    def __setitem__(self, k, v): self._addCol(k,v)
    def __getitem__(self, k):
        if isinstance(k, (slice, np.ndarray)):
            return Dataframe(OrderedDict((a, b[k]) for a,b in self.items()))
        else:
            return self._getCol(k)

    def __delitem__(self, k):
        del self._df_cols[k]

    def __len__(self):
        assert self._df_n is not None
        return self._df_n

    def keys(self): return self._df_cols.keys()
    def values(self): return self._df_cols.values()
    def items(self): return self._df_cols.items()

    def __iter__(self): return self.keys()
    def contains(self, k): return k in self.keys()

    def __copy__(self):
        return Dataframe(OrderedDict(self.items()))
    def __deepcopy__(self, _memo):
        return Dataframe(OrderedDict((k, v.copy()) for k,v in self.items()))

    #########################
    def _addCol(self, k, v):
        assert isinstance(k, str)
        assert isinstance(v, np.ndarray) and (v.ndim == 1)
        if self._df_n is not None:
            assert v.size == self._df_n
        else:
            object.__setattr__(self, '_df_n', v.size)
        self._df_cols[k] = v

    def _getCol(self, k):
        return self._df_cols[k]

    def filter(self, mask):
        mask = np.array(mask)
        new_cols = {}
        for k,v in self.items():
            new_cols[k] = v[mask]
        return Dataframe(new_cols)
        
    #####################
    @classmethod
    def concat(cls, xs):
        res = cls()
        xs = list(xs)
        if xs:
            cols = funnel(x.keys() for x in xs)
            for col in cols:
                temp = [x[col] for x in xs]
                funnel(x.dtype for x in temp)
                res[col] = np.concatenate(temp)
        return res

    ###############
    def safeHdf5(self, path):
        with h5py.File(path, 'w') as f:
            for k,v in self.items():
                data = v.astype(np.string_) if v.dtype == np.object_ or v.dtype.type is np.unicode_ else v
                f.create_dataset(k, data=data)

    @staticmethod
    def loadHdf5(path):
        with h5py.File(path, 'r') as f:
            cols = OrderedDict([(k, v[:]) for k,v in f.iteritems()])
            n = funnel([v.shape[0] for v in f.itervalues()]) if len(f) else 0
            return Dataframe(cols, n)

    @staticmethod
    def loadNpz(path):
        raw = np.load(path)
        return Dataframe(dict(raw))











#-----------------------------------


def doit(frame, n_frames, n_pixels):
   return 0.0 + np.random.rand(n_pixels, n_pixels)*0.1 + float(frame)/n_frames*0.2

"""
- Filetype should be 'mp4'
- Always square
- Always greyscale, 0.0 is white, 1.0 is black
- Filetype should be mp4
"""
def makeAnimation(path, get_frame, n_frames, n_pixels):
    fig = plt.figure(figsize=(1, 1))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    im = ax.imshow(np.zeros((n_pixels, n_pixels)), cmap="gray", vmin=0.0, vmax=1.0)


    def update_img(n):
        im.set_data(1.0 - get_frame(n, n_frames, n_pixels) )
        return im

    ani = animation.FuncAnimation(fig,update_img,n_frames,interval=40)
    writer = animation.writers['ffmpeg'](fps=25)

    ani.save(path, writer=writer,dpi=n_pixels)


#makeAnimation('test.mp4', doit, 25*8*4*1, 512)

#------------------------


def selectRandomAmongOptions(values):
    values = np.array(values).astype(np.float64)
    assert np.all(values >= 0.0)
    assert values.ndim == 1
    tot = np.sum(values)
    p = values/tot if tot > 0.0 else np.ones(len(values)/len(values))
    return np.random.choice(len(values),  p=p)








def doCurrent(beats, beats_sec_per_idx):
    fps = 25 #hardcoded into makeAnimation

    sec_per_frame = 1.0/fps

    secs = len(beats)*beats_sec_per_idx

    locs = [ np.array([0.5, 0.5]) ]

    direc = np.array([1.0, 0.0])
    last_beat = 0

    const = 0.5

    for i in range(1, len(beats)):
        if beats[i] > 0:
            time = (i - last_beat)*beats_sec_per_idx

            loc = locs[-1] + direc*time*const

            options = []

            for flip in [False, True]:
                amp = beats[i]
                assert amp <= 1.0
                theta = (pi - 0.1) * amp * (-1.0 if flip else 1.0)

                c, s = np.cos(theta), np.sin(theta)
                rot = np.array(((c, -s), (s, c)))

                head, tail = locs[-1], loc

                cand = rot.dot(tail - head)

                options.append( cand/np.linalg.norm(cand)  )

            def getScore(cand):
                dist = np.linalg.norm(loc + cand - np.array([0.5, 0.5]))
                score = exp(-dist)
                return score**4



            scores = list(map(getScore, options))
            idx = selectRandomAmongOptions(scores)

            direc = options[idx]
            locs.append(loc)


            last_beat = i

    locs.append(locs[-1] + direc*(len(beats)-last_beat)*beats_sec_per_idx*const)

    print(locs)


    def doit2(frame, n_frames, n_pixels):

        def getLoc(raw, off=0):
            return max(min(int(raw*n_pixels)+off, n_pixels-1), 0)

        def drawPt(vec, x_off=0, y_off=0, weight=1.0):
                def getLoc(raw, off):
                        raw = (raw%1.0 + 1.0)%1.0
                        return max(min(int(raw*n_pixels)+off, n_pixels-1), 0)
                
                img[getLoc(vec[0], x_off), getLoc(vec[1], y_off)] += weight


        def drawDot(loc):
            for x_off in range(-1, 2):
                for y_off in range(-1, 2):
                    drawPt(loc, x_off, y_off)



        def drawLine(x, y, n_pts, weight):
            vec = y-x 
            for i in range(n_pts):
                drawPt(x+vec*((i+0.5)/n_pts), weight=weight)
                


        img = np.zeros((n_pixels, n_pixels))

        loop_frames = n_frames
        
        #segments = np.array([[0.5, 0.9], [0.40, 0.5], [0.5, 0.1], [0.6, 0.5]]*4)
        segments = np.array(locs)

        distances = np.sqrt(np.sum((segments[1:,:] - segments[:-1,:])**2, axis=1))
        total_distance = np.sum(distances)

        distance_acc = 0.0
        for i in range(0, len(segments)-1):
            head, tail = segments[i], segments[i+1]
            next_dist = np.linalg.norm(tail-head)

            frame_center = distance_acc / total_distance * n_frames

            alpha = (16 - abs(frame - frame_center))/16.0

            if i > 0 and alpha >= 0:
                a, b, c = segments[i-1], segments[i], segments[i+1]
                direc = (b-a) / np.linalg.norm(b-a) + (c-b) / np.linalg.norm(c-b)
                direc /= np.linalg.norm(direc)
                drawLine(b, b+direc*0.01, 10, alpha)
                drawLine(b, b-direc*0.01, 10, alpha)


            distance_acc += next_dist
            

        
        goal_dist = (frame / loop_frames) * total_distance

        distance_acc = 0.0
        for i in range(len(segments)-1):
            head, tail = segments[i], segments[i+1]
            next_dist = np.linalg.norm(tail-head)
            if goal_dist < distance_acc + next_dist:
                alpha = ((goal_dist - distance_acc) / next_dist)
                #alpha2 = copysign (  (((alpha - 0.5)*2)  ** 2) / 2.0, alpha-0.5) + 0.5
                alpha2 = alpha
                loc = (tail-head)*alpha2 + head

                drawDot(loc)

                break
            else:
                distance_acc += next_dist

        return img
            

    #hack 4x frames
    makeAnimation('test.mp4', doit2, int(secs*fps), 512)
    
"""
returns (array, sec_per_idx)
"""
def getTotalSoundPower(data, rate):
    from scipy import fft

    # Spectrogram estimation:
    N = 256
    T = []
    for k in range(0, data.shape[0]+1, N):
        x = fft.fftshift(fft.fft(data[k:k+N,0], n=N))[N//2:N]
        # assert np.allclose(np.imag(x*np.conj(x)), 0)
        raw_px = np.real(x*np.conj(x))
        T.append(10*np.log10(np.sum(raw_px)))

    return np.array(T), N/rate


def getEma(a, half_life_per_unit, units_per_idx=1):
    half_life_per_idx = half_life_per_unit/units_per_idx
    alpha = exp(log(0.5)/half_life_per_idx)

    n, = a.shape
    res = np.empty_like(a)

    weight = 0.0
    acc = 0.0

    for i in range(n):
        weight += 1.0
        acc += a[i]
        res[i] = acc/weight
        weight *= alpha
        acc *= alpha


    return np.array(res)

def testGetEma():
    assert np.array_equal( getEma(np.array([1.0, 0.0, 0.0]), 1), np.array([1.0, 1/3.0, 1/7.0]) )


def makeSinWavelet(width_units, units_per_idx):
    return np.sin(np.arange(width_units / units_per_idx) * 2*pi / (width_units/units_per_idx))

def myConvolve(signal, a):
    ret = np.convolve(np.flip(a), signal)[len(a)-1:] 
    assert len(ret) == len(signal)
    return ret

def specGram(rate, data):
    from scipy import fft

    assert rate == 44100


    # Spectrogram estimation:
    N = 256
    S = []
    T = []
    T2 = []
    for k in range(0, data.shape[0]+1, N):
        x = fft.fftshift(fft.fft(data[k:k+N,0], n=N))[N//2:N]
        # assert np.allclose(np.imag(x*np.conj(x)), 0)
        raw_px = np.real(x*np.conj(x))
        Pxx = 10*np.log10(raw_px)
        S.append(Pxx)
        T.append(10*np.log10(np.sum(raw_px[35:70])))
        T2.append(10*np.log10(np.sum(raw_px[:5])))
    S = np.array(S)

    # Frequencies:
    f = fft.fftshift(fft.fftfreq(N, d=1/rate))[N//2:N]
    # array([    0. ,   187.5,   375. , ..., 23625. , 23812.5])

    # Spectrogram rendering:
    n_beats = (data.shape[0] / rate) * (64 / 24) 

    fig, axs = plt.subplots(3)

    axs[0].imshow(S.T, origin='lower', extent = [0, n_beats, 0, S.shape[1]], aspect='auto')
    

    axs[1].plot(np.linspace(0, n_beats, len(T)), T)
    axs[1].margins(x=0, y=0)

    axs[2].plot(np.linspace(0, n_beats, len(T2)), T2)
    axs[2].margins(x=0, y=0)




    plt.show()


#==============================================


class BrainNode(object):
    #default=None will assert if anyone tries to use it
    def __init__(self, default=None):
        self.prev_output = None #this better not matter
        self.cur_output = default
        self.tick = None
        self.prev_tick = None

        self.working = False

    def yourTurn(self, framework):
        tick = framework.getTick()
        if self.tick == tick:
            return

        self.prev_output = self.cur_output
        self.prev_tick = tick #mark prev as ready

        assert not self.working
        self.working = True
        self.cur_output = self.crunch(framework)
        self.working = False

        self.tick = tick



    def getCurOutput(self, framework):
        self.yourTurn(framework)
        return self.cur_output

    def getPrevOutput(self, framework):
        assert self.prev_tick == framework.getTick()
        assert self.prev_output is not None
        return self.prev_output



class BrainNodeMulDecay(BrainNode):
    def __init__(self, in_node, half_life_per_idx, initial):
        self.in_node = in_node 
        self.alpha = exp(log(0.5)/half_life_per_idx)
        BrainNode.__init__(self, initial)
        
    def crunch(self, framework):
        return (1.0-self.alpha)*self.in_node.getCurOutput(framework) + self.alpha*self.getPrevOutput(framework)

class BrainNodeMaxDecay(BrainNode):
    def __init__(self, in_node, half_life_per_idx):
        self.in_node = in_node 
        self.alpha = exp(log(0.5)/half_life_per_idx)
        BrainNode.__init__(self, -np.inf)
        
    def crunch(self, framework):
        return max(self.in_node.getCurOutput(framework), self.alpha*self.getPrevOutput(framework))

class BrainNodeDelay(BrainNode):
    def __init__(self, in_node, default):
        self.in_node = in_node 
        self.next = default
        BrainNode.__init__(self)

    def crunch(self, framework):
        ret = self.next
        self.next = self.in_node.getCurOutput(framework)
        return ret
    
class BrainNodeLambda(BrainNode):
    def __init__(self, in_nodes, lambd):
        self.lambd = lambd
        self.in_nodes = in_nodes 
        BrainNode.__init__(self)
        
    def crunch(self, framework):
        return self.lambd( *[x.getCurOutput(framework) for x in self.in_nodes] )

class BrainNodeInput(BrainNode):
    def __init__(self, input_key):
        self.input_key = input_key
        BrainNode.__init__(self)

    def crunch(self, framework):
        return framework.getInput(self.input_key)




        
#Hardcoded to FrameworkDf (only kind for now)
class Framework(object):
    def __init__(self, df):
        self.df = df
        self.idx = 0 

    def getInput(self, input_key):
        return self.df[input_key][self.idx]

    def advance(self):
        self.idx += 1
        if self.idx == len(self.df):
            return False
        else:
            return True

    def getTick(self):
        return self.idx


def evalNetwork(node, framework):
    ret = [] 
    while True:
        ret.append(node.getCurOutput(framework))
        if not framework.advance():
            break
    return np.array(ret)
     

def evalNetworkDf(node, df):
    return evalNetwork(node, Framework(df))


def makeBasicNetwork(powers, sec_per_idx):
    in_node = BrainNodeInput('a')
    recent = BrainNodeMulDecay(in_node, 0.2 / sec_per_idx, np.mean(powers))
    spike = BrainNodeLambda( [in_node, recent], lambda cur, rec: max(cur-rec, 0.0))
    out_norm = BrainNodeMaxDecay(spike, 10.0 / sec_per_idx)
    return BrainNodeLambda([spike, out_norm], lambda a, b:  a / b if b>0.0 and a/b>0.25 else 0.0)


"""
wishlist:
- normalization based on loooong max
- how to deal with a signal that keeps growing for multiple samples?  wait?
"""
def makeNetwork(powers, sec_per_idx):

    in_node = BrainNodeInput('a')
    recent = BrainNodeMulDecay(in_node, 0.2 / sec_per_idx, np.mean(powers))
    spike = BrainNodeLambda( [in_node, recent], lambda cur, rec: max(cur-rec, 0.0))
    recent_max_spike = BrainNodeMaxDecay(spike, 0.05 / sec_per_idx)
    raw_out = BrainNodeLambda( [spike, BrainNodeDelay(BrainNodeLambda([recent_max_spike], lambda x: 2.5*x), 0.0)], lambda cur, rec: cur if cur > rec else 0.0)

    out_norm = BrainNodeMaxDecay(raw_out, 10.0 / sec_per_idx)

    return BraanNodeLambda([raw_out, out_norm], lambda a, b:  a / b if b>0.0 and a/b>0.25 else 0.0)
    

def getRotMatrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([(1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0,  c, -s), (0.0, 0.0, s, c)])

def getTransMatrix(x):
    return np.array([(1.0, 0.0, x, 0.0), (0, 1.0, 0.0, x), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)])

def genArm(angles):
    lens = [0.3, 0.25, 0.05]
    mat = np.array([0.5, 0.0, 0.0, 1.0])
    res = []
    for i in range(3):
        mat = getRotMatrix(angles[i]).dot(mat)
        res.append(mat[:2])
        print(mat)
        mat = getTransMatrix(lens[i]).dot(mat)
    res.append(mat[:2])
    return np.array(res)


def drawArm(angles):
    arm = genArm(angles)

    ret = np.zeros((n_pixels, n_pixels))

    im = ax.imshow(np.zeros((n_pixels, n_pixels)), cmap="gray", vmin=0.0, vmax=1.0)



class DanceCreature(object):
    def getDefaultParams(self, params):
        ...

    def getNewParams(self, auditory_slice, old_params):
        ...
    
    def renderParams(self, ax, params):
        ...





#-------------------------------------------


def gatherStuff():
    rate, data = scipy.io.wavfile.read('out_there.wav')
    powers, sec_per_idx = getTotalSoundPower(data[rate*10:rate*40], rate)
    spikes = evalNetworkDf(makeBasicNetwork(powers, sec_per_idx), Dataframe({'a': powers}))
    return rate, data, powers, sec_per_idx, spikes

"""
- len(return) = radius*2
- wavelength is in indices
- phase_shift is 0 to 1

TODO ensure center is at radius (both envelope and base wavelet)
"""
def getLocalWavelet(radius, wavelength, phase_shift, do_envelope):
    raw_x = torch.arange(2*radius)
    x = raw_x * 2*pi/wavelength + phase_shift*2*pi
    for i in range(2):
        x = x + torch.sin(x)
    ret = torch.cos( x ) + 1.0

    if do_envelope:
        ret = ret * (radius - torch.abs(radius-raw_x)) / radius

    return ret




class TestModule(nn.Module):
    def __init__(self, hits, sec_per_idx) -> None:
        super().__init__()
        self.hits = torch.tensor(hits)

        self.n = len(hits)

        self.sec_per_idx = sec_per_idx

        self.radius = int(4.0 / sec_per_idx) #TODO hack

        self.base_wl = 1.0 / sec_per_idx #60bpm

        self.param_n = self.n // self.radius

        self.wls = nn.Parameter(torch.ones(self.param_n) * self.base_wl)

        self.pss = nn.Parameter(torch.ones(self.param_n) * 10.0)      #start at 10 for nice normalization


        """
        self.hits = torch.tensor(hits)
        self.sec_per_idx = sec_per_idx

        init = np.zeros(len(self.hits))
        init[0] = 1.0

        self.params = nn.Parameter(torch.tensor(init))
        """

        print(f'{sec_per_idx=} {self.radius=} {self.base_wl=} {self.param_n=}')


    def getCombo(self):
        cur_n = self.n + 2*self.radius
        res = torch.zeros(cur_n)

        for i in range(self.param_n):
            wavelet = getLocalWavelet(self.radius, self.wls[i], self.pss[i], True)
            begin = i*self.radius
            end = (i+2)*self.radius
            #res[begin:end] += wavelet
            res = res + torch.nn.functional.pad(wavelet, (begin, cur_n - begin - 2*self.radius))
        return res[:self.n]
        
    def forward(self, verbose=False):

        spike_reward = torch.sum(self.getCombo() * self.hits) / (torch.mean(self.getCombo()) * torch.mean(self.hits) * self.n)

        return -spike_reward


def doLearningOld(model):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    for i in range(10000):
        loss = model()
        if i%1000==0:
                print(loss.item())
                model(verbose=True)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()




def getLocalWavelet2(radius, wavelength, phase_shift, do_envelope):
    raw_x = np.arange(2*radius)
    x = raw_x * 2*pi/wavelength + phase_shift*2*pi
    for i in range(2):
        x = x + np.sin(x)
    ret = np.cos( x ) + 1.0

    if do_envelope:
        ret = ret * (radius - np.abs(radius-raw_x)) / radius

    return ret





class TestModule2(object):
    def __init__(self, hits, sec_per_idx) -> None:
        self.hits = hits

        self.n = len(hits)

        self.sec_per_idx = sec_per_idx

        self.radius = int(4.0 / sec_per_idx) #TODO hack

        self.base_wl = 1.0 / sec_per_idx #60bpm

        self.param_n = self.n // self.radius

        self.params = np.empty((2, self.param_n))

        self.wls = self.params[0, :]
        self.wls[:] = self.base_wl

        self.pss = self.params[1, :]
        self.pss[:] = 10.0  #start at 10 for nice normalization

        print(f'{sec_per_idx=} {self.radius=} {self.base_wl=} {self.param_n=}')

    def getCombo(self):
        cur_n = self.n + 2*self.radius
        res = np.zeros(cur_n)

        for i in range(self.param_n):
            wavelet = getLocalWavelet2(self.radius, self.wls[i], self.pss[i], True)
            begin = i*self.radius
            end = (i+2)*self.radius
            res[begin:end] += wavelet
        return res[:self.n]
        
    def forward(self, verbose=False):
        spike_reward = np.sum(self.getCombo() * self.hits) / (np.mean(self.getCombo()) * np.mean(self.hits) * self.n)

        return -spike_reward

#maxiter x100
def optimize(state, critic, maxiter=100, verbose=False):
    params_shape = state.params.shape    
    eval_cnt = 0

    def setParams(upd_x):
        state.params[:] = upd_x.reshape(params_shape)

    def doForward(verbose=False):
        return critic.forward(state, verbose)

    doForward() #dry run for errors


    def fun(x):
        nonlocal eval_cnt

        if eval_cnt%1000==0 and verbose:
            fwd = doForward(verbose=True)
            print(f'{eval_cnt=} {fwd=}')

        eval_cnt += 1
        
        setParams(x)
        return doForward()
       
    x0 = state.params.reshape(-1)
    ret = scipy.optimize.minimize(fun, x0, options={'maxiter':maxiter})

    setParams(ret.x)
    assert(ret.fun == doForward())

    print(f'{eval_cnt=} {ret.fun=}')


#----------------------

"""
assume time runs from 0 to 1
NOTE: only works with width in (0,1]
"""
#TODO think about if we want to keep letting center be whatever it wants
def getPart(center, width, scale, ts):
    center = math.fmod(center, 1.0)
    center = center + 1.0 if center < 0.0 else center

    x = (np.where(ts > center, ts - center, ts - center + 1.0)/width)

    inside = ( 3*(x**2) - 2*(x**3) ) * (2*pi)

    not_prime = ( np.sin( inside ) ) * scale

    prime = ( np.cos( inside ) * ( (6*x - 6*(x**2)) * (2*pi) ) ) * scale/width

    def maskit(xx):
        return np.where( (x > 0.0) & (x < 1.0), xx, 0.0)

    return np.stack( (maskit(not_prime), maskit(prime)), axis=-1)

def getPenalty(raw_vals, a_min, a_max):
    vals = (raw_vals/(a_max-a_min)) - a_min
    return np.square(np.maximum(np.maximum(vals - 1.0, 0.0 - vals), 0.0))*1e6

class FastTrajectory(object):
    def __init__(self, base_traj):
        self.base_traj = base_traj 
        
        self.n = 500
        self.mem = np.empty((self.getNJoint(), self.n+1, 2)) #2 for f and fp

        ts = self._getT(np.arange(self.n+1))
        
        for i in range(self.getNJoint()):
            self.mem[i,:,:] = self.base_traj.getTrajectories(i, ts)

    def _getT(self, idx):
        return (idx/self.n)*self.base_traj.getLoopSec()

    
    def getLoopSec(self):
        return self.base_traj.getLoopSec()

    def getNJoint(self):
        return self.base_traj.getNJoint()

    def getRunSecs(self):
        return self.base_traj.getRunSecs()

    def getTrajectory(self, joint_idx, t):
        idx = int( (t/self.base_traj.getLoopSec()) * self.n )
        assert idx >= 0 and idx < self.n
        a, b = self.mem[joint_idx, idx, :], self.mem[joint_idx, idx+1,:]
        a_t, b_t = self._getT(idx), self._getT(idx+1)
        alpha = (t-a_t) / (b_t-a_t)

        return b*alpha + a*(1.0-alpha)




class BasicTrajectory(object):
    def __init__(self, seed, loop_sec=2.0, n_wavelet=10, n_joint=2) -> None:
        self.loop_sec=loop_sec
        self.n_wavelet = n_wavelet
        self.n_joint = n_joint

        np.random.seed(seed)
        self.params = np.random.uniform(size=(self.n_joint, self.n_wavelet, 3))
        self.params[:,:,2] = self.params[:,:,2] * 2.0 - 1.0 #scale in (-1.0, 1.0)
        self.params[:,:,1] = self.params[:,:,1] * 0.9 + 0.1 #width in (0.1, 1.0) 

    def getLoopSec(self):
        return self.loop_sec

    def getNJoint(self):
        return self.n_joint

    def getRunSecs(self):
        return self.loop_sec

    @classmethod
    def genStill(cls):
        ret = cls()
        ret.params[:,:,:2] = 1.0
        ret.params[:,:,2] = 0.0 
        return ret

    @classmethod
    def genBasic(cls):
        ret = cls.genStill()
        ret.params[0,0,:] = (0.0, 1.0, 1.0)
        ret.params[1,0,:] = (0.0, 1.0, 1.0)
        return ret


    def plot(self):
        fig, axs = plt.subplots(self.n_joint)
        t = np.linspace(0.0, self.getLoopSec(), 500)
        for i in range(self.n_joint):
            tra = self.getTrajectories(i, t)
            axs[i].plot(t, tra[:,0], color='tab:blue')
            axs[i].twinx().plot(t, tra[:,1], color='tab:red')


    """
    Intention is an output beween -1 and 1
    TODO: add this to optimization
    """
    def getCombo(self, joint_idx, ts):
        res = np.zeros( (len(ts), 2) )

        for i in range(self.n_wavelet):
            res += getPart(*self.params[joint_idx, i,:], ts)

        return res
         

    """
    Intention is an output in (-pi/2, pi/2)
    """
    def getTrajectories(self, joint_idx, ts):
        input_t = np.fmod(ts/self.loop_sec, 1.0)

        res = self.getCombo(joint_idx, input_t)
        res *= pi/2.0
        res[:,1] /= self.loop_sec

        return res

    def getTrajectory(self, joint_idx, t):
        bla = self.getTrajectories(joint_idx, np.array([t]))
        return bla[0][0], bla[0][1]


class TestCritic(object):
    def __init__(self, only_basic=False, use_fast_traj=True, pool=None):
        self.only_basic = only_basic
        self.use_fast_traj = use_fast_traj

    def forward(self, state, verbose=False):
        def getBasicPenalty(jidx):
            max_val = np.max(np.abs(state.getCombo(jidx, np.linspace(0, 1.0, 500))[:,0]))
            max_val_p = np.square(np.maximum(max_val-1.0, 0.0))*1e6

            if verbose:
                print(f'\t{jidx=} {max_val=} {max_val_p=}')

            return max_val_p

        param_shape_penalties = getPenalty(state.params[:,:,1], 0.025, 1.0)  #TODO think about this minimum
        param_shape_penalty = np.sum(param_shape_penalties)
        if verbose:
            print(f'\t{param_shape_penalty=}')
            
        #TODO
        score = getBasicPenalty(0) + getBasicPenalty(1) + param_shape_penalty

        if self.only_basic:

            return score

        fast_traj = FastTrajectory(state) if self.use_fast_traj else state

        ret = ArmSim.run(fast_traj)
        ctrl_l1 = np.mean(np.abs(ret.ctrls))
        #ctrl_l1_penalty = (ctrl_l1 - 10.0)**2
        ctrl_l1_penalty = ctrl_l1**2

        ctrl_l2 = float(np.mean(np.square(ret.ctrls)))
        
        vels = ret.getTipVels()
        vel_raw = float(np.sqrt(vels[len(vels)//4]) + np.sqrt(vels[len(vels)*3//4]))
        vel_reward = float((vel_raw*10.0)**2)

        pos_a, pos_b = float(ret.qs[0,len(vels)//4]), float(ret.qs[0,len(vels)*3//4])
        pos_penalty = float((pos_a - pi/4)**2) + float((pos_b + pi/4)**2)

        pos_aa, pos_bb = float(ret.qs[1,len(vels)//4]), float(ret.qs[1,len(vels)*3//4])
        pos_penalty2 = float((pos_aa)**2) + float((pos_bb)**2)

        off = np.mean((ret.qs[1]*10)**2)
        off2 = np.max(ret.tips[:,0]) - np.min(ret.tips[:, 0])


        if verbose:
            print(f'{ctrl_l1=}  {vel_raw=} {vel_reward=} {pos_a=} {pos_b=} {ctrl_l2=} {pos_penalty=} {pos_bb=} {pos_aa=} {pos_penalty2=} {off=} {off2=}')

        #score += ctrl_l1_penalty
        score += ctrl_l2
        #score += pos_penalty*1000
        #score += pos_penalty2*1000
        score += ((2.5-off2)**2)*1000
        #score -= vel_reward

        if verbose:
            print(f'{score=}')

        return score

 
