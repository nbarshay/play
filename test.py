import matplotlib.animation as animation
import numpy as np
from pylab import plt
import numpy as np
from math import cos, pi, sqrt, exp
import random
import matplotlib.pyplot as plt


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





locs = [ np.array([0.5, 0.5]), np.array([0.6, 0.6]) ]

n_loops = 16

for loop in range(n_loops):
    for i in range(8):
        x = ((i+1)%8)/8
        raw_amp = cos(2*pi*x) + cos(4*pi*x)  + cos(8*pi*x) + cos(16*pi*x) 
        #if i+1 == 0:
        #    raw_amp -= 2.0
        #elif i+1==2 or i+1==6:
        #    raw_amp += 3.0

        options = []

        for flip in [False, True]:
            amp = (raw_amp + 1/sqrt(2) + 0.5) / (4 + 1/sqrt(2) + 0.5)
            theta = (pi - 0.1) * amp * (-1.0 if flip else 1.0)

            c, s = np.cos(theta), np.sin(theta)
            rot = np.array(((c, -s), (s, c)))

            head, tail = locs[-2], locs[-1]

            options.append( rot.dot(tail - head) + tail)

        def getScore(new_loc):
            dist = np.linalg.norm(new_loc - np.array([0.5, 0.5]))
            score = exp(-dist)
            return score**4



        scores = list(map(getScore, options))
        idx = selectRandomAmongOptions(scores)
        #idx = np.argmax(scores)

        locs.append(options[idx])



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


def doCurrent():
    #hack 4x frames
    makeAnimation('test.mp4', doit2, (25*60*4*n_loops)//81, 512)
    



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

