import matplotlib.animation as animation
import numpy as np
from pylab import *
import numpy as np


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





def doit2(frame, n_frames, n_pixels):

    def getLoc(raw, off=0):
        return max(min(int(raw*n_pixels)+off, n_pixels-1), 0)

    def drawPt(vec, x_off=0, y_off=0, weight=1.0):
            def getLoc(raw, off):
                    return max(min(int(raw*n_pixels)+off, n_pixels-1), 0)
            
            img[getLoc(vec[0], x_off), getLoc(vec[1], y_off)] += weight


    def drawDot(loc):
        for x_off in range(-2, 3):
            for y_off in range(-2, 3):
                drawPt(loc, x_off, y_off)



    def drawLine(x, y, n_pts, weight):
        vec = y-x 
        for i in range(n_pts):
            drawPt(x+vec*((i+0.5)/n_pts), weight=weight)
            


    img = np.zeros((n_pixels, n_pixels))

    loop_frames = n_frames
    
    segments = np.array([[0.5, 0.9], [0.40, 0.5], [0.5, 0.1], [0.6, 0.5]]*4)

    distances = np.sqrt(np.sum((segments[1:,:] - segments[:-1,:])**2, axis=1))
    total_distance = np.sum(distances)

    distance_acc = 0.0
    for i in range(0, len(segments)-1):
        head, tail = segments[i], segments[i+1]
        next_dist = np.linalg.norm(tail-head)

        frame_center = distance_acc / total_distance * n_frames

        alpha = 4 - abs(frame - frame_center) 

        if i > 0 and alpha >= 0:
            a, b, c = segments[i-1], segments[i], segments[i+1]
            direc = (b-a) / np.linalg.norm(b-a) + (c-b) / np.linalg.norm(c-b)
            direc /= np.linalg.norm(direc)
            drawLine(b, b+direc*0.02, 10, alpha)
            drawLine(b, b-direc*0.02, 10, alpha)


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

bpm = 85


makeAnimation('test.mp4', doit2, (25*60*2*4)//85, 512)
    




