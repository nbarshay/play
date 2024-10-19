import matplotlib.animation as animation
import numpy as np
from pylab import *
import numpy as np


def doit(frame, n_frames, n_pixels):
   return 0.0 + np.random.rand(n_pixels, n_pixels)*0.1 + float(frame)/n_frames*0.2

"""
- Filetype should be 'mp4'
- Always square
- Always greyscale
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

    ani = animation.FuncAnimation(fig,update_img,300,interval=40)
    writer = animation.writers['ffmpeg'](fps=25)

    ani.save(path, writer=writer,dpi=n_pixels)


makeAnimation('test.mp4', doit, 400, 512)


