import matplotlib.animation as animation
import numpy as np
from pylab import *
import numpy as np

pixels = 24

def doit():
   return np.random.rand(pixels, pixels)/0.5


fig = plt.figure(figsize=(1, 1))
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
im = ax.imshow(doit(), cmap="gray")


def update_img(n):
    im.set_data(doit())
    return im

ani = animation.FuncAnimation(fig,update_img,300,interval=40)
writer = animation.writers['ffmpeg'](fps=25)

ani.save('demo.mp4',writer=writer,dpi=pixels)


