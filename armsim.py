import mujoco as mj
from mujoco.glfw import glfw
import os
from pylab import plt
import time

import numpy as np

class ArmSimReturn(object):
    def __init__(self, n_joint, ts, qs, q_refs, qdots, qdot_refs, ctrls, tips):
        self.n_joint = n_joint
        self.ts = ts
        self.qs = qs
        self.q_refs = q_refs
        self.qdots = qdots
        self.qdot_refs = qdot_refs
        self.ctrls = ctrls

        self.tips = tips

    def getTipVels(self):
        padded = np.pad(self.tips, [(1,1), (0,0)], mode='edge')
        vels = np.sqrt(np.sum(np.square(padded[2:] - padded[:-2]), axis=1))
        return vels

    def plot(self):
        fig, axs = plt.subplots(self.n_joint+1)
        for i in range(self.n_joint):
            axs[i].plot(self.ts, self.qs[i,:], color='tab:blue')
            axs[i].twinx().plot(self.ts, self.qdots[i,:], color='tab:red')
            axs[i].twinx().plot(self.ts, self.ctrls[i,:], color='tab:olive')

        for i in range(3):
            axs[-1].plot(self.ts, self.tips[:,i])
        axs[-1].twinx().plot(self.ts, self.getTipVels(), color='tab:red')

    def printStats(self):
        for j in range(self.n_joint):
            ctrl_l1 = float(np.mean(np.abs(self.ctrls[j])))
            q_l2_error = float(np.mean(np.square(self.q_refs[j] - self.qs[j])))
            qdot_l2_error = float(np.mean(np.square(self.qdot_refs[j] - self.qdots[j])))
            print(f'{j=} {ctrl_l1=} {q_l2_error=} {qdot_l2_error=}')






class ArmSim(object):


    @classmethod
    def run(cls, traj, show=False):
        n_joint = traj.getNJoint()
        assert(n_joint == 2)

        xml_path = '2D_double_pendulum.xml' #xml file (assumes this is in the same folder as this file)

        #get the full path
        dirname = os.path.dirname(__file__)
        abspath = os.path.join(dirname + "/" + xml_path)
        xml_path = abspath

        # MuJoCo data structures
        model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
        data = mj.MjData(model)                # MuJoCo data
        





        run_secs = traj.getRunSecs()
        if(show):
            run_secs *= 10.0

        ts = []

        qs = [[] for i in range(n_joint)]
        q_refs = [[] for i in range(n_joint)]

        qdots = [[] for i in range(n_joint)]
        qdot_refs = [[] for i in range(n_joint)]

        ctrls = [[] for i in range(n_joint)]
        
        tips = []

        if show:
            cam = mj.MjvCamera()                        # Abstract camera
            opt = mj.MjvOption()                        # visualization options

            # Init GLFW, create window, make OpenGL context current, request v-sync
            glfw.init()
            window = glfw.create_window(1200, 900, "Demo", None, None)
            glfw.make_context_current(window)
            glfw.swap_interval(1)

            # initialize visualization data structures
            mj.mjv_defaultCamera(cam)
            mj.mjv_defaultOption(opt)
            scene = mj.MjvScene(model, maxgeom=10000)
            context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)


            # Example on how to set camera configuration
            cam.azimuth = 90 ; cam.elevation = 5 ; cam.distance =  6
            cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])

        for i in range(n_joint):
            data.qpos[i], data.qvel[i] = traj.getTrajectory(i, 0.0)
            data.qvel[i] = 0.0


        if show:
            start_wall = time.time()

        #for use in loop
        q_ref = np.empty(n_joint)
        qdot_ref = np.empty(n_joint)

        while data.time < run_secs:

            time_prev = data.time

            while (data.time - time_prev < 1.0/60.0 and data.time < run_secs):
                mj.mj_step1(model, data)

                for i in range(n_joint):
                    q_ref[i], qdot_ref[i] = traj.getTrajectory(i, data.time)


                #model-based control (feedback linearization)
                #tau = M*(PD-control) + f
                M = np.zeros((n_joint,n_joint))
                mj.mj_fullM(model,M,data.qM)
                f = np.array([data.qfrc_bias[i] for i in range(n_joint)])

                kp = 500
                kd = 2*np.sqrt(kp)
                pd_control = np.array([-kp*(data.qpos[i]-q_ref[i])-kd*(data.qvel[i]-qdot_ref[i]) for i in range(n_joint)])
                tau_M_pd_control = np.matmul(M,pd_control)
                tau = np.add(tau_M_pd_control,f)
                
                data.ctrl[:n_joint] = tau

                ts.append(data.time)
                for i in range(n_joint):
                    qs[i].append(data.qpos[i])
                    q_refs[i].append(q_ref[i])
                    qdots[i].append(data.qvel[i])
                    qdot_refs[i].append(qdot_ref[i])
                    ctrls[i].append(data.ctrl[i])
                tips.append(data.site_xpos[0].copy())

                mj.mj_step2(model, data)


            if show:
                while(time.time() - start_wall < data.time):
                    pass

                viewport_width, viewport_height = glfw.get_framebuffer_size(
                    window)
                viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

                # Update scene and render
                mj.mjv_updateScene(model, data, opt, None, cam,
                                   mj.mjtCatBit.mjCAT_ALL.value, scene)
                mj.mjr_render(viewport, scene, context)

                # swap OpenGL buffers (blocking call due to v-sync)
                glfw.swap_buffers(window)

                # process pending GUI events, call GLFW callbacks
                glfw.poll_events()

        if show:
            glfw.terminate()


        return ArmSimReturn(n_joint, np.array(ts), np.array(qs), np.array(q_refs), np.array(qdots), np.array(qdot_refs), np.array(ctrls), np.array(tips))



