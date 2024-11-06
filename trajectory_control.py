import mujoco as mj
from mujoco.glfw import glfw
import matplotlib.pyplot as plt
import numpy as np
import os
import test

def doit():

    xml_path = '2D_double_pendulum.xml' #xml file (assumes this is in the same folder as this file)
    simend = 10 #simulation time
    print_camera_config = 0 #set to 1 to print camera config
                            #this is useful for initializing view of the model)


    my_model = test.TestModule3()
    test.doLearning2(my_model)


    q0_init,_ = my_model.getTrajectory(0, 0.0)
    q1_init,_ = my_model.getTrajectory(1, 0.0)
    q0_end,_ = my_model.getTrajectory(0, 1.0)
    q1_end,_ = my_model.getTrajectory(0, 1.0)

    t = []
    qact0 = []
    qref0 = []
    qact1 = []
    qref1 = []


    def controller(model, data):
        #put the controller here. This function is called inside the simulation.
        #pass
        nonlocal my_model

        time = data.time

        q_ref0, qdot_ref0 = my_model.getTrajectory(0, time)
        q_ref1, qdot_ref1 = my_model.getTrajectory(1, time)


        #model-based control (feedback linearization)
        #tau = M*(PD-control) + f
        M = np.zeros((2,2))
        mj.mj_fullM(model,M,data.qM)
        f0 = data.qfrc_bias[0]
        f1 = data.qfrc_bias[1]
        f = np.array([f0,f1])

        kp = 500
        kd = 2*np.sqrt(kp)
        pd_0 = -kp*(data.qpos[0]-q_ref0)-kd*(data.qvel[0]-qdot_ref0)
        pd_1 = -kp*(data.qpos[1]-q_ref1)-kd*(data.qvel[1]-qdot_ref1)
        pd_control = np.array([pd_0,pd_1])
        tau_M_pd_control = np.matmul(M,pd_control)
        tau = np.add(tau_M_pd_control,f)
        data.ctrl[0] = tau[0]
        data.ctrl[1] = tau[1]

        t.append(data.time)
        qact0.append(data.qpos[0])
        qref0.append(q_ref0)
        qact1.append(data.qpos[1])
        qref1.append(q_ref1)


    #get the full path
    dirname = os.path.dirname(__file__)
    abspath = os.path.join(dirname + "/" + xml_path)
    xml_path = abspath

    # MuJoCo data structures
    model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
    data = mj.MjData(model)                # MuJoCo data
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
    # cam.azimuth = 90
    # cam.elevation = -45
    # cam.distance = 2
    # cam.lookat = np.array([0.0, 0.0, 0])
    cam.azimuth = 90 ; cam.elevation = 5 ; cam.distance =  6
    cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])

    data.qpos[0] = q0_init
    data.qpos[1] = q1_init

    #initialize the controller
    init_controller(model,data)

    #set the controller
    mj.set_mjcb_control(controller)

    while not glfw.window_should_close(window):
        time_prev = data.time

        while (data.time - time_prev < 1.0/60.0):
            mj.mj_step(model, data)

        if (data.time>=simend):
            plt.figure(1)
            plt.subplot(2, 1, 1)
            # plt.plot(t,qact0,'r-')
            # plt.plot(t,qref0,'k');
            plt.plot(t,np.subtract(qref0,qact0),'k')
            plt.ylabel('error position joint 0');
            plt.subplot(2, 1, 2)
            # plt.plot(t,qact1,'r-')
            # plt.plot(t,qref1,'k');
            plt.plot(t,np.subtract(qref1,qact1),'k')
            plt.ylabel('error position joint 1');
            plt.show(block=False)
            plt.pause(10)
            plt.close()
            break;

        # get framebuffer viewport
        viewport_width, viewport_height = glfw.get_framebuffer_size(
            window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

        #print camera configuration (help to initialize the view)
        if (print_camera_config==1):
            print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
            print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

        # Update scene and render
        mj.mjv_updateScene(model, data, opt, None, cam,
                           mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)

        # swap OpenGL buffers (blocking call due to v-sync)
        glfw.swap_buffers(window)

        # process pending GUI events, call GLFW callbacks
        glfw.poll_events()

    glfw.terminate()



doit()
