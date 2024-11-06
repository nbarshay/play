# General notes:
    - bodies have parent bodies | geoms,joints,sites,cameras,lights are contained in bodies

# usefold docs:
    - https://mujoco.readthedocs.io/en/stable/XMLreference.html
    - https://mujoco.readthedocs.io/en/stable/python.html (ctrlf: Basic Usage)


# abstract camera
    - azimuth: This attribute specifies the initial azimuth of the free camera around the vertical z-axis, in degrees. A value of 0 corresponds to looking in the positive x direction, while the default value of 90 corresponds to looking in the positive y direction. 
    - elevation: This attribute specifies the initial elevation of the free camera with respect to the lookat point. Note that since this is a rotation around a vector parallel to the camera’s X-axis (right in pixel space), negative numbers correspond to moving the camera up from the horizontal plane, and vice-versa.
    - az, el = 90-eps, -(90-eps) is looking "straight" down the z axis from "distance" units up.  On film: +y is up, +x is left



