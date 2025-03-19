# VLM Benchmarks on the Koch v1.1 Manipulator
## Introduction
This repository aims to reproduce the results of recent publications that use vision-language models (VLMs) for robot manipulation tasks on low-cost DIY manipulators. The goal is to create a centralized hub for VLM-based manipulator projects, enabling rapid testing and benchmarking. I chose the [Koch v1.1](https://github.com/jess-moss/koch-v1-1) manipulator to start, due to its compatibility with [lerobot](https://github.com/huggingface/lerobot).

Note: The koch v1-1 has only 5DoF, which may be limiting for more complex experiments. For future projects, I would recommend a low-cost 6DoF robot (ex. [Simple Automation](https://docs.google.com/spreadsheets/d/1i-t-i7dLayyafxtfTy8_VctcmbbnCp6Mays1JUR9Qg4/edit?gid=47726668#gid=47726668)).

## Koch v1.1 Manipulator
Please follow the build instructions found on the [original repository](https://github.com/jess-moss/koch-v1-1?tab=readme-ov-file#assembly-instructions). Additionally, follow the [lerobot example](https://github.com/huggingface/lerobot/blob/main/examples/7_get_started_with_real_robot.md) for running the code.

To simplify the forward and inverse kinematics, I set ![LaTeX Equation](https://latex.codecogs.com/svg.image?\theta_4=\pi/2). This is good enough to achieve most pick-and-place tasks.

### DH Table
| Joint | a (Link Length) | α (Twist) | d (Offset) | θ (Joint Angle) | Joint Limits (rad) |
|-------|----------------|-----------|------------|----------------|---------------------|
| 1     | ![LaTeX Equation](https://latex.codecogs.com/svg.image?0)             | ![LaTeX Equation](https://latex.codecogs.com/svg.image?-\frac{\pi}{2})      | ![LaTeX Equation](https://latex.codecogs.com/svg.image?d_1=5.5)        | ![LaTeX Equation](https://latex.codecogs.com/svg.image?\theta_1)            | ![LaTeX Equation](https://latex.codecogs.com/svg.image?\left[-\frac{\pi}{2},\frac{\pi}{2}\right])        |
| 2     | ![LaTeX Equation](https://latex.codecogs.com/svg.image?a_2=10.68)         | ![LaTeX Equation](https://latex.codecogs.com/svg.image?0)         | ![LaTeX Equation](https://latex.codecogs.com/svg.image?0)          | ![LaTeX Equation](https://latex.codecogs.com/svg.image?\theta_2)            | ![LaTeX Equation](https://latex.codecogs.com/svg.image?\left[0,\frac{\pi}{2}\right])           |
| 3     | ![LaTeX Equation](https://latex.codecogs.com/svg.image?a_3=10)           | ![LaTeX Equation](https://latex.codecogs.com/svg.image?0)         | ![LaTeX Equation](https://latex.codecogs.com/svg.image?0)          | ![LaTeX Equation](https://latex.codecogs.com/svg.image?\theta_3)            | ![LaTeX Equation](https://latex.codecogs.com/svg.image?\left[-\frac{\pi}{2},\frac{\pi}{2}\right])        |
| 4     | ![LaTeX Equation](https://latex.codecogs.com/svg.image?0)             | ![LaTeX Equation](https://latex.codecogs.com/svg.image?\frac{\pi}{2})       | ![LaTeX Equation](https://latex.codecogs.com/svg.image?0)          | ![LaTeX Equation](https://latex.codecogs.com/svg.image?\frac{\pi}{2})            | ![LaTeX Equation](https://latex.codecogs.com/svg.image?\left[-\frac{\pi}{2},\frac{\pi}{2}\right])        |
| 5     | ![LaTeX Equation](https://latex.codecogs.com/svg.image?0)            | ![LaTeX Equation](https://latex.codecogs.com/svg.image?0)         | ![LaTeX Equation](https://latex.codecogs.com/svg.image?d_5=10.5)       | ![LaTeX Equation](https://latex.codecogs.com/svg.image?\theta_5)            | ![LaTeX Equation](https://latex.codecogs.com/svg.image?\left[-\pi,\pi\right])            |


### Inverse Kinematics
<img src="images/inv_kin.png" alt="inv_kin">

### Experiment Setup
For all experiments, a single ZED mini stereo camera was positioned across from the Koch v1.1 manipulator, ensuring that it had a clear view of the manipulator's workspace.

The Perspective-n-Point (PnP) pose computation (`cv2.solvePnP`) was used to calculate the rotation and translation matrices between the camera frame and the robot/world frame. A blue object, held by the robot's end-effector, was tracked across the image to obtain pixel coordinates. The corresponding world coordinates were derived using inverse kinematics. See video below:

https://github.com/user-attachments/assets/acdfefd9-b190-459c-8ed2-c6aaa87b24a0

## Demonstrations

### [ReKep](https://rekep-robot.github.io/)

#### Experiment 1: Eraser into Tape
https://github.com/user-attachments/assets/99434062-c455-40b1-b682-657e4cad514d

#### Experiment 2: Chess
https://github.com/user-attachments/assets/3496987d-d6bc-4b77-9a39-7f86e06efc25

#### Experiment 3: Block Stacking
https://github.com/user-attachments/assets/f0b55d51-8e67-4500-9fd6-8f15c189fb1c


### [Pi0](https://github.com/Physical-Intelligence/openpi)
TBD
