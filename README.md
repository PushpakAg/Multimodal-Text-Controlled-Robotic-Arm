# llm_robotic_arm


```bashrc
cd llm_robotic_arm
catkin_make
source devel/setup.bash
roslaunch levelManager lego-world.launch
```
then new terminal

```bashrc
rosrun levelMnager levelManager.py -l 1
rosrun motion_planning motion_planning.py
```

```bashrc
rosrun vision lego-vision.py
```

