# ROS integration for Franka Emika research robots

[![CI](https://github.com/frankaemika/franka_ros/actions/workflows/ci.yml/badge.svg)](https://github.com/frankaemika/franka_ros/actions/workflows/ci.yml)


See the [Franka Control Interface (FCI) documentation][fci-docs] for more information.

## License

All packages of `franka_ros` are licensed under the [Apache 2.0 license][apache-2.0].

[apache-2.0]: https://www.apache.org/licenses/LICENSE-2.0.html
[fci-docs]: https://frankaemika.github.io/docs


## Disclaimer
This repository is **not** an official release of `franka_ros`.  
It is a copy of the upstream package with additional experimental controllers.
The code may diverge from the official repository and **does not automatically include upstream updates**.  

# Visualization
roslaunch franka_visualization franka_visualization.launch robot_ip:=172.16.0.2 load_gripper:=True robot:=panda

# RVIZ
roslaunch franka_example_controlers joint_impedance_example_controller.launch robot_ip:=172.16.0.2 robot:=panda
