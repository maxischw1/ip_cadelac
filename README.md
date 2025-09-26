# cadelac

# Installation

## Install the package
`
pip install -e .
`

## Set Up Conda Environment
Create and activate the conda environment:  
`
conda env create -f cadelac_env.yml
conda activate cadelac
`

Install l4casadi  
`
pip install l4casadi==1.4.1 --no-build-isolation
`



### Install Acados
(Original installation instructions)[https://docs.acados.org/installation/].
```
git clone https://github.com/acados/acados.git
cd acados
git submodule update --recursive --init
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON ..
# add more optional arguments e.g. -DACADOS_WITH_OSQP=OFF/ON -DACADOS_INSTALL_DIR=<path_to_acados_installation_folder> above
make install -j4
```


Install Acados Interface
```
pip install -e acados/interfaces/acados_template
```

Add to .bashrc
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"<acados_root>/lib"
export ACADOS_SOURCE_DIR="<acados_root>"
```


### libfranka install
Add installed compiler (gcc-12) to the path:
```
export CC=/home/${USER}/miniconda3/envs/${ENV_NAME}/bin/x86_64-conda-linux-gnu-gcc
export CX=/home/${USER}/miniconda3/envs/${ENV_NAME}/bin/x86_64-conda-linux-gnu-g++
```
Install libfrank 0.13.3

Compile franka_ros
catkin_make -DCMAKE_BUILD_TYPE=Release -DFranka_DIR:PATH={PATH_TO_LIBFRANKA}/libfranka/build


roslaunch franka_example_controllers effort_joint_controller.launch robot_ip:=172.16.0.2 load_gripper:=true robot:=panda

PPFLAGS
acados commit:
0d03b8570

v0.4.3