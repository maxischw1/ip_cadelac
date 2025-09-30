# CaDeLaC

Source code for **Context-Aware Deep Lagrangian Networks for Model Predictive Control (CaDeLaC)**.

If you find this work useful, please consider citing:
```
@misc{schulze2025_cadelac,
      title={Context-Aware Deep Lagrangian Networks for Model Predictive Control}, 
      author={Lucas Schulze and Jan Peters and Oleg Arenz},
      year={2025},
      eprint={2506.15249},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.15249}, 
}
```

# Installation - Training and Simulation

### 1. Clone the Repository
```bash
git clone git@github.com:Schulze18/cadelac.git
cd cadelac
git submodule update --recursive --init
```

### 2. Set Up Conda Environment
```bash
conda env create -f cadelac_env.yml
conda activate cadelac
```

### 3. Install CaDeLaC as a python pkg and additional Dependencies
```bash
pip install -e .
pip install l4casadi==1.4.1 --no-build-isolation
```


### 4. Install Acados (v0.4.3)
Follow the [official installation guide](https://docs.acados.org/installation/):  
```bash
cd acados
mkdir -p build && cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j4
```

Install the Python interface:  
```bash
pip install -e acados/interfaces/acados_template
```

Add the following to your `.bashrc`:  
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"<acados_root>/lib"
export ACADOS_SOURCE_DIR="<acados_root>"
```

---

# Installation for Real Robot Experiments
### 1. Create a ROS Workspace
```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone git@github.com:Schulze18/cadelac.git
cd cadelac
git submodule update --recursive --init
```


### 2. Set Up Conda Environment with ROS
```bash
conda env create -f cadelac_ros_env.yml
conda activate cadelac
```

To ensure **Acados** and **Libfranka** use the same compiler, add the installed compiler (e.g., `gcc-12`) to your path (recommended in `.bashrc`):  
```bash
export CC=$HOME/miniconda3/envs/cadelac_ros/bin/x86_64-conda-linux-gnu-gcc
export CXX=$HOME/miniconda3/envs/cadelac_ros/bin/x86_64-conda-linux-gnu-g++
```

### 3. Install CaDeLaC as a python pkg and additional Dependencies
```bash
pip install -e .
pip install l4casadi==1.4.1 --no-build-isolation
```

### 4. Install Acados (v0.4.3)
Same procedure as in the *Training and Simulation* section.


### 5. Install Libfranka (0.13.3)
[Libfranka](https://github.com/frankaemika/libfranka) provides low-level control of Franka Emika research robots.
1. Complete **System Requirements** and **Installing dependencies** from the official repository.
2. Clone and build:  
   ```bash
   git clone --recurse-submodules https://github.com/frankarobotics/libfranka.git
   cd libfranka
   ```
3. Checkout to version 0.13.3: 
   ```bash
   git checkout 0.13.3
   git submodule update
   ```
4. Build and compile   
   ```bash
   mkdir build && cd build
   cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/openrobots/lib/cmake -DBUILD_TESTS=OFF ..
   make
   ```

### 6. Build the ROS Workspace
```bash
cd ~/catkin_ws
catkin_make -DCMAKE_BUILD_TYPE=Release -DFranka_DIR:PATH={PATH_TO_LIBFRANKA}/libfranka/build -j2
source devel/setup.bash
```

# Context-Aware DeLaN
All scripts related to training the proposed models are located in the `learning` folder.  
The training pipeline is adapted from [Deep Lagrangian Networks](https://github.com/milutter/deep_lagrangian_networks) and implemented in **PyTorch**.

## Datasets
The datasets used in this project is hosted on [Hugging Face Datasets](https://huggingface.co/datasets/schulze18/cadelac).

To download it directly into the repository, run:
```
git clone https://huggingface.co/datasets/schulze18/cadelac learning/data/datasets/panda
```

This dataset was originally generated with the script `sim/create_dataset.py`.  
If you prefer to create it yourself, simply run:
```
python3 -m cadelac.sim.create_dataset
```

## Training
To train the Context-Aware DeLaN on the dataset, run:
```
python3 -m cadelac.learning.train_panda
```