# CaDeLaC

Source code for **Context-Aware Deep Lagrangian Networks for Model Predictive Control (CaDeLaC)**.

If you find this work useful, please consider citing:
```
@inproceedings{schulze2025contextawaredelan,
  author={Schulze, Lucas and Peters, Jan and Arenz, Oleg},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Context-Aware Deep Lagrangian Networks for Model Predictive Control}, 
  year={2025},
  volume={},
  number={},
  pages={6939-6946},
  keywords={},
  doi={10.1109/IROS60139.2025.11246292}
}
```

For experiment videos, check the [project website](https://schulze18.github.io/cadelac_website/).

# Installation - Training and Simulation

### Training and Simulation Setup
1. Clone the Repository
   ```bash
   git clone git@github.com:Schulze18/cadelac.git
   cd cadelac
   git submodule update --recursive --init
   ```

2. Set Up Conda Environment
   ```bash
   conda env create -f cadelac_env.yml
   conda activate cadelac
   ```

3. Install CaDeLaC as a Python pkg and additional Dependencies
   ```bash
   pip install -e .
   pip install l4casadi==1.4.1 --no-build-isolation
   ```


4. Install Acados (v0.4.3)
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

   Add the following lines to your `.bashrc` file:  
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"<acados_root>/lib"
   export ACADOS_SOURCE_DIR="<acados_root>"
   ```

---

### Real Robot Setup
1. Create a ROS Workspace
   ```bash
   mkdir -p ~/catkin_ws/src
   cd ~/catkin_ws/src
   git clone git@github.com:Schulze18/cadelac.git
   cd cadelac
   git submodule update --recursive --init
   ```


2. Set Up Conda with ROS
   ```bash
   conda env create -f cadelac_ros_env.yml
   conda activate cadelac_ros
   ```

   To ensure **Acados** and **Libfranka** use the same compiler, add the installed compiler (`gcc-12`) to your `PATH` (recommended in `.bashrc`):
   ```bash
   export CC=$HOME/miniconda3/envs/cadelac_ros/bin/x86_64-conda-linux-gnu-gcc
   export CXX=$HOME/miniconda3/envs/cadelac_ros/bin/x86_64-conda-linux-gnu-g++
   ```

3. **Install CaDeLaC and dependencies**
   ```bash
   pip install -e .
   pip install l4casadi==1.4.1 --no-build-isolation
   ```

4. **Install Acados (v0.4.3)**  
   Follow the step 4 from **Training and Simulation Setup**.


5. **Install Libfranka (v0.13.3)**  
   [Libfranka](https://github.com/frankaemika/libfranka) provides low-level control for Franka Emika research robots.

   ```bash
   git clone --recurse-submodules https://github.com/frankarobotics/libfranka.git
   cd libfranka
   git checkout 0.13.3
   git submodule update
   mkdir build && cd build
   cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/openrobots/lib/cmake -DBUILD_TESTS=OFF ..
   make
   ```

6. **Build the ROS workspace**
   ```bash
   cd ~/catkin_ws
   catkin_make -DPYTHON_EXECUTABLE=$(which python) -DCMAKE_BUILD_TYPE=Release -DFranka_DIR:PATH={PATH_TO_LIBFRANKA}/libfranka/build -j2
   source devel/setup.bash
   ```

## Datasets

The datasets used for training are available on [Hugging Face](https://huggingface.co/datasets/schulze18/cadelac):

```bash
git clone https://huggingface.co/datasets/schulze18/cadelac cadelac/learning/datasets/
```

## Training and Evaluation
All scripts related to training the proposed models are located in the `learning` folder.  
The training pipeline is adapted from [Deep Lagrangian Networks](https://github.com/milutter/deep_lagrangian_networks) and implemented in **PyTorch**.


### Evaluate Pretrained Model (IROS 2025)
```bash
python -m cadelac.learning.train_panda -l 2
```

### Learn Residual Dynamics
```bash
python -m cadelac.learning.train_panda -l 0
```
> **Note:**  
Due to the dataset size, data loading may take a few minutes. During training, each epoch took around 30s on an NVIDIA GeForce RTX 4080. In the paper, we trained for 3000 epochs, but 1000 epochs already achieve similar performance.

### Evaluate Trained Model
```bash
python -m cadelac.learning.train_panda -l 1
```

### Learn Robot Model with DeLaN
```bash
python -m cadelac.learning.train_panda -l 0 -f 1
```

### Evaluate DeLaN
```bash
python -m cadelac.learning.train_panda -l 1 -f 1
```

---

## Control Experiments

### Mujoco Simulation

To evaluate joint tracking under random loads:
```bash
python -m cadelac.control.eval_controllers_multiple_envs -c 2
```

Controller options:
- `-c 0`: Nominal MPC
- `-c 1`: MPC + EKF
- `-c 2`: CaDeLaC (Context-Aware MPC)

---

### Real Franka Robot

**Terminal 1:** Launch hardware interface
```bash
conda activate cadelac_ros
source catkin_ws/devel/setup.bas
roslaunch franka_example_controllers effort_joint_controller.launch robot_ip:={ROBOT_IP} load_gripper:=true robot:=panda
```

**Terminal 2:** Run CaDeLaC controller
```bash
conda activate cadelac_ros
source catkin_ws/devel/setup.bash 
cd catkin_ws/src/cadelac/
python -m cadelac.ros.cadelac_node -c 2
```

> **Note:**  
As Acados needs to compile the controller in the first run, which will take several minutes for CaDeLaC, the safety flag [compilation_run](https://github.com/Schulze18/cadelac/blob/main/cadelac/ros/cadelac_node.py#L63) prevents the controller to be executed and holds the robot in place. Once the compilation is done, you can stop the script, set the flag to `False` and run again the script which will load the compiled controller. If you run a new controller for the first time, set the flag to `True` again.

---
## TODO
- [ ] Additional implementation details
- [ ] Dataset collection scripts

## Exoskeleton Extension: Script Structure and Usage

This fork extends CaDeLaC with a simplified **2-DOF hip-knee exoskeleton setup** for Context-Aware DeLaN training and evaluation.

The original CaDeLaC training pipeline remains centered around:

```text
cadelac/learning/train_panda.py
```

Additional helper scripts for dataset preparation, training shortcuts, and evaluation are organized under the `scripts/` directory.

---

### Script Directory Structure

```text
scripts/
├── data/
│   ├── make_exo_pkl.py
│   ├── make_all_exo_pkls.py
│   └── fix_exo_pkl_time.py
│
├── training/
│   ├── train_exo_context_current_config.sh
│   └── eval_exo_context_current_config.sh
│
└── evaluation/
    ├── evaluate_context_checkpoints.py
    ├── plot_context_checkpoint_metrics.py
    ├── plot_zoomed_context_torque_prediction.py
    └── plot_left_leg_torque_grid.py
```

General usage from the repository root:

```bash
cd ~/code/ip_cadelac
conda activate cadelac
```

---

## 1. Dataset Preparation Scripts

Dataset preparation scripts are stored in:

```text
scripts/data/
```

These scripts convert processed exoskeleton CSV data into the `.pkl` format expected by the CaDeLaC/DeLaN training pipeline.

Generated datasets are written to:

```text
cadelac/learning/datasets/panda/
```

---

### `scripts/data/make_exo_pkl.py`

Creates a simple **single left-leg 2-DOF exoskeleton dataset** from two local CSV files:

```text
~/Downloads/Exo.csv
~/Downloads/Joint_Moments_Filt.csv
```

The script extracts left hip and knee angles, velocities, and joint moments. Angles are converted from degrees to radians, accelerations are computed from filtered velocities, and the trajectory is split into segments.

Output:

```text
cadelac/learning/datasets/panda/exo_hip_knee_delan_2dof_left_only.pkl
```

Run:

```bash
python scripts/data/make_exo_pkl.py
```

Use this script mainly for quick single-file tests or debugging the dataset conversion pipeline.

---

### `scripts/data/make_all_exo_pkls.py`

Creates the main **left- and right-leg all-trials exoskeleton datasets** from a full processed data folder.

Expected input folder:

```text
~/Downloads/codeocean_exo_data
```

The script searches recursively for matching:

```text
Exo.csv
Joint_Moments_Filt.csv
```

It creates segmented datasets for both sides:

```text
cadelac/learning/datasets/panda/exo_hip_knee_delan_2dof_left_all_trials.pkl
cadelac/learning/datasets/panda/exo_hip_knee_delan_2dof_right_all_trials.pkl
```

It also creates Context-Aware dataset variants with the `_context.pkl` suffix.

For the simplified Context-Aware setup, the filtered joint moment signals from `Joint_Moments_Filt.csv` are used as the torque target `tau`.

In this setup, the Context-Aware residual target is set directly to this torque target:

```text
diff_tau = tau
```
```bash
tau_cols = ["hip_flexion_l_moment", "knee_angle_l_moment"]
```


Therefore, the Context-Aware training dataset is:

```text
cadelac/learning/datasets/panda/exo_hip_knee_delan_2dof_left_all_trials_context.pkl
```

Run:

```bash
python scripts/data/make_all_exo_pkls.py
```

This is the main dataset generation script for the current exoskeleton experiments.

---

### `scripts/data/fix_exo_pkl_time.py`

Fixes the time axis of an already generated exoskeleton `.pkl` dataset.

Current target file:

```text
cadelac/learning/datasets/panda/exo_hip_knee_delan_2dof_left_all_trials.pkl
```

The script creates a backup first:

```text
exo_hip_knee_delan_2dof_left_all_trials.before_time_fix.pkl
```

Then it enforces a uniform timestep of:

```text
dt = 0.005  # 200 Hz
```

and recomputes joint accelerations from `qv`.

Run:

```bash
python scripts/data/fix_exo_pkl_time.py
```

Use this script only when the generated dataset has non-uniform or inconsistent timestep information.

---

## 2. Training Scripts

Training scripts are stored in:

```text
scripts/training/
```

They are thin shell wrappers around:

```text
cadelac/learning/train_panda.py
```

They do not define a separate training configuration. Instead, they use the current configuration inside `train_panda.py`.

Experiment parameters such as `max_epoch`, `hist_length`, network sizes, and dataset names remain centralized in the main training file.

---

### `scripts/training/train_exo_context_current_config.sh`

Starts a new Context-Aware exoskeleton training run with the currently configured settings in `train_panda.py`.

Internally, it runs:

```bash
python -u -m cadelac.learning.train_panda \
  -l 0 \
  -f 0 \
  -m 1 \
  -r 0 \
  -c 0
```

Meaning:

```text
-l 0  do not load an existing model; start training from scratch
-f 0  use the Context-Aware/residual branch
-m 1  save the trained model
-r 0  do not render plots during training
-c 0  run on CPU
```

Run:

```bash
bash scripts/training/train_exo_context_current_config.sh
```

Log output:

```text
logs/exo_context_current_config_train.log
```

---

### `scripts/training/eval_exo_context_current_config.sh`

Loads and evaluates a saved Context-Aware exoskeleton model using the currently configured settings in `train_panda.py`.

Internally, it runs:

```bash
python -u -m cadelac.learning.train_panda \
  -l 1 \
  -f 0 \
  -m 0 \
  -r 0 \
  -c 0
```

Meaning:

```text
-l 1  load a saved model
-f 0  use the Context-Aware/residual branch
-m 0  do not save a new model
-r 0  do not render additional figures
-c 0  run on CPU
```

Run:

```bash
bash scripts/training/eval_exo_context_current_config.sh
```

Log output:

```text
logs/exo_context_current_config_eval.log
```

---

## 3. Evaluation Scripts

Evaluation scripts are stored in:

```text
scripts/evaluation/
```

These scripts evaluate saved Context-Aware DeLaN models, compare predicted torque against measured torque, and generate plots and metric CSV files.

The main Context-Aware dataset expected by the evaluation scripts is:

```text
cadelac/learning/datasets/panda/exo_hip_knee_delan_2dof_left_all_trials_context.pkl
```

The expected model folder is:

```text
cadelac/learning/trained_models/res_model/panda/ContextAware/
```

Generated plots and metrics are written to:

```text
logs/
```

---

### `scripts/evaluation/evaluate_context_checkpoints.py`

Evaluates all available Context-Aware checkpoints and final models on the BT24 test split.

It searches for models in:

```text
cadelac/learning/trained_models/res_model/panda/ContextAware/
cadelac/learning/trained_models/res_model/panda/ContextAware/checkpoint/
```

It computes torque prediction metrics and writes them to:

```text
logs/exo_context_checkpoint_metrics.csv
```

Run:

```bash
python scripts/evaluation/evaluate_context_checkpoints.py
```

Use this script to compare model performance across checkpoints or training epochs.

---

### `scripts/evaluation/plot_context_checkpoint_metrics.py`

Plots checkpoint-level metrics from:

```text
logs/exo_context_checkpoint_metrics.csv
```

It creates:

```text
logs/exo_context_torque_mse_over_epochs.png
logs/exo_context_torque_rmse_over_epochs.png
```

Run:

```bash
python scripts/evaluation/plot_context_checkpoint_metrics.py
```

Use this after running:

```bash
python scripts/evaluation/evaluate_context_checkpoints.py
```

---

### `scripts/evaluation/plot_zoomed_context_torque_prediction.py`

Creates a detailed torque prediction plot comparing:

```text
measured torque
Context-Aware DeLaN predicted torque
```

The plot shows Joint 0 and Joint 1 torque over time and reports:

```text
total Torque MSE
total Torque RMSE
Joint 0 MSE/RMSE
Joint 1 MSE/RMSE
```

The script also saves a small metrics CSV next to the generated plot.

Set a model path:

```bash
MODEL="cadelac/learning/trained_models/res_model/panda/ContextAware/epochs_3000exo_hip_knee_delan_2dof_left_all_trials_context.torch"
```

Run on the full BT24 test split:

```bash
python scripts/evaluation/plot_zoomed_context_torque_prediction.py \
  --model "$MODEL" \
  --output-dir logs/torque_zoom
```

Run only for `ball_toss` segments:

```bash
python scripts/evaluation/plot_zoomed_context_torque_prediction.py \
  --model "$MODEL" \
  --segment ball_toss \
  --output-dir logs/torque_zoom
```

Run only for `incline_walk` segments:

```bash
python scripts/evaluation/plot_zoomed_context_torque_prediction.py \
  --model "$MODEL" \
  --segment incline_walk \
  --output-dir logs/torque_zoom
```

Optionally limit the plotted time window:

```bash
python scripts/evaluation/plot_zoomed_context_torque_prediction.py \
  --model "$MODEL" \
  --segment incline_walk \
  --max-samples 500 \
  --output-dir logs/torque_zoom
```

---

### `scripts/evaluation/plot_left_leg_torque_grid.py`

Creates a grid-style torque plot for BT24 left-leg movement segments.

Each column corresponds to one movement segment. The two rows correspond to:

```text
Joint 0: hip
Joint 1: knee
```

The plot compares measured torque against Context-Aware DeLaN prediction and writes a metrics CSV.

Set a model path:

```bash
MODEL="cadelac/learning/trained_models/res_model/panda/ContextAware/epochs_3000exo_hip_knee_delan_2dof_left_all_trials_context.torch"
```

Run for `ball_toss`:

```bash
python scripts/evaluation/plot_left_leg_torque_grid.py \
  --model "$MODEL" \
  --movement ball_toss \
  --output-dir logs/left_leg_torque_grid
```

Run for `incline_walk`:

```bash
python scripts/evaluation/plot_left_leg_torque_grid.py \
  --model "$MODEL" \
  --movement incline_walk \
  --output-dir logs/left_leg_torque_grid
```

Use this script for quick visual comparison across multiple BT24 movement segments.

---

## 4. Current Exoskeleton Configuration

The current exoskeleton setup uses a simplified 2-DOF hip-knee model:

```text
n_dof = 2
```

The two modeled joints are:

```text
Joint 0: hip
Joint 1: knee
```

The Context-Aware LSTM history input is built from:

```text
q history
qdot history
measured torque history tau
```

In this simplified setup:

```text
diff_tau = tau
```

Artificial training noise is disabled because the exoskeleton data already comes from real measured motion data.

---

## 5. Train/Test Split

The current exoskeleton experiments use a subject-wise split.

The test split is selected automatically by searching for labels containing:

```text
BT24
```

BT24 segments are therefore used for testing. The remaining non-BT24 labels are used for training.

This is intended to evaluate whether the Context-Aware model can generalize to a subject that was not part of the training split.

---

## 6. Repository Hygiene

Generated experiment artifacts should remain local and should not be committed.

Ignored artifacts include:

```text
logs/
generated plots
generated metric CSV files
generated .pkl datasets
trained .torch model files
checkpoint folders
zip packages
local helper scripts
```

This keeps the repository focused on source code, scripts, and documentation.

Datasets, trained models, logs, and plots should be regenerated locally or shared separately when needed.
