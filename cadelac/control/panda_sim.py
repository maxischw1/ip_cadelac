import mujoco
import mujoco.viewer
from dm_control import mjcf
import pinocchio as pin

import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import copy
from pathlib import Path

from cadelac.control.mj_getters import MjGetters
from cadelac.control.casadi_model import CasadiModel
from cadelac.control.kf_state_force import KFStateFee

class PandaSim(MjGetters):
    def __init__(self,
                tracking_mode = 'joint',
                use_viewer = False,
                robot_name = 'panda',
                seed = None,
                run_name = 'run',
                random_init = True,
            ):
        
        self.use_viewer = use_viewer
        self.robot_name = robot_name
        self.tracking_mode = tracking_mode
        self.random_init = random_init

        # Setup Mujoco
        self.rand_env = 2
        # box_pos = np.array([0.07, 0.1, 0.2])
        box_pos = None
        box_inertia_flag = True
        box_mass = np.array([5])

        self.sim_xml_handles, self.xml_handles, _ = build_models(num_rand_envs=self.rand_env, box_pos=box_pos, box_inertia_flag=box_inertia_flag, box_mass=box_mass, seed=seed)

        self.env_id = 0
        self.mj_model = mujoco.MjModel.from_xml_string(self.xml_handles[self.env_id].to_xml_string(), assets=self.xml_handles[self.env_id].get_assets())
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mujoco.mj_step(self.mj_model, self.mj_data)

        # self.sim_mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.sim_mj_model = mujoco.MjModel.from_xml_string(self.sim_xml_handles[self.env_id].to_xml_string(), assets=self.sim_xml_handles[self.env_id].get_assets())
        self.sim_mj_data = mujoco.MjData(self.sim_mj_model)
        mujoco.mj_resetDataKeyframe(self.sim_mj_model, self.sim_mj_data, 0)
        mujoco.mj_forward(self.sim_mj_model, self.sim_mj_data)

        self.joint_ids, self.ee_id, self.box_id = self.get_model_ids(self.mj_model, act_blacklist = ['actuator8'])
        self.nq = len(self.joint_ids)
        self.joint_ranges = self.get_joint_ranges(self.mj_model, self.joint_ids)

        self.ee_pos_home = self.get_ee_pos(self.sim_mj_data)
        self.q_home = self.get_joint_pos(self.sim_mj_data)

        self.last_time = time.time_ns()
        self.initial_time = time.time_ns()

        self.set_joint_properties()

        # Create Casadi model to get nominal model using pinocchio
        self.cs_robot_model = CasadiModel()
        self.cs_robot_model.model()

        # Init data
        self.init_exc_ref_ok = True
        self.viewer = None
        self.sim_steps = 0
        self.reset(run_name=run_name, seed=seed)


    def init_kf_fee(self, q_init, qd_init, tau_init):
        self.kf_state_Fee = KFStateFee(self.cs_robot_model.model_pin,
                                     self.cs_robot_model.data_pin,
                                     self.joint_ctrl_period,
                                     q_init,
                                     qd_init,
                                     Fee_init=None,
                                     u_init=tau_init,
                                     )

    
    def compute_joint_linear_lqr(self):

        dt = self.joint_ctrl_period
        q = self.get_joint_pos(self.sim_mj_data)
        qd = self.get_joint_vel(self.sim_mj_data)

        Mq = self.get_Mq(self.sim_mj_model, self.sim_mj_data)
        Mq_inv = self.get_Mq_inv(self.sim_mj_model, self.sim_mj_data)

        ## State space update
        ncord = 7
        nx = 2*ncord
        nu = ncord
        Ac = np.zeros((nx,nx))
        Bc = np.zeros((nx,nu))
        Ac[:ncord,ncord:] = np.eye(ncord)
        Bc[ncord:,:] =  Mq_inv

        Ad = np.eye(nx) + dt*Ac
        Bd = dt*Bc

        # LQR weight matrix
        Qpos = 100*np.ones(ncord)
        Qvel = 100*np.ones(ncord)
        R = 0.05*np.eye(nu)

        Q = np.diag(np.concatenate((Qpos, Qvel)))

        # Solve discrete Riccati equation.
        P = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)

        # Compute the feedback gain matrix K.
        Klqr = np.linalg.inv(R + Bd.T @ P @ Bd) @ Bd.T @ P @ Ad

        x = np.concatenate((q, qd))
        x_ref = np.concatenate((self.q_pos_ref, self.q_vel_ref))
        x_error = x_ref - x
        tau_lqr = Klqr @ x_error

        tau_cmp = self.sim_mj_data.qfrc_bias[self.joint_ids]

        return tau_lqr + tau_cmp


    def compute_ee_linear_lqr(self, n_steps):

        dt = self.joint_ctrl_period
        # Current current state
        ee_pos = self.get_ee_pos(self.sim_mj_data)
        ee_vel = self.get_ee_vel(self.sim_mj_data)

        q = self.get_joint_pos(self.sim_mj_data)
        qd = self.get_joint_vel(self.sim_mj_data)

        ee_vel = self.get_ee_vel(self.sim_mj_data)

        Jee = self.get_body_jac(self.sim_mj_model, self.sim_mj_data, self.ee_id)
        if n_steps > 0:
            Jee_d = (Jee - self.Jee_old) / dt
        self.Jee_old = Jee
        Mq = self.get_Mq(self.sim_mj_model, self.sim_mj_data)
        Mq_inv = self.get_Mq_inv(self.sim_mj_model, self.sim_mj_data)

        ## State space update
        ncord = 3
        nx = 2*ncord
        nu = ncord
        Ac = np.zeros((nx,nx))
        Bc = np.zeros((nx,nu))
        Ac[:ncord,ncord:] = np.eye(ncord)
        Bc[ncord:,:] = Jee @ Mq_inv @ Jee.T

        Ad = np.eye(nx) + dt*Ac
        Bd = dt*Bc

        # LQR weight matrix
        Qpos = 100*np.ones(ncord)
        Qvel = 1000*np.ones(ncord)
        Q = np.diag(np.concatenate((Qpos, Qvel)))
        R = 1*np.eye(nu)

        # Solve discrete Riccati equation.
        P = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)

        # Compute the feedback gain matrix K.
        Klqr = np.linalg.inv(R + Bd.T @ P @ Bd) @ Bd.T @ P @ Ad

        x = np.concatenate((ee_pos, ee_vel))
        x_ref = np.concatenate((self.ee_pos_ref, self.ee_vel_ref))
        x_error = x_ref - x
        Force_lqr = Klqr @ x_error

        tau_lqr = Jee.T @ Force_lqr
        
        tau_g = self.sim_mj_data.qfrc_bias[self.joint_ids]
        tau_cmp = tau_g
        if n_steps > 0:
            tau_cmp += Mq @ np.linalg.pinv(Jee) @ Jee_d @ qd

        return tau_lqr + tau_cmp


    def step(self):

        ## Update reference
        if self.tracking_mode == 'ee':
            if self.lin_sin_ref:
                self.ee_pos_ref, self.ee_vel_ref = self.generate_lin_sin_ref(self.sim_steps * self.joint_ctrl_period)
            else:
                self.ee_pos_ref = np.copy(self.ee_pos_init)
                self.ee_pos_ref[2] -= 0.05
                self.ee_vel_ref = np.zeros(3)
        else:
            self.q_pos_ref, self.q_vel_ref = self.generate_joint_sin_ref(self.sim_steps * self.joint_ctrl_period)


        ## Log new data
        if self.tracking_mode == 'ee':
            self.log_new_data(self.sim_steps * self.joint_ctrl_period, ee_pos_ref = self.ee_pos_ref, ee_vel_ref = self.ee_vel_ref)
        else:
            self.log_new_data(self.sim_steps * self.joint_ctrl_period, qp_ref = self.q_pos_ref, qv_ref = self.q_vel_ref)
            

        if self.tracking_mode == 'ee':
            torque = self.compute_ee_linear_lqr(self.sim_steps)
        else:
            torque = self.compute_joint_linear_lqr()
            self.torque_lqr = np.copy(torque)

        self.update_mj_sim(torque)
        
        self.sim_steps += 1
        if self.viewer is not None:
            self.update_viewer()

        ee_pos = self.get_ee_pos(self.sim_mj_data)
        # print(f'ee_pos {ee_pos} | ee_pos_ref {self.ee_pos_ref} | ee_pos_error {self.ee_pos_ref - ee_pos}')


    def reset(self, run_name = 'run', seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Get a new env if
        if self.rand_env > 1:
            # self.env_id = np.random.randint(low=0, high=self.rand_env-1)
            # if self.env_id == 0:
            self.env_id = int(not self.env_id)
            print(f'self.env_id {self.env_id }')
            self.mj_model = mujoco.MjModel.from_xml_string(self.xml_handles[self.env_id].to_xml_string(), assets=self.xml_handles[self.env_id].get_assets())
            self.mj_data = mujoco.MjData(self.mj_model)
            self.sim_mj_model = mujoco.MjModel.from_xml_string(self.sim_xml_handles[self.env_id].to_xml_string(), assets=self.sim_xml_handles[self.env_id].get_assets())
            self.sim_mj_data = mujoco.MjData(self.sim_mj_model)

        # Set initial pos
        if self.random_init:
            self.q_init = np.copy(self.q_home) + np.random.uniform(
                low=-0.5, high=0.5, size=self.nq
            )
            self.q_init = np.clip(np.copy(self.q_init), self.joint_ranges[:,0], self.joint_ranges[:,1])
        else:
            self.q_init = np.copy(self.q_home)
        self.mj_data.qpos[:] = np.copy(self.q_init)
        self.sim_mj_data.qpos[:] = np.copy(self.q_init)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mujoco.mj_forward(self.sim_mj_model, self.sim_mj_data)

        # Generate Reference
        self.init_reference_data()
        self.init_logger(run_name)

        if self.use_viewer:
            if self.viewer:
                self.viewer.close()
            self.viewer = mujoco.viewer.launch_passive(self.sim_mj_model, self.sim_mj_data)
        else:
            self.viewer = None

        # Init data
        self.sim_steps = 0

    def update_mj_sim(self, torque):
        for j in range(int(self.joint_ctrl_period / self.sim_mj_model.opt.timestep)):
            self.sim_mj_data.ctrl = torque
            mujoco.mj_step(self.sim_mj_model, self.sim_mj_data)


    def update_viewer(self):
        self.viewer.sync()
        update_time = time.time_ns()
        elapsed_time = (update_time - self.last_time)*1e-9
        self.last_time = update_time
        sleep_time = np.max([self.joint_ctrl_period - elapsed_time,0])
        if sleep_time > 0.0:
            time.sleep(sleep_time)


    def init_reference_data(self):

        if self.tracking_mode == 'ee':
            self.ee_pos_ref = np.copy(self.ee_pos_init)
            self.ee_vel_ref = np.zeros(3)

            self.lin_sin_ref = True
            self.lin_sin_ref_amp = np.array([0.1, 0.2, 0.15])
            self.lin_sin_ref_freq = np.array([0.15, 0.2, 0.1])
            self.lin_sin_ref_phase = np.array([0.1, 0.1, 0.0])
            self.lin_sin_ref_offset = np.copy(self.ee_pos_init)

        elif self.tracking_mode == 'joint':
            # self.q_sin_ref_amp = np.array([2.5, 1.3, 2.5, 1.5, 2.0, 1.5, 2.0])
            # self.q_sin_ref_freq = 0.3*np.ones(self.nq)

            # self.q_sin_ref_amp = np.array([np.pi, 1.5, 1.2, 2.5, 1.8, 1.0, 2.0])
            # self.q_sin_ref_freq = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.1, 0.1])

            # # self.q_sin_ref_amp = np.array([np.pi, 1.5, 1.2, 0.5, 0.8, 1.0, 2.0])
            # # # self.q_sin_ref_amp = np.zeros(self.nq)
            # # # self.q_sin_ref_amp[3] = 1.8
            # # # self.q_sin_ref_amp[4] = 1.0
            # # # self.q_sin_ref_amp[5] = 1.0
            # # # self.q_sin_ref_amp[6] = np
            # # self.q_sin_ref_freq = np.array([0.05, 0.2, 0.05, 0.1, 0.2, 0.1, 0.05])
            # self.q_sin_ref_phase = np.array([0.05, 1.0, 0.7, 0.15, 0.2, 0.1, 0.3])
            # self.q_sin_ref_offset = np.copy(self.q_init)

            # self.q_sin_ref_amp = np.random.uniform(low=np.zeros(self.nq), high=np.array([2.5, 0.5, 1.2, 0.5, 0.8, 1.0, 1.7]), size=self.nq)
            # self.q_sin_ref_freq = np.random.uniform(low=np.zeros(self.nq), high=0.1*np.ones(self.nq), size=self.nq)
            # self.q_sin_ref_phase = np.random.uniform(low=np.zeros(self.nq), high=0.5*np.ones(self.nq), size=self.nq)

            # # # self.q_sin_ref_amp = np.random.uniform(low=np.zeros(self.nq), high=np.array([2.5, 1.3, 2.5, 1.5, 2.0, 1.5, 2.0]), size=self.nq)
            # # # self.q_sin_ref_freq = np.random.uniform(low=np.zeros(self.nq), high=0.3*np.ones(self.nq), size=self.nq)
            # # self.q_sin_ref_amp = np.random.uniform(low=np.zeros(self.nq), high=np.array([2.5, 1.3, 2.5, 1.5, 2.0, 1.5, 2.0]), size=self.nq)
            # # self.q_sin_ref_freq = np.random.uniform(low=np.zeros(self.nq), high=0.2*np.ones(self.nq), size=self.nq)
            # self.q_sin_ref_amp = np.random.uniform(low=np.zeros(self.nq), high=np.array([1.5, 1.0, 1.5, 1.0, 1.5, 1.0, 1.5]), size=self.nq)
            # self.q_sin_ref_freq = np.random.uniform(low=np.zeros(self.nq), high=0.15*np.ones(self.nq), size=self.nq)
            # self.q_sin_ref_phase = np.random.uniform(low=np.zeros(self.nq), high=0.5*np.ones(self.nq), size=self.nq)
            self.q_sin_ref_amp = np.random.uniform(low=np.zeros(self.nq), high=np.array([1.5, 1.0, 1.5, 1.0, 1.5, 1.0, 1.5]), size=self.nq)
            # self.q_sin_ref_freq = np.random.uniform(low=np.zeros(self.nq), high=0.3*np.ones(self.nq), size=self.nq)
            self.q_sin_ref_freq = np.random.uniform(low=np.zeros(self.nq), high=0.15*np.ones(self.nq), size=self.nq)
            self.q_sin_ref_phase = np.random.uniform(low=np.zeros(self.nq), high=np.zeros(self.nq), size=self.nq)
            self.q_sin_ref_offset = np.copy(self.q_init)
            

    def generate_lin_sin_ref(self, current_time):
        sin_ref = np.zeros((3))
        sin_vel_ref = np.zeros((3))
        sin_time = current_time + self.joint_ctrl_period
        for i in range(3):
            sin_ref[i] = self.lin_sin_ref_amp[i] * np.sin(2 * np.pi * self.lin_sin_ref_freq[i] * sin_time + self.lin_sin_ref_phase[i]) + self.lin_sin_ref_offset[i]
            sin_vel_ref[i] = 2 * np.pi * self.lin_sin_ref_freq[i] * self.lin_sin_ref_amp[i] * np.cos(2 * np.pi * self.lin_sin_ref_freq[i] * sin_time + self.lin_sin_ref_phase[i])

        return sin_ref, sin_vel_ref
    
    def generate_joint_sin_ref(self, current_time):
        sin_ref = np.zeros((self.nq))
        sin_vel_ref = np.zeros((self.nq))
        # sin_time = current_time + self.joint_ctrl_period
        sin_time = current_time
        for i in range(self.nq):
            sin_ref[i] = self.q_sin_ref_amp[i] * np.sin(2 * np.pi * self.q_sin_ref_freq[i] * sin_time + self.q_sin_ref_phase[i]) + self.q_sin_ref_offset[i]
            sin_vel_ref[i] = 2 * np.pi * self.q_sin_ref_freq[i] * self.q_sin_ref_amp[i] * np.cos(2 * np.pi * self.q_sin_ref_freq[i] * sin_time + self.q_sin_ref_phase[i])

            # Disable margin if element is negative
            if self.q_ref_margin[i] >= 0.0:
                org_ref = np.copy(sin_ref[i])
                sin_ref[i] = np.clip(sin_ref[i], self.joint_ranges[i,0]+self.q_ref_margin[i], self.joint_ranges[i,1]-self.q_ref_margin[i])
                if sin_ref[i] != org_ref:
                    # Set zero velocity
                    sin_vel_ref[i] = 0

            if self.q_ref_vel_margin[i] >= 0.0:
                sin_vel_ref[i] = np.clip(sin_vel_ref[i], -self.vel_lim[i]+self.q_ref_vel_margin[i], self.vel_lim[i]-self.q_ref_vel_margin[i])

        return sin_ref, sin_vel_ref

    def init_exc_ref(self, run_fourier_config, run_robot_config, init_pos, init_vel):
        run_robot_config.update({"init_pos": init_pos,
                                "init_vel": init_vel})

        params = self.ref_gen.exc_ref_get_params(run_fourier_config,
                                                 run_robot_config)

        # If it fails to get a valid trajectory
        if params is None:
            self.init_exc_ref_ok = False
            
        else:
            self.init_exc_ref_ok = True

            Ts = self.joint_ctrl_period
            exc_traj_data = self.ref_gen.exc_ref_generate_full_traj(run_fourier_config,
                                                                    run_robot_config,
                                                                    params,
                                                                    Ts)
            self.exc_time_traj, self.exc_q_traj, self.exc_qd_traj, self.exc_qdd_traj = exc_traj_data
            self.exc_q_traj = self.exc_q_traj.T
            self.exc_qd_traj = self.exc_qd_traj.T
            self.exc_qdd_traj = self.exc_qdd_traj.T

    def get_exc_ref(self, index_time):
        q_ref = self.exc_q_traj[:,index_time]
        qd_ref = self.exc_qd_traj[:,index_time]
        return q_ref, qd_ref

    def set_joint_properties(self):
        self.joint_ctrl_period = 0.02
        self.torque_lim = np.array([87, 87, 87, 87, 12, 12, 12])
        self.vel_lim = np.array([2, 2, 2, 2, 2.3, 2.3, 2.3])
        self.q_ref_margin = 0.1 * np.ones(self.nq)
        self.q_ref_vel_margin = 0.3 * np.ones(self.nq)

    def log_new_data(self, time,
                    ee_pos_ref = np.zeros(3), ee_vel_ref = np.zeros(3),
                    qp_ref = np.zeros(3), qv_ref = np.zeros(3)):
        qp = self.get_joint_pos(self.sim_mj_data)
        qv = self.get_joint_vel(self.sim_mj_data)
        qa = self.get_joint_acc(self.sim_mj_data)
        Mq = self.get_Mq(self.sim_mj_model, self.sim_mj_data)

        ee_pos = self.get_ee_pos(self.sim_mj_data)
        ee_vel = self.get_ee_vel(self.sim_mj_data)

        tau_m = Mq @ qa
        tau_cg = np.copy(self.sim_mj_data.qfrc_bias.reshape((self.nq,1))).reshape(self.nq,)
        tau = tau_m + tau_cg

        # Get old data from nominal model before update
        # qp_old = self.get_joint_pos(self.mj_data)
        # qv_old = self.get_joint_vel(self.mj_data)
        # Get Variabels from Nominal model
        self.update_nominal_model()
        Mq_nom = self.get_Mq(self.mj_model, self.mj_data)
        tau_m_nom = Mq_nom @ qa
        tau_cg_nom = np.copy(self.mj_data.qfrc_bias.reshape((self.nq,)))
        tau_nom = tau_m_nom + tau_cg_nom

        # self.logged_data['labels'].append('run_' + str(self.seed))
        self.logged_data['t'].append(time)
        self.logged_data['qp'].append(qp)
        self.logged_data['qv'].append(qv)
        self.logged_data['qa'].append(qa)

        self.logged_data['tau'].append(tau)
        self.logged_data['m'].append(tau_m)
        self.logged_data['c'].append(tau_cg)
        self.logged_data['g'].append(np.zeros(self.nq))
        self.logged_data['Mq'].append(Mq.reshape(-1))

        self.logged_data['ee_pos'].append(ee_pos)
        self.logged_data['ee_vel'].append(ee_vel)
        self.logged_data['ee_pos_ref'].append(ee_pos_ref)
        self.logged_data['ee_vel_ref'].append(ee_vel_ref)

        self.logged_data['qp_ref'].append(qp_ref)
        self.logged_data['qv_ref'].append(qv_ref)

        self.logged_data['p'].append(np.zeros(self.nq))
        self.logged_data['pdot'].append(np.zeros(self.nq))

        self.logged_data['tau_nom'].append(tau_nom)
        self.logged_data['m_nom'].append(tau_m_nom)
        self.logged_data['c_nom'].append(tau_cg_nom)
        self.logged_data['g_nom'].append(np.zeros(self.nq))
        self.logged_data['Mq_nom'].append(Mq_nom.reshape(-1))

        self.logged_data['diff_tau_nom'].append(tau-tau_nom)
        self.logged_data['diff_tau_m_nom'].append(tau_m-tau_m_nom)
        self.logged_data['diff_tau_c_nom'].append(tau_cg-tau_cg_nom)
        self.logged_data['diff_tau_g_nom'].append(np.zeros(self.nq))

        self.logged_data['env_id'].append(self.env_id)
        self.logged_data['enc_input'].append(self.enc_input)

        Mq_nom_pin = np.array(self.cs_robot_model.cpin_H_fn(qp))
        tau_m_nom_pin = (Mq_nom_pin @ qa).reshape(self.nq)
        tau_cg_nom_pin = np.array(self.cs_robot_model.cpin_tau_cg_fn(qp, qv)).reshape(self.nq)
        tau_nom_pin = tau_m_nom_pin + tau_cg_nom_pin
        self.logged_data['tau_nom_pin'].append(tau_nom_pin)
        self.logged_data['m_nom_pin'].append(tau_m_nom_pin)
        self.logged_data['c_nom_pin'].append(tau_cg_nom_pin)
        self.logged_data['g_nom_pin'].append(np.zeros(self.nq))
        self.logged_data['Mq_nom_pin'].append(Mq_nom_pin.reshape(-1))

        self.logged_data['diff_tau_nom_pin'].append(tau - tau_nom_pin)
        self.logged_data['diff_tau_m_nom_pin'].append(tau_m - tau_m_nom_pin)
        self.logged_data['diff_tau_c_nom_pin'].append(tau_cg - tau_cg_nom_pin)
        self.logged_data['diff_tau_g_nom_pin'].append(np.zeros(self.nq))

    def log_new_value(self, key, value):
        if key in self.logged_data.keys():
            self.logged_data[key].append(value)

    def log_init_value(self, key):
        self.logged_data[key] = []

    def update_nominal_model(self):
        # Update Mujoco Model for Controller
        self.mj_data.qpos[:] = np.copy(self.sim_mj_data.qpos)
        self.mj_data.qvel[:] = np.copy(self.sim_mj_data.qvel)
        mujoco.mj_forward(self.mj_model, self.mj_data)


    def init_logger(self, run_name = 'run'):
        self.logged_data = {}
        self.logged_data['labels'] = run_name
        self.logged_data['t'] = []
        self.logged_data['qp'] = []
        self.logged_data['qv'] = []
        self.logged_data['qa'] = []

        self.logged_data['tau'] = []
        self.logged_data['m'] = []
        self.logged_data['c'] = []
        self.logged_data['g'] = []
        self.logged_data['Mq'] = []

        self.logged_data['ee_pos'] = []
        self.logged_data['ee_vel'] = []
        self.logged_data['ee_pos_ref'] = []
        self.logged_data['ee_vel_ref'] = []

        self.logged_data['qp_ref'] = []
        self.logged_data['qv_ref'] = []

        self.logged_data['p'] = []
        self.logged_data['pdot'] = []

        self.logged_data['tau_nom'] = []
        self.logged_data['m_nom'] = []
        self.logged_data['c_nom'] = []
        self.logged_data['g_nom'] = []
        self.logged_data['Mq_nom'] = []

        self.logged_data['diff_tau_nom'] = []
        self.logged_data['diff_tau_m_nom'] = []
        self.logged_data['diff_tau_c_nom'] = []
        self.logged_data['diff_tau_g_nom'] = []

        self.logged_data['env_id'] = []
        self.logged_data['enc_input'] = []
        self.enc_input = 0

        self.logged_data['tau_nom_pin'] = []
        self.logged_data['m_nom_pin'] = []
        self.logged_data['c_nom_pin'] = []
        self.logged_data['g_nom_pin'] = []
        self.logged_data['Mq_nom_pin'] = []

        self.logged_data['diff_tau_nom_pin'] = []
        self.logged_data['diff_tau_m_nom_pin'] = []
        self.logged_data['diff_tau_c_nom_pin'] = []
        self.logged_data['diff_tau_g_nom_pin'] = []

    def convert_log_data_to_np(self):
        for key in self.logged_data.keys():
            self.logged_data[key] = np.array(self.logged_data[key])

    def plot_ee_pos(self, logged_data = None):
        if logged_data is None:
            logged_data = self.logged_data
        time_traj = logged_data['t']
        ee_pos = logged_data['ee_pos']
        ee_pos_ref = logged_data['ee_pos_ref']

        # Linear Pos
        labels = ['x', 'y', 'z']
        fig, axes = plt.subplots(3, 1)
        for cor in range(3):
            axes[cor].plot(time_traj, ee_pos[:,cor], label=f'Real - {labels[cor]}')
            axes[cor].plot(time_traj, ee_pos_ref[:,cor], '--r', label=f'Ref - {labels[cor]}')
            axes[cor].grid()
            axes[cor].legend()
        axes[0].set_title(f'End Effector Position')
        
    def plot_ee_pos_3d(self, logged_data = None):
        if logged_data is None:
            logged_data = self.logged_data
        time_traj = logged_data['t']
        ee_pos = logged_data['ee_pos']
        ee_pos_ref = logged_data['ee_pos_ref']

        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(ee_pos[:,0], ee_pos[:,1], ee_pos[:,2], label=f'Real')
        ax.scatter(ee_pos_ref[:,0], ee_pos_ref[:,1], ee_pos_ref[:,2], '--r', label=f'Ref')

        ax.set_title(f'End Effector Position')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.set_xlim([-1.0, 1.0])
        ax.set_ylim([-1.0, 1.0])
        ax.set_zlim([0.0, 1.5])

        ax.legend()

    def plot_ee_vel(self, logged_data = None):
        if logged_data is None:
            logged_data = self.logged_data
        time_traj = logged_data['t']
        ee_vel = logged_data['ee_vel']
        ee_vel_ref = logged_data['ee_vel_ref']

        # Linear Pos
        labels = ['vx', 'vy', 'vz']
        fig, axes = plt.subplots(3, 1)
        for cor in range(3):
            axes[cor].plot(time_traj, ee_vel[:,cor], label=f'Real - {labels[cor]}')
            axes[cor].plot(time_traj, ee_vel_ref[:,cor], '--r', label=f'Ref - {labels[cor]}')
            axes[cor].grid()
            axes[cor].legend()
        axes[0].set_title(f'End Effector Velocity')

    def plot_joint_pos(self, logged_data = None):
        if logged_data is None:
            logged_data = self.logged_data
        time_traj = logged_data['t']
        qp = logged_data['qp']
        qp_ref = logged_data['qp_ref']

        labels = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7']
        fig, axes = plt.subplots(self.nq, 1)
        for cor in range(self.nq):
            axes[cor].plot(time_traj, qp[:,cor], label=f'Real - {labels[cor]}')
            axes[cor].plot(time_traj, qp_ref[:,cor], '--r', label=f'Ref - {labels[cor]}')
            axes[cor].grid()
            if cor == 0:
                axes[cor].legend(loc="upper right")
        axes[0].set_title(f'Joint Position')

    def plot_joint_vel(self, logged_data = None):
        if logged_data is None:
            logged_data = self.logged_data
        time_traj = logged_data['t']
        qv = logged_data['qv']
        qv_ref = logged_data['qv_ref']

        labels = ['q1_d', 'q2_d', 'q3_d', 'q4_d', 'q5_d', 'q6_d', 'q7_d']
        fig, axes = plt.subplots(self.nq, 1)
        for cor in range(self.nq):
            axes[cor].plot(time_traj, qv[:,cor], label=f'Real - {labels[cor]}')
            axes[cor].plot(time_traj, qv_ref[:,cor], '--r', label=f'Ref - {labels[cor]}')
            axes[cor].grid()
            if cor == 0:
                axes[cor].legend(loc="upper right")
        axes[0].set_title(f'Joint Velocity')

    def plot_joint_acc(self, logged_data = None):
        if logged_data is None:
            logged_data = self.logged_data
        time_traj = logged_data['t']
        qa = logged_data['qa']

        labels = ['q1_dd', 'q2_dd', 'q3_dd', 'q4_dd', 'q5_dd', 'q6_dd', 'q7_dd']
        fig, axes = plt.subplots(self.nq, 1)
        for cor in range(self.nq):
            axes[cor].plot(time_traj, qa[:,cor], label=f'Real - {labels[cor]}')
            axes[cor].grid()
            if cor == 0:
                axes[cor].legend(loc="upper right")
        axes[0].set_title(f'Joint Aceleration')

    def plot_torque(self, logged_data = None):
        if logged_data is None:
            logged_data = self.logged_data
        time_traj = logged_data['t']
        tau = logged_data['tau']
        tau_cg = logged_data['c_nom_pin']

        fig, axes = plt.subplots(7, 1)
        for cor in range(7):
            axes[cor].plot(time_traj, tau[:,cor], label=f'Real')
            axes[cor].plot(time_traj, tau_cg[:,cor], label=f'Nom coriolis gravity')
            axes[cor].plot(time_traj, np.ones(time_traj.shape[0])*self.torque_lim[cor], '--r', label=f'Limit')
            axes[cor].plot(time_traj, -np.ones(time_traj.shape[0])*self.torque_lim[cor], '--r')
            axes[cor].grid()
            if cor == 0:
                axes[cor].legend(loc="upper right")
        axes[0].set_title(f'Joint Torques')

    def plot_enc_input(self, logged_data = None):
        if logged_data is None:
            logged_data = self.logged_data
        time_traj = logged_data['t']
        enc_input = logged_data['enc_input']

        n_enc_input = enc_input.shape[1]
        fig, axes = plt.subplots(n_enc_input, 1)
        for cor in range(n_enc_input):
            axes[cor].plot(time_traj, enc_input[:,cor], label=f'Real')
            axes[cor].grid()
            if cor == 0:
                axes[cor].legend(loc="upper right")
        axes[0].set_title(f'Enc input')

    def plot_env_id(self, logged_data = None):
        if logged_data is None:
            logged_data = self.logged_data
        time_traj = logged_data['t']
        env_id = logged_data['env_id']

        fig, ax = plt.subplots()
        ax.plot(time_traj, env_id)
        ax.set_title(f'Env Id')

    def plot_diff_tau_nom(self, logged_data = None):
        if logged_data is None:
            logged_data = self.logged_data
        time_traj = logged_data['t']
        diff_tau_nom = logged_data['diff_tau_nom']

        fig, axes = plt.subplots(self.nq, 1)
        for cor in range(self.nq):
            axes[cor].plot(time_traj, diff_tau_nom[:,cor], label=f'Real')
            axes[cor].grid()
            if cor == 0:
                axes[cor].legend(loc="upper right")
        axes[0].set_title(f'Diff torque nominal')


def build_models(num_rand_envs, box_pos=None, box_inertia_flag=False, box_mass=1.0, collision=True, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # Setup Mujoco
    robot_name = 'panda'
    if collision:
        xml_name = "scene.xml"
    else:
        xml_name = "scene_no_collision.xml"

    xml_path = (
            Path(__file__).resolve().parent.parent
            / "robots"
            / str(robot_name)
            / xml_name
        ).as_posix()

    original_xml_handle = mjcf.from_path(xml_path)
    xml_handles = []
    xml_no_inertia_handles = []
    for i in range(num_rand_envs):
        box_xml, box_no_inertia_xml = add_box_ee(copy.copy(original_xml_handle), box_pos, box_inertia_flag, box_mass, i)
        xml_handles.append(box_xml)
        xml_no_inertia_handles.append(box_no_inertia_xml)

    return xml_handles, xml_no_inertia_handles, original_xml_handle

pos = np.random.uniform(low=-0.5, high=0.5, size=3)
mass_box = np.random.uniform(low=0.0, high=5, size=1)

def add_box_ee(xml_handle, box_pos, flag_inertia, mass_value, index):

    ee_link = xml_handle.find("body", "end_effector")
    scaling = 1.0
    size = np.array([0.05, 0.05, 0.05]) * scaling
    pos_zero = np.array([0.0, 0.0, 0.0])

    if box_pos is None:
        # pos = np.random.uniform(low=-0.5, high=0.5, size=3)
        pos = np.random.uniform(low=-0.3, high=0.3, size=3)
    else:
        if box_pos.ndim > 1:
            pos = box_pos[index]
        else:
            pos = box_pos

    ee_box_link = ee_link.add("body", name="ee_box", pos=pos.tolist(), euler=[0.0, 0.0, 0.0])
    ee_box_link.add("geom", name="ee_box", type="box", dclass="visual", size=size.tolist(), pos=pos_zero,
                rgba=[0.5, 0.5, 0.5, 0.5], euler=[0.0, 0.0, 0.0])

    xml_handle_no_inertia = copy.copy(xml_handle)
    if flag_inertia:
        if mass_value is None:
            # mass_box = np.random.uniform(low=0.0, high=5,    size=1)
            mass_box = np.random.uniform(low=0.0, high=4, size=1)
        else:
            if mass_value.size > 1:
                mass_box = mass_value[index]
            else:
                mass_box = mass_value
        ee_box_link.add(
                'inertial',
                mass=mass_box*scaling,
                pos=pos_zero,
                diaginertia=0.001*scaling*np.ones(3),
            )

    return xml_handle, xml_handle_no_inertia


if __name__ == "__main__":
    panda_sim = PandaSim(use_viewer=True, seed=0)

    # Simulation Parameters
    Tsim = 20
    Nsim = int(Tsim / panda_sim.joint_ctrl_period)

    init_time = time.time()
    # Simulate
    for i in range(Nsim):
        panda_sim.step()
        # if i == Nsim/2:
        #     panda_sim.reset()

    print(f'Sim total time {time.time() - init_time}')
    
    panda_sim.convert_log_data_to_np()
    
    rms_error = np.sqrt(np.mean(np.square(panda_sim.logged_data['qp_ref']-panda_sim.logged_data['qp']),axis=0))
    print(f'rms_error {rms_error} | q_init {panda_sim.q_init} | q_home {panda_sim.q_home}')


    if panda_sim.tracking_mode == 'ee':
        panda_sim.plot_ee_pos()
        panda_sim.plot_ee_pos_3d()
        panda_sim.plot_ee_vel()
    else:
        panda_sim.plot_joint_pos()
        panda_sim.plot_joint_vel()
        panda_sim.plot_joint_acc()
        panda_sim.plot_ee_pos_3d()

    # # if panda_eval.tracking_mode == 'ee':
    # #     panda_eval.plot_ee_pos()
    # #     panda_eval.plot_ee_pos_3d()
    # #     panda_eval.plot_ee_vel()
    # # else:
    # #     panda_eval.plot_joint_pos()
    # #     panda_eval.plot_joint_vel()
    # #     panda_eval.plot_ee_pos_3d()

    # # panda_sim.plot_torque()

    
    plt.show()
