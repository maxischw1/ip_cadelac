#!/usr/bin/env python3

import time
import copy
import torch

import numpy as np
import scipy
import pickle

# ROS
import rospy
import tf
import actionlib

# Franka ROS
from franka_example_controllers.msg import FFWDTorquePDUpdate
from control_msgs.msg import FollowJointTrajectoryAction, GripperCommandAction, GripperCommandGoal, GripperCommandActionGoal
from cadelac_msgs.msg import MPCDebug

from .util import go_to, JointStateListener, ControllerManager, JointStateFilteredListener, first_order_low_pass_filter

from cadelac.learning.models.context_aware_delan import ContextAwareDeLaN

from cadelac.control.acados_mpc import AcadosMPC
from cadelac.control.casadi_model import RealtimeApprox
from cadelac.control.kf_state_force import KFStateFee
from cadelac.control.logger import Logger
from cadelac.control.pin_utils import *
from cadelac.control.reference_generator import ReferenceGenerator
from cadelac.control.l4c_context_aware_delan import L4CContextAwareDeLaN


class CaDeLaCNode:
    def __init__(self):
        self.pos_safe_init = np.array([0, -0.4, 0, -2.4, 0, 2.2, -0.7853])
        self.control_period = 0.02
        rospy.loginfo("Initialize CaDeLaC")
        self.control_mode = 'MPC'
        # self.control_mode = 'LQR'
        
        # Init MPC
        self.nq = 7
        self.init_time_data()
        self.init_mpc()
        self.init_move_avg()
        self.end_traj = False
        # self.ref_type = 'sin'
        self.ref_type = 'FULL_INF'
        # self.ref_type = 'PICK_AND_PLACE'

        self.logger = Logger(time_log=True)

        # # Initialize ROS node

        # Publisher to send joint trajectory commands
        self.controller_manager = ControllerManager(["effort_joint_trajectory_controller", "feedforward_joint_pd_controller"])
        self.ffwd_cmd_pub = rospy.Publisher('/franka_ffwd_pd_controller/command', FFWDTorquePDUpdate, queue_size=1)
        self.gripper_client = actionlib.SimpleActionClient("/franka_gripper/gripper_action", GripperCommandAction)
        self.mpc_debug_pub = rospy.Publisher('/panda_mpc/debug', MPCDebug, queue_size=1)
        self.gripper_cmd_pub = rospy.Publisher('/franka_gripper/gripper_action/goal', GripperCommandActionGoal, queue_size=1)
        

        self.transform_listener = tf.TransformListener()
        default_joint_state_listener = JointStateListener()
        self.joint_state_listener = JointStateFilteredListener(self.mpc.robot_model.tau_nom_inv_dyn_fn)

        ## Initial Setup ##

        rospy.sleep(1)
        # init_joint_state = copy.copy(self.joint_state_listener.get_data())
        # self.init_joint_pos = np.array(init_joint_state.position[:7])
        # self.joint_state_sub = rospy.Subscriber("/joint_states", JointState, self.joint_state_callback)
        # self.limits = Limits()

        self.controller_manager.activate("effort_joint_trajectory_controller", ["feedforward_joint_pd_controller"])
        print('Controller effort_joint_trajectory_controller activated.')
        self.trajectory_client = actionlib.SimpleActionClient(f"effort_joint_trajectory_controller/follow_joint_trajectory",
                                                     FollowJointTrajectoryAction)
        self.trajectory_client.wait_for_server()

        # Initialize the robot in a safe position
        joint_state = default_joint_state_listener.get_data()
        go_to(self.trajectory_client, joint_state.name[:7], joint_state.position[:7],
              self.pos_safe_init, max_vel=0.25)
        
        init_joint_state = copy.copy(default_joint_state_listener.get_data())
        self.init_joint_pos = np.array(init_joint_state.position[:7])

        current_joint_pos = copy.copy(init_joint_state.position[:7])
        print(np.abs(self.pos_safe_init - current_joint_pos))
        print('Arrived at initial position.')

        self.flag_init_mpc = False
        # self.ref_init_length = self.hist_length + 5
        self.ref_init_length = 0

        # get_load_pos = np.array

        if self.ref_type != 'FULL_INF':
            self.open_gripper()
        # self.open_gripper_no_wait()
        # rospy.sleep(1)

        # self.close_gripper_no_wait()
        # rospy.sleep(1)

        # Init Kalman Filter
        init_kf_q = np.array(copy.copy(default_joint_state_listener.get_data()).position[:7])
        init_kf_qd = np.zeros_like(init_kf_q)
        init_kf_tau = np.array(copy.copy(default_joint_state_listener.get_data()).effort[:7])
        self.init_kf_fee(np.copy(init_kf_q), np.copy(init_kf_qd), np.copy(init_kf_tau), self.mpc_period)

        # To guarantee the references were computed before activate controller
        self.reset_exp(init_joint_state)

        # Send gravity compensation command to guarantee some reference at the feedforward_joint_pd_controller
        self.update_gravity_comp(copy.copy(default_joint_state_listener.get_data()))
        # Activate feedforward_joint_pd_controller
        self.controller_manager.activate("feedforward_joint_pd_controller", ["effort_joint_trajectory_controller"])
        print('Controller feedforward_joint_pd_controller activated.')
        self.update_gravity_comp(copy.copy(default_joint_state_listener.get_data()))

        rospy.sleep(0.1)
        if self.joint_state_listener.get_data() is None:
            raise ValueError("Stopping the node! joint_state_listener.get_data() is None")

        # self.close_gripper()
        # print('Gripper closed.')

        # Wait for the publisher to establish connection
        # rospy.sleep(1)
        # self.reset_exp(init_joint_state)

        self.timer = rospy.Timer(rospy.Duration(self.control_period), self.control_callback)
        self.timer_debug = rospy.Timer(rospy.Duration(self.control_period), self.debug_callback)
        print('Node initialized.')


    def close_gripper(self):
        self.gripper_client.wait_for_server()
        gripper_goal = GripperCommandGoal()
        gripper_goal.command.position = 0.0005
        gripper_goal.command.max_effort = 0.0
        self.gripper_client.send_goal(gripper_goal)
        self.gripper_client.wait_for_result()

    def open_gripper(self):
        self.gripper_client.wait_for_server()
        gripper_goal = GripperCommandGoal()
        gripper_goal.command.position = 0.025
        gripper_goal.command.max_effort = 0.0
        self.gripper_client.send_goal(gripper_goal)
        self.gripper_client.wait_for_result()

    def close_gripper_no_wait(self):
        gripper_goal = GripperCommandActionGoal()
        gripper_goal.goal.command.position = 0.0005
        gripper_goal.goal.command.max_effort = 0.0
        self.gripper_cmd_pub.publish(gripper_goal)

    def open_gripper_no_wait(self):
        gripper_goal = GripperCommandActionGoal()
        gripper_goal.goal.command.position = 0.035
        gripper_goal.goal.command.max_effort = 0.0
        self.gripper_cmd_pub.publish(gripper_goal)

    def init_time_data(self):
        self.solver_status = 0
        self.total_time_solver = 0
        self.total_time_param = 0
        self.total_time_ref = 0
        self.total_time_log = 0
        self.total_time_pub = 0
        self.total_time_update = 0
        self.total_time_enc = 0

        self.solver_time = 0
        self.pub_cmd_time = 0
        self.update_time = 0
        self.enc_time = 0

    ## MPC
    def init_mpc(self):

        # MPC Parameters
        self.mpc_period = self.control_period
        self.mpc_freq = 1.0 / self.mpc_period
        self.N_horizon = 12
        self.T_horizon = self.N_horizon * self.mpc_period
        self.expl_dyn = True
        realtime_approx = RealtimeApprox.NO_APPROX


        # controller = 'kf'
        # controller = 'delan'
        controller = 'nominal'

        self.hist_length = 15
        if controller == 'kf':
            self.delan_model = 'KF'
            # name_suffix = '_kf_new_QR'
            # name_suffix = '_kf_QR_v2'
            name_suffix = '_kf_QR_v2_repeat'

        elif controller == 'delan':

            n_lstm_output = 10
            n_enc_input = n_lstm_output
            fwd_embedding = False

            # model_folder = 'learned_dynamics/models/res_model/panda_good_models/'
            # # filename = 'lr_sch_Exp_hist_15_lstm_in_21_h_10_out_10_d_5_act_ld_Softplus_epochs_3000_nw_inertia_30_20_nw_pot_30_20_norm_tau_1_ref_exec_init3_panda_rand_envs_25_kf_1_samples_515000_stime_10.torch'
            # # name_suffix = '_delan_v3_envs_25_kf_1_new_QR'
            # filename = 'lr_sch_Exp_hist_15_lstm_in_21_h_10_out_10_d_5_act_ld_Softplus_epochs_3000_nw_inertia_30_20_nw_pot_30_20_pin__mb_1024_norm_tau_1_ref_exec_init3_panda_rand_envs_100_kf_0_samples_515000_stime_10.torch'
            # name_suffix = '_delan_v3_envs_100_kf_0_new_QR'

            model_folder = 'learned_dynamics/models/res_model/panda_good_models/good_models_2025_02_24/'
            filename = 'lr_sch_No_hist_15_lstm_in_21_h_10_out_10_d_5_act_ld_Softplus_epochs_3000_nw_inertia_30_20_nw_pot_30_20_pin_noise__mb_1024_norm_tau_1_ref_exec_init3_rand_envs_nom_101_kf_0_samples_1040300_stime_10.torch'
            name_suffix = '_delan_v4_envs_101_noise_kf_0_new_QR_repeat'


            # filename = 'lr_sch_Exp_hist_15_lstm_in_21_h_10_out_10_d_5_act_ld_Softplus_epochs_3000_nw_inertia_30_20_nw_pot_30_20_pin_noise__mb_1024_norm_tau_1_ref_exec_init3_rand_envs_nom_101_kf_0_samples_1040300_stime_10.torch'
            # name_suffix = '_delan_v5_envs_101_noise_kf_0_QR_v2'

            load_file = model_folder + filename
            print(f'Loading file {load_file}')

            torch_model = torch.load(load_file, map_location=torch.device('cpu'), weights_only=False)
            if 'n_width' not in torch_model['hyper'].keys():
                torch_model['hyper']['n_width'] = torch_model['hyper']['n_width_inertia']
                torch_model['hyper']['n_depth'] = torch_model['hyper']['n_depth_inertia']
            l4c_delan = L4CContextAwareDeLaN(torch_model, n_dof=self.nq, n_enc_input=n_enc_input,
                                    fwd_embedding=fwd_embedding, device='cpu')
            self.delan_model = l4c_delan

            self.filter_enc_input_freq = 2
            filter_Ts = 1.0 / 50.0
            self.filter_enc_input_alpha = filter_Ts / (filter_Ts + 1.0 / (2 * np.pi * self.filter_enc_input_freq))
            print(f'Enc input filtered {self.filter_enc_input_alpha}')
            self.filter_enc_input_filtered_old = np.zeros(10)

        else:
            self.delan_model = None
            # name_suffix = '_nominal_new_QR'
            # name_suffix = '_nominal_QR_v2'
            name_suffix = '_nominal_QR_v2_repeat'

        self.inference_delan = False
        if self.inference_delan:
            # model_folder = 'learned_dynamics/models/res_model/panda_good_models/'
            # filename = 'lr_sch_Exp_hist_15_lstm_in_21_h_10_out_10_d_5_act_ld_Softplus_epochs_3000_nw_inertia_30_20_nw_pot_30_20_pin__mb_1024_norm_tau_1_ref_exec_init3_panda_rand_envs_100_kf_0_samples_515000_stime_10.torch'

            model_folder = 'learned_dynamics/models/res_model/panda_good_models/good_models_2025_02_24/'
            filename = 'lr_sch_No_hist_15_lstm_in_21_h_10_out_10_d_5_act_ld_Softplus_epochs_3000_nw_inertia_30_20_nw_pot_30_20_pin_noise__mb_1024_norm_tau_1_ref_exec_init3_rand_envs_nom_101_kf_0_samples_1040300_stime_10.torch'

            load_file = model_folder + filename
            print(f'Loading mode for inference {load_file}')

            state = torch.load(load_file, map_location=torch.device('cpu'), weights_only=False)
            self.torch_delan_model = ContextAwareDeLaN(self.nq, **state['hyper'])
            self.torch_delan_model.load_state_dict(state['state_dict'])
            self.torch_delan_model = self.torch_delan_model.cpu()

            self.torch_zeros = torch.zeros(1,self.nq).float().to(self.torch_delan_model.device)

            self.torch_enc_input_freq = 2
            filter_Ts = 1.0 / 50.0
            self.torch_enc_input_alpha = filter_Ts / (filter_Ts + 1.0 / (2 * np.pi * self.torch_enc_input_freq))
            print(f'Torch Enc input filtered {self.torch_enc_input_alpha}')
            self.torch_enc_input_filtered_old = np.zeros(10)

        # # Setup MPC
        RTI_mode = True
        self.mpc = AcadosMPC(N_horizon = self.N_horizon,
                       T_horizon = self.T_horizon,
                       expl_dyn = self.expl_dyn,
                       delan_model = self.delan_model,
                       realtime_approx = realtime_approx,
                       name_suffix = name_suffix,
                       RTI_mode = RTI_mode,
                    )
        self.nx = self.mpc.nx
        self.nu = self.mpc.nu
        self.pin_ee_id = self.mpc.robot_model.model_pin.getFrameId('end_effector')

        # Create Casadi model to get nominal model using pinocchio
        self.cs_robot_model = self.mpc.robot_model

        # Historical Data
        self.mpc_torque_debug = np.zeros(self.nq)
        self.debug_enc_input = np.zeros(10)

    def init_move_avg(self):
        self.mavg_window_size = 1
        self.mavg_diff_tau_data = np.zeros((self.mavg_window_size, self.nq))

    def update_move_avg(self, new_diff_tau):
        self.mavg_diff_tau_data = np.roll(self.mavg_diff_tau_data, -1, axis=0)
        self.mavg_diff_tau_data[-1,:] = np.copy(new_diff_tau)
        self.mavg_diff_tau = np.mean(self.mavg_diff_tau_data, axis=0)
        # print(f'\n new hist')
        # print(self.mavg_diff_tau_data)
        # print(f'new avg {self.mavg_diff_tau}')
        return self.mavg_diff_tau

    def init_kf_fee(self, q_init, qd_init, tau_init, Ts):
        x_init = np.concatenate((qd_init, q_init))
        self.tau_old_kf = tau_init
        self.debug_kf_fee_est = np.zeros(3)
        self.debug_kf_tau_est = np.zeros_like(qd_init)

        self.kf_state_Fee = KFStateFee(self.mpc.robot_model.model_pin,
                                self.mpc.robot_model.data_pin,
                                Ts,
                                q_init,
                                qd_init,
                                Fee_init=None,
                                u_init=tau_init,
                                )




    def get_mpc_state(self, joint_state_msg):
        q = np.array(joint_state_msg.position[:7])
        qd = np.array(joint_state_msg.velocity[:7])
        x_mpc = np.concatenate((qd, q))
        return x_mpc, q, qd
    
    def get_mpc_ref(self, time_index):
        if self.flag_init_mpc:
            remaining_steps = self.q_ref_traj.shape[1] - time_index
            if remaining_steps > self.N_horizon + 1:
                q_ref = self.q_ref_traj[:,time_index:(time_index+self.N_horizon+1)]
                qd_ref = self.qd_ref_traj[:,time_index:(time_index+self.N_horizon+1)]

            elif remaining_steps < 2:
                # q_ref = np.tile(self.q_ref_traj[:, -1:], (1, self.N_horizon + 1))
                # qd_ref = np.tile(self.qd_ref_traj[:, -1:], (1, self.N_horizon + 1))

                if self.ref_type == 'PICK_AND_PLACE':
                    # Hold last reference
                    last_q_ref = self.q_ref_traj[:,-1]
                    q_ref = np.tile(np.copy(last_q_ref).reshape((self.nq,1)), (1, self.N_horizon + 1))
                else:
                    # Set initial position as reference
                    q_ref = np.tile(np.copy(self.init_joint_pos).reshape((self.nq,1)), (1, self.N_horizon + 1))
                qd_ref = np.zeros((self.nq, self.N_horizon + 1))

                if self.end_traj is False:
                    self.end_traj = True
                    print(f'End trajectory')
                    logger_final = copy.deepcopy(self.logger)
                    logger_final.convert_log_data_to_np()
                    traj_length = int(self.exp_total_time / self.mpc_period)
                    rms_q_track_error = np.sqrt(np.mean(np.square(logger_final.logged_data['qp_ref'][:traj_length,:]-logger_final.logged_data['qp'][:traj_length,:]),axis=0))
                    rms_qd_track_error = np.sqrt(np.mean(np.square(logger_final.logged_data['qv_ref'][:traj_length,:]-logger_final.logged_data['qv'][:traj_length,:]),axis=0))
                    rms_x_pred_error = np.sqrt(np.mean(np.square(logger_final.logged_data['xpred_error'][:traj_length,:]),axis=0))
                    print(f'rms_q_track_error {rms_q_track_error}')
                    print(f'rms_qd_track_error {rms_qd_track_error}')
                    print(f'rms_q_pred_error {rms_x_pred_error[self.nq:]}')
                    print(f'rms_qd_pred_error {rms_x_pred_error[:self.nq]}')
                    print(f'avg solver time {np.mean(logger_final.logged_data["time_solver"])}')
                    print(f'avg pub cmd time {np.mean(logger_final.logged_data["time_pub"])}')
                    print(f'avg full update time {np.mean(logger_final.logged_data["time_update"])}')
                    print(f'avg enc time {np.mean(logger_final.logged_data["time_enc"])}')
                    
                    
                    print(f'Saving data')
                    init_time = time.time()
                    dataset = {}
                    for key in logger_final.logged_data.keys():
                        # if run == 0:
                        if True:
                            dataset[key] = []
                        dataset[key].append(logger_final.logged_data[key])
                    folder_name = 'datasets_real/2025_03_05/'
                    n_samples = dataset['t'][-1].shape[0]
                    # model_str = 'acc_nom_mpc_'
                    # model_str = model_str + 'cte_ref_'
                    # model_str = model_str + 'q1_ref_'
                    # model_str = model_str + 'q1235_ref_'
                    # model_str = model_str + '2kg_load_q1_ref'
                    # model_str = model_str + '3kg_load_static_ref'
                    if self.delan_model is None:
                        model_str = 'nom_'
                    elif self.delan_model == 'KF':
                        model_str = 'kf_' + str(self.kf_state_Fee.nFee) + '_'
                    else:
                        model_str = 'delan_v4_'
                    # model_str = model_str + 'gripper_'
                    # model_str = model_str + '_q456_exc'
                    # model_str = model_str + '_q0123456_exc_ref_v4_'
                    model_str = model_str + '1kg_'
                    # model_str = model_str + '2kg_'
                    # model_str = model_str + '3kg_'
                    # model_str = model_str + 'gripper_1kg_'
                    # model_str = model_str + 'gripper_2kg_'
                    # model_str = model_str + 'gripper_3kg_'
                    if self.ref_type == 'FULL_INF':
                        # model_str = model_str + '_full_inf_v7_'
                        model_str = model_str + '_full_inf_v8_theta_0_'
                    elif self.ref_type == 'PICK_AND_PLACE':
                        model_str = model_str + '_PP_v8_'
                    else:
                        # model_str = model_str + '_setpoint_'
                        # model_str = model_str + '_q1_ref_v2_'
                        # model_str = model_str + '_q1_ref_v3_'
                        # model_str = model_str + 'q1_ref_'
                        model_str = model_str + 'q12_ref_'
                        # model_str = model_str + 'q123_ref_'
                        # model_str = model_str + 'q1235_ref_'
                        # model_str = model_str + 'q012356_ref_'
                    model_str = model_str + 'QR_V2_'
                    filename = model_str + 'samples_' + str(n_samples) + '_dataset_mpc.pkl'
                    with open(folder_name + filename, 'wb') as fp:
                        pickle.dump(dataset, fp)
                        print(f'dictionary saved successfully to file {filename} | Nsamples {n_samples}')
                    print(f'diff time {time.time() - init_time}')

            else:
                q_ref = self.q_ref_traj[:, time_index:]
                qd_ref = self.qd_ref_traj[:, time_index:]

                # Pad the references with the last value to complete the horizon size
                q_ref = np.pad(q_ref, ((0, 0), (0, 1 + self.N_horizon - remaining_steps)), mode='edge')
                qd_ref = np.pad(qd_ref, ((0, 0), (0, 1 + self.N_horizon - remaining_steps)), mode='edge')
                # print(f'q_ref {q_ref}')

        else:
            q_ref = np.tile(np.copy(self.init_joint_pos).reshape((self.nq,1)), (1, self.N_horizon + 1))
            qd_ref = np.zeros((self.nq, self.N_horizon + 1))

        x_ref = np.vstack((qd_ref,
                           q_ref))
        return x_ref, q_ref, qd_ref
    
    ## Hist Data
    def init_hist_data(self):
        self.hist_q = np.tile(self.pos_safe_init, (self.hist_length, 1))
        self.hist_qd = np.zeros((self.hist_length, self.nq))
        self.hist_diff_tau_nom = np.zeros((self.hist_length, self.nq))

    def update_hist_data(self, q_old, qd_old, qdd, tau, tau_nom = None):
        # Roll data
        self.hist_q = np.roll(self.hist_q, -1, axis=0)
        self.hist_qd = np.roll(self.hist_qd, -1, axis=0)
        self.hist_diff_tau_nom = np.roll(self.hist_diff_tau_nom, -1, axis=0)

        # Get torque from nominal model
        if tau_nom is None:
            tau_nom = self.mpc.robot_model.tau_nom_inv_dyn_fn(q_old, qd_old, qdd)
            tau_nom = np.array(tau_nom).squeeze()
        diff_nom_torque = tau - tau_nom

        # Update data
        self.hist_q[-1,:] = q_old
        self.hist_qd[-1,:] = qd_old
        self.hist_diff_tau_nom[-1,:] = np.copy(self.update_move_avg(diff_nom_torque))
        # print(f'self.hist_diff_tau_nom\n {self.hist_diff_tau_nom}')

    ## Reference
    def init_full_infinity_ref(self, q0):
        self.ref_gen = ReferenceGenerator()
        inf_ee_pos_init = pin_get_link_pos(copy.deepcopy(self.mpc.robot_model.model_pin),
                                           copy.deepcopy(self.mpc.robot_model.data_pin),
                                           q0,
                                           self.pin_ee_id)
        print(f'inf_ee_pos_init {inf_ee_pos_init}')
        # freq_traj = 0.5
        # amp = np.array([0, 0.30, 0.15]) 
        # theta = -20/180*np.pi
        freq_traj = 0.7
        amp = np.array([0, 0.40, 0.15])
        theta = -0/180*np.pi
        
        inf_params = {}
        # inf_params['amp'] = np.array([0, 0.2, 0.2])
        # inf_params['amp'] = np.array([0, 0.5, 0.25])
        # inf_params['theta'] = -30/180*np.pi
        # inf_params['amp'] = np.array([0, 0.25, 0.15])
        # inf_params['amp'] = np.array([0, 0.20, 0.1]) # freq 0.4
        # inf_params['amp'] = np.array([0, 0.30, 0.12]) # freq 0.5
        # inf_params['amp'] = np.array([0, 0.30, 0.15]) # freq 0.5
        inf_params['amp'] = amp
        inf_params['theta'] = theta
        inf_params['freq'] = freq_traj
        # inf_params['theta'] = theta
        inf_params['center_pos'] = inf_ee_pos_init
        inf_params['Ts'] = self.mpc_period

        Nrepeat = int(np.ceil(self.exp_total_time / (1.0 / freq_traj))) + 1

        
        time_traj, q_traj, qd_traj, pos_traj, vel_traj = self.ref_gen.generate_full_infinity_joint(model_pin = self.mpc.robot_model.model_pin,
                                                                                                   data_pin = self.mpc.robot_model.data_pin,
                                                                                                   params = inf_params,
                                                                                                   q0 = q0,
                                                                                                   return_ee_traj = True,
                                                                                                   Nrepeat = Nrepeat)
        traj_length = int(self.exp_total_time / self.mpc_period)
        self.full_inf_time_traj = np.arange(traj_length + self.N_horizon + 10) * self.mpc_period
        self.full_inf_q_traj = q_traj.T
        self.full_inf_qd_traj = qd_traj.T
        self.full_inf_ee_pos_traj = pos_traj.T
        self.full_inf_ee_vel_traj = vel_traj.T

        print(f'full_inf_time_traj shape {self.full_inf_time_traj.shape}')
        print(f'full_inf_q_traj shape {self.full_inf_q_traj.shape}')

        compute_qd_using_diff = True
        if compute_qd_using_diff:
            qd_diff_traj = []
            traj_length = self.full_inf_q_traj.shape[1]
            for i in range(traj_length-1):
                qd_new = (self.full_inf_q_traj[:,i+1] - self.full_inf_q_traj[:,i]) / self.mpc_period
                qd_diff_traj.append(qd_new)
            qd_diff_traj.append(qd_new) # Hold last one
            self.full_inf_qd_traj = np.array(qd_diff_traj).T

        if self.ref_init_length > 0:

            init_q_ref = np.tile(np.copy(q0).reshape((self.nq,1)), (1, self.ref_init_length))
            init_qd_ref = np.zeros_like(init_q_ref)

            init_ee_pos = np.copy(self.full_inf_ee_pos_traj[:,0])
            init_ee_pos = np.tile(np.copy(init_ee_pos).reshape((3,1)), (1, self.ref_init_length))
            init_ee_vel = np.zeros_like(init_ee_pos)

            self.full_inf_q_traj = np.hstack((init_q_ref, self.full_inf_q_traj))
            self.full_inf_qd_traj = np.hstack((init_qd_ref, self.full_inf_qd_traj))
            self.full_inf_ee_pos_traj = np.hstack((init_ee_pos, self.full_inf_ee_pos_traj))
            self.full_inf_ee_vel_traj = np.hstack((init_ee_vel, self.full_inf_ee_vel_traj))

        self.q_ref_traj = self.full_inf_q_traj
        self.qd_ref_traj = self.full_inf_qd_traj


    def init_pick_and_place_ref(self, q0):

        pp_ee_pos_init = pin_get_link_pos(copy.deepcopy(self.mpc.robot_model.model_pin),
                                    copy.deepcopy(self.mpc.robot_model.data_pin),
                                    q0,
                                    self.pin_ee_id)

        pp_params = {}
        # pp_params['amp'] = np.array([0, 0.35, 0.35]) # Desired ones
        # pp_params['freq'] = 0.2 # Desired ones

        pp_params['amp'] = np.array([0, 0.35, 0.35])
        pp_params['freq'] = 0.2
        pp_params['theta'] = 90/180*np.pi
        pp_params['init_pos'] = pp_ee_pos_init
        pp_params['center_pos'] = pp_ee_pos_init + np.array([0.1, -0.15, -0.02])
        pp_params['Ts'] = self.mpc_period
        pp_params['phase_offset'] = 45/180*np.pi
        pp_params['grasp_time'] = 3.0
        pp_params['spline_height'] = 0.3
        pp_params['spline_time'] = 1.5
        pp_params['spline_x_diff'] = 0.0
        pp_params['spline_y_diff'] = 0.30

        full_time_traj, pp_q_traj, pp_qd_traj, pp_ee_pos_traj, pp_ee_vel_traj, gripper_events = self.ref_gen.compute_pick_and_place_traj(model_pin = self.mpc.robot_model.model_pin,
                                                                                                   data_pin = self.mpc.robot_model.data_pin,
                                                                                                   full_param = pp_params, q0 = q0, return_time_events=True)

        self.full_pp_q_traj = pp_q_traj.T
        self.full_pp_qd_traj = pp_qd_traj.T
        self.full_pp_ee_pos_traj = pp_ee_pos_traj.T
        self.full_pp_ee_vel_traj = pp_ee_vel_traj.T
        self.pp_gripper_events = gripper_events

        print(f'full_inf_q_traj shape {self.full_pp_q_traj.shape}')

        compute_qd_using_diff = True
        if compute_qd_using_diff:
            qd_diff_traj = []
            traj_length = self.full_pp_q_traj.shape[1]
            for i in range(traj_length-1):
                qd_new = (self.full_pp_q_traj[:,i+1] - self.full_pp_q_traj[:,i]) / self.mpc_period
                qd_diff_traj.append(qd_new)
            qd_diff_traj.append(qd_new) # Hold last one
            self.full_pp_qd_traj = np.array(qd_diff_traj).T

        if self.ref_init_length > 0:

            init_q_ref = np.tile(np.copy(q0).reshape((self.nq,1)), (1, self.ref_init_length))
            init_qd_ref = np.zeros_like(init_q_ref)

            init_ee_pos = np.copy(self.full_pp_ee_pos_traj[:,0])
            init_ee_pos = np.tile(np.copy(init_ee_pos).reshape((3,1)), (1, self.ref_init_length))
            init_ee_vel = np.zeros_like(init_ee_pos)

            self.full_pp_q_traj = np.hstack((init_q_ref, self.full_pp_q_traj))
            self.full_pp_qd_traj = np.hstack((init_qd_ref, self.full_pp_qd_traj))
            self.full_pp_ee_pos_traj = np.hstack((init_ee_pos, self.full_pp_ee_pos_traj))
            self.full_pp_ee_vel_traj = np.hstack((init_ee_vel, self.full_pp_ee_vel_traj))

        self.q_ref_traj = self.full_pp_q_traj
        self.qd_ref_traj = self.full_pp_qd_traj
        self.full_pp_time_traj = np.arange(self.full_pp_q_traj.shape[1]) * self.mpc_period

        total_traj_time = self.full_pp_time_traj[-1]
        total_traj_length = self.q_ref_traj.shape[1]
        print(f'total_traj_length {total_traj_length} total_traj_time {total_traj_time}')
        if total_traj_time < self.exp_total_time:
            print(f'traj too short')
        else:
            print(f'traj too big')

        self.pp_gripper_enable = True
        self.pp_event_id = 0



    def init_mpc_sin_ref(self):
        # self.q_sin_ref_amp = np.random.uniform(low=np.zeros((self.nq,1)), high=np.array([[1.5, 1.0, 1.5, 1.0, 1.5, 1.0, 1.5]]).T, size=(self.nq,1))
        # self.q_sin_ref_freq = np.random.uniform(low=np.zeros((self.nq,1)), high=0.15*np.ones((self.nq,1)), size=(self.nq,1))
        self.q_sin_ref_amp = np.zeros((self.nq,1))
        self.q_sin_ref_freq = np.zeros((self.nq,1))
        # self.q_sin_ref_amp = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((self.nq,1))
        # self.q_sin_ref_freq = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((self.nq,1))
        self.q_sin_ref_amp = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((self.nq,1)) # tested ref
        self.q_sin_ref_freq = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((self.nq,1))
        # self.q_sin_ref_amp = np.array([0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((self.nq,1)) # v2
        # self.q_sin_ref_freq = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((self.nq,1))
        # self.q_sin_ref_amp = np.array([0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((self.nq,1)) # v3
        # self.q_sin_ref_freq = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape((self.nq,1))

        # self.q_sin_ref_amp = np.array([0.0, 0.15, 0.15, 0.0, 0.0, 00, 0.0]).reshape((self.nq,1))
        # self.q_sin_ref_freq = np.array([0.0, 0.8, 0.7, 0.0, 0.0, 0.0, 0.0]).reshape((self.nq,1))

        # self.q_sin_ref_amp = np.array([0.0, 0.15, 0.15, 0.15, 0.0, 0.0, 0.0]).reshape((self.nq,1))
        # self.q_sin_ref_freq = np.array([0.0, 0.8, 0.7, 0.2, 0.0, 0.0, 0.0]).reshape((self.nq,1))
        # self.q_sin_ref_amp = np.array([0.0, 0.1, 0.1, 0.2, 0.3, 0.0, 0.0]).reshape((self.nq,1))
        # self.q_sin_ref_freq = np.array([0.0, 0.8, 0.6, 0.15, 0.15, 0.0, 0.0]).reshape((self.nq,1))
        # self.q_sin_ref_amp = np.array([0.0, 0.15, 0.15, 0.15, 0.0, 0.3, 0.0]).reshape((self.nq,1))
        # self.q_sin_ref_freq = np.array([0.0, 0.8, 0.7, 0.2, 0.0, 0.1, 0.0]).reshape((self.nq,1))
        # self.q_sin_ref_amp = np.array([0.0, 0.1, 0.1, 0.1, 0.05, 0.02, 0.0]).reshape((self.nq,1))
        # self.q_sin_ref_freq = np.array([0.0, 0.8, 0.6, 0.15, 0.15, 0.1, 0.0]).reshape((self.nq,1))

        # Reference working for Delan MPC
        # self.q_sin_ref_amp = np.array([0.1, 0.15, 0.15, 0.15, 0.2, 0.3, 0.2]).reshape((self.nq,1))
        # self.q_sin_ref_freq = np.array([0.2, 0.8, 0.7, 0.2, 0.1, 0.1, 0.1]).reshape((self.nq,1))

        # EXC System only with gripper
        # self.q_sin_ref_amp = np.array([0.2, 0.3, 0.2, 0.3, 0.5, 0.3, 0.3]).reshape((self.nq,1)) #v3
        # self.q_sin_ref_freq = np.array([0.5, 0.3, 0.5, 0.15, 0.2, 0.2, 0.2]).reshape((self.nq,1))
        # self.q_sin_ref_amp = np.array([0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5]).reshape((self.nq,1)) #v4
        # self.q_sin_ref_freq = np.array([0.5, 0.3, 0.5, 0.25, 0.2, 0.2, 0.2]).reshape((self.nq,1))

        # self.q_sin_ref_amp = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.5, 1.5]).reshape((self.nq,1)) 
        # self.q_sin_ref_freq = np.array([0.0, 0.0, 0.0, 0.0, 0.15, 0.2, 0.2]).reshape((self.nq,1))


        self.q_sin_ref_phase = np.zeros((self.nq,1))
        self.q_sin_ref_offset = np.copy(self.pos_safe_init).reshape((self.nq,1))
        # self.q_sin_ref_offset = np.copy(self.init_joint_pos).reshape((self.nq,1))
        print(f'Offset qsin reference {self.q_sin_ref_offset}')

    def generate_reference(self, total_traj_time):
        traj_length = int(total_traj_time / self.mpc_period)
        print(f'traj_length {traj_length}')
        time_traj = np.arange(traj_length+self.N_horizon+10)*self.mpc_period
        self.q_ref_traj = self.q_sin_ref_amp * np.sin(2 * np.pi * self.q_sin_ref_freq * time_traj + self.q_sin_ref_phase) + self.q_sin_ref_offset
        self.qd_ref_traj = 2 * np.pi * self.q_sin_ref_freq * self.q_sin_ref_amp * np.cos(2 * np.pi * self.q_sin_ref_freq * time_traj + self.q_sin_ref_phase)

        if self.ref_init_length > 0:

            init_q_ref = np.copy(self.q_ref_traj[:,0])
            init_q_ref = np.tile(np.copy(init_q_ref).reshape((self.nq,1)), (1, self.ref_init_length))
            init_qd_ref = np.zeros_like(init_q_ref)

            self.q_ref_traj = np.hstack((init_q_ref, self.q_ref_traj))
            self.qd_ref_traj = np.hstack((init_qd_ref, self.qd_ref_traj))

    def get_enc_input(self):
        if self.delan_model is None or self.delan_model == 'KF':
            enc_input = np.zeros(1)
        else:
            if self.hist_length == 0:
                enc_input = np.zeros(self.delan_model.n_enc_input)
                enc_input[self.env_id] = 1
            else:
                # Call lstm
                lstm_input = np.concatenate((self.hist_q, self.hist_qd, self.hist_diff_tau_nom), axis=-1)
                raw_enc_input = self.delan_model.eval_lstm_np(lstm_input)

                if self.filter_enc_input_freq > 100:
                    enc_input = raw_enc_input
                else:
                    enc_input = first_order_low_pass_filter(raw_enc_input, self.filter_enc_input_filtered_old, self.filter_enc_input_alpha)
                    self.filter_enc_input_filtered_old = np.copy(enc_input)

        return enc_input

    def reset_exp(self, reset_joint_state):
        # self.init_loggger_data('test_run')
        self.exp_total_time = 26
        
        if self.ref_type == 'FULL_INF':
            self.exp_total_time = 20
            self.ref_gen = ReferenceGenerator()

            inf_q_init = np.array(reset_joint_state.position[:7]).reshape((self.nq,))
            self.init_full_infinity_ref(inf_q_init)

        elif self.ref_type == 'PICK_AND_PLACE':
            self.exp_total_time = 26
            self.ref_gen = ReferenceGenerator()

            pp_q_init = np.array(reset_joint_state.position[:7]).reshape((self.nq,))
            self.init_pick_and_place_ref(pp_q_init)

        else:
            self.init_mpc_sin_ref()
            self.generate_reference(self.exp_total_time)

        if self.hist_length > 0:
            self.init_hist_data()

        # Initialize solver
        x0, _, _ = self.get_mpc_state(reset_joint_state)
        # u0 = np.zeros((self.nu,))
        u0 = np.array(reset_joint_state.effort[:7]).reshape((self.nu,))
        self.mpc.init_solver(x0, u0)
        self.debug_xpred = x0

        self.logger.reset_logger('test_run', x0)
        self.logger.set_np_ignored_keys(['run_name', 'labels', 'env_id'])

        self.exp_steps = 0
        print('Finish reset!')

    ## Kalman Filter
    def update_state_kf(self, q, qd, tau):
        self.q_old_kf = q
        self.qd_old_kf = qd
        self.tau_old_kf = tau

    def pub_ffwd_cmd(self, pos, vel, torque):
        new_cmd = FFWDTorquePDUpdate()
        new_cmd.pos = list(pos)
        new_cmd.vel = list(vel)
        new_cmd.torque = list(torque)
        self.ffwd_cmd_pub.publish(new_cmd)


    def update_gravity_comp(self, joint_state_msg):
        des_pos = self.init_joint_pos + 0*np.array([0.0, -0.2, 0.0, 0.0, 0.0,  0.0, 0.0])
        des_vel = np.zeros_like(des_pos)
        # des_tau = np.zeros_like(des_pos)
        des_tau = np.array(self.cs_robot_model.cpin_tau_cg_fn(des_pos, des_vel)).reshape(-1)
        des_tau = des_tau 
        self.pub_ffwd_cmd(des_pos, des_vel, des_tau)
        # print(des_pos)
        # print(des_tau)

    def update_mpc(self, joint_state_msg):

        init_update_time = time.time()

        ## Update MPC State
        x_mpc, q, qd = self.get_mpc_state(joint_state_msg)
        init_enc_time = time.time()
        enc_input = self.get_enc_input()
        self.debug_enc_input = np.copy(enc_input)
        self.enc_time = time.time() - init_enc_time
        self.total_time_enc += self.enc_time

        # Get Reference
        init_ref_time = time.time()
        y_ref, self.q_pos_ref_horizon, self.q_vel_ref_horizon = self.get_mpc_ref(self.exp_steps)
        self.total_time_ref += time.time() - init_ref_time
        qp_ref = self.q_pos_ref_horizon[:,0]
        qd_ref = self.q_vel_ref_horizon[:,0]

        # Log current state
        init_log_time = time.time()
        # self.log_new_data(self.exp_steps * self.joint_ctrl_period, qp_ref = qp_ref, qv_ref = qd_ref)
        self.total_time_log += time.time() - init_log_time

        # Update MPC reference
        torque_ref = np.zeros((self.mpc.nu, self.N_horizon))
        self.mpc.set_mpc_ref(y_ref, torque_ref)

        ## Update Kalman Filter
        kf_fee_est, kf_tau_est = self.kf_state_Fee.update(x_mpc[self.nq:,], x_mpc[:self.nq,], self.tau_old_kf)
        self.debug_kf_fee_est = np.copy(kf_fee_est)
        self.debug_kf_tau_est = np.copy(kf_tau_est)

        init_param_comp = time.time()
        # if self.delan_model is not None and self.realtime_approx != RealtimeApprox.NO_APPROX:
        # tau_kf = np.zeros_like(torque_ref)
        # fee_est_new = np.zeros(3)
        if self.delan_model == 'KF':
            # self.log_kf_data()
            param = kf_tau_est.reshape((-1,1))
            # value_tf = np.zeros_like(kf_tau_est)
            # param = value_tf.reshape((-1,1))
        elif self.delan_model is not None:
            param = self.compute_delan_param(enc_input)
        self.total_time_param += time.time() - init_param_comp

        # Update MPC Parameters
        if self.mpc.model.p.shape[0] > 0.0:
            self.mpc.set_mpc_param(param)

        # Compute Control MPC
        self.mpc.iterate_solver_warm_start()
        start_solver_time = time.time()
        u_mpc = self.mpc.solver.solve_for_x0(x_mpc, False, False)
        self.solver_time = time.time() - start_solver_time
        self.total_time_solver += np.copy(self.solver_time)
        self.solver_status = self.mpc.solver.get_status()

        # Save data for historical data
        q_old = np.copy(q)
        qd_old = np.copy(qd)
        self.tau_old_kf = np.copy(u_mpc)

        # Save data for Kalman Filter
        self.update_state_kf(np.copy(q_old), np.copy(qd_old), np.copy(u_mpc))

        # Publish Command
        q_mpc_torque = u_mpc
        self.mpc_torque_debug = np.copy(q_mpc_torque)
        # print(f'mpc torque {q_mpc_torque}')
        if self.flag_init_mpc:
            # rospy.sleep(0.010)
            self.pub_ffwd_cmd(qp_ref, qd_ref, q_mpc_torque)

        self.pub_cmd_time = time.time() - init_update_time
        self.total_time_pub += np.copy(self.pub_cmd_time)

        ## Update Logger
        if self.flag_init_mpc:
            u_horizon, xpred_horizon = self.mpc.get_mpc_solution_full_horizon()
            self.debug_xpred = xpred_horizon[:,1]
            ee_pos, ee_vel = self.get_ee_data(np.copy(q), np.copy(qd))
            qdd = np.array(joint_state_msg.acceleration[:7])
            tau_read = np.array(joint_state_msg.effort[:7])

            if self.ref_type == 'FULL_INF':
                traj_length = self.full_inf_ee_pos_traj.shape[1]
                if self.exp_steps < traj_length:
                    ee_pos_ref = np.copy(self.full_inf_ee_pos_traj[:,self.exp_steps])
                    ee_vel_ref = np.copy(self.full_inf_ee_vel_traj[:,self.exp_steps])
                else:
                    ee_pos_ref = np.copy(self.full_inf_ee_pos_traj[:,-1])
                    ee_vel_ref = np.copy(self.full_inf_ee_vel_traj[:,-1])

            elif self.ref_type == 'PICK_AND_PLACE':
                traj_length = self.full_pp_ee_pos_traj.shape[1]
                if self.exp_steps < traj_length:
                    ee_pos_ref = np.copy(self.full_pp_ee_pos_traj[:,self.exp_steps])
                    ee_vel_ref = np.copy(self.full_pp_ee_vel_traj[:,self.exp_steps])
                else:
                    ee_pos_ref = np.copy(self.full_pp_ee_pos_traj[:,-1])
                    ee_vel_ref = np.copy(self.full_pp_ee_vel_traj[:,-1])
            else:
                ee_pos_ref = np.zeros(3)
                ee_vel_ref = np.zeros(3)

            self.logger.log_real_data(time = self.exp_steps * self.mpc_period,
                                    q = np.copy(q), qd = np.copy(qd),
                                    qdd = np.copy(qdd), tau_read = np.copy(tau_read), tau_ctrl = u_mpc,
                                    ee_pos = ee_pos, ee_vel = ee_vel,
                                    tau_kf = kf_tau_est, fext_kf = kf_fee_est,
                                    u_horizon = u_horizon, xpred_horizon = xpred_horizon,
                                    env_id = 0, enc_input = enc_input,
                                    model_pin = self.mpc.robot_model.model_pin, data_pin = self.mpc.robot_model.data_pin,
                                    mpc_cost = self.mpc.solver.get_cost(), mpc_times = self.mpc.get_solver_times(),
                                    qp_ref = np.copy(self.q_pos_ref_horizon[:,0]), qv_ref = np.copy(self.q_vel_ref_horizon[:,0]),
                                    ee_pos_ref = ee_pos_ref, ee_vel_ref = ee_vel_ref,
                                    )
            self.logger.add_time_data(self.solver_time, self.pub_cmd_time,
                                      self.update_time, self.enc_time)

        ## Updated predicted data and hist data
        in_hist_q = np.copy(q_old)

        # in_hist_qd = np.zeros_like(qd_old)
        in_hist_qd = np.copy(qd_old)

        # in_hist_qdd = np.zeros_like(qd_old)
        in_hist_qdd = np.copy(np.array(joint_state_msg.acceleration[:7]))

        # in_hist_torque = np.copy(q_mpc_torque)
        in_hist_torque = np.copy(np.array(joint_state_msg.effort[:7]))
        in_hist_torque_nom = np.copy(np.array(joint_state_msg.tau_nom_filtered[:7]))

        if self.hist_length > 0:
            self.update_hist_data(in_hist_q, in_hist_qd, 
                                  in_hist_qdd, in_hist_torque,
                                  in_hist_torque_nom)

        param_mpc_dyn = param[0] if self.mpc.model.p.shape[0] > 0.0 else None
        # self.log_torque_mpc_dynamics(q_old, qd_old, qdd_new, q_mpc_torque, param_mpc_dyn)
        # self.log_pred_error()

        if self.ref_type == 'PICK_AND_PLACE' and self.pp_gripper_enable:
            if self.pp_event_id < len(self.pp_gripper_events['event']):
                current_time = self.exp_steps * self.mpc_period
                if current_time == self.pp_gripper_events['times'][self.pp_event_id]:
                    if self.pp_gripper_events['event'][self.pp_event_id]:
                        self.close_gripper_no_wait()
                    else:
                        self.open_gripper_no_wait()
                    self.pp_event_id = self.pp_event_id + 1

        self.update_time = time.time() - init_update_time
        self.total_time_update += np.copy(self.update_time)

    def compute_delan_param(self, enc_input):
        if enc_input.shape[0] > 1:
            enc_input_horizon = np.tile(enc_input, (self.N_horizon, 1))
        else:
            enc_input_horizon = None

        param = enc_input_horizon

        return param

    def update_lqr(self, joint_state_msg):
        dt = self.control_period
        qp = np.array(joint_state_msg.position[:7])
        qv = np.array(joint_state_msg.velocity[:7])

        Minertia = np.array(self.cs_robot_model.cpin_H_fn(qp))
        Mq_inv = np.linalg.inv(Minertia)

        # Get Current Reference
        y_ref, self.q_pos_ref_horizon, self.q_vel_ref_horizon = self.get_mpc_ref(self.exp_steps)
        qp_ref = self.q_pos_ref_horizon[:,0]
        qv_ref = self.q_vel_ref_horizon[:,0]

        ## State space update
        ncord = 7
        nx = 2*ncord
        nu = ncord
        Ac = np.zeros((nx,nx))
        Bc = np.zeros((nx,nu))
        Ac[:ncord,ncord:] = np.eye(ncord)
        Bc[ncord:,:] = Mq_inv

        Ad = np.eye(nx) + dt*Ac
        Bd = dt*Bc

        # LQR weight matrix
        Qpos = 100*np.ones(ncord)
        Qvel = 1000*np.ones(ncord)
        Q = np.diag(np.concatenate((Qpos, Qvel)))
        # R = 1*np.eye(nu)
        R = 0.01*np.eye(nu)

        # Solve discrete Riccati equation.
        P = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)

        # Compute the feedback gain matrix K.
        Klqr = np.linalg.inv(R + Bd.T @ P @ Bd) @ Bd.T @ P @ Ad

        x = np.concatenate((qp, qv))
        x_ref = np.concatenate((qp_ref, qv_ref))
        x_error = x_ref - x
        tau_lqr = Klqr @ x_error

        tau_cmp = np.array(self.cs_robot_model.cpin_tau_cg_fn(qp_ref, qv_ref)).reshape(-1)

        q_torque_cmd = tau_lqr + tau_cmp

        self.pub_ffwd_cmd(qp_ref, qv_ref, q_torque_cmd)


    def control_callback(self, event):
        # print('new cb')
        joint_state_msg = copy.copy(self.joint_state_listener.get_data())

        # self.update_gravity_comp(joint_state_msg)
        if self.control_mode == 'MPC':
            self.update_mpc(joint_state_msg)
            if self.flag_init_mpc:
                self.exp_steps += 1
            else:
                self.flag_init_mpc = True

        if self.control_mode == 'LQR':
            self.update_lqr(joint_state_msg)
            self.exp_steps += 1


    def get_ee_data(self, q_new, qd_new):
        ee_pos = pin_get_link_pos(self.mpc.robot_model.model_pin,
                                  self.mpc.robot_model.data_pin,
                                  q_new,
                                  self.pin_ee_id)
        pin_jac = pin_get_frame_jacobian(self.mpc.robot_model.model_pin,
                                         self.mpc.robot_model.data_pin,
                                         qd_new,
                                         self.pin_ee_id)
        ee_vel = pin_jac @ qd_new
        return ee_pos, ee_vel

    def debug_callback(self, event):
        debug_msg = MPCDebug()

        debug_joint_state = copy.copy(self.joint_state_listener.get_data())

        x_mpc, q, qd = self.get_mpc_state(debug_joint_state)

        # qdd = debug_joint_state.aceleration
        tau_read = np.array(debug_joint_state.effort[:7])
        qdd_est = np.array(debug_joint_state.acceleration[:7])
        qdd = np.array(debug_joint_state.acceleration[:7]) ## Using filtered one

        y_ref, self.q_pos_ref_horizon, self.q_vel_ref_horizon = self.get_mpc_ref(self.exp_steps)
        q_ref = self.q_pos_ref_horizon[:,0]
        qd_ref = self.q_vel_ref_horizon[:,0]

        # Joint
        debug_msg.q = q.tolist()
        debug_msg.qd = qd.tolist()
        debug_msg.qdd = qdd.tolist()
        debug_msg.qdd_est = qdd_est.tolist()
        debug_msg.tau_read = tau_read.tolist()

        debug_msg.q_ref = q_ref.tolist()
        debug_msg.qd_ref = qd_ref.tolist()

        # End-Effector
        ee_pos, ee_vel = self.get_ee_data(q, qd)
        debug_msg.ee_pos = ee_pos.tolist()
        debug_msg.ee_vel = ee_vel.tolist()

        # Kalman Filter
        debug_msg.tau_kf = self.debug_kf_tau_est.tolist()
        debug_msg.fext_kf = self.debug_kf_fee_est.tolist()

        # MPC
        debug_msg.mpc_time = [self.solver_time]
        debug_msg.x_mpc = x_mpc.tolist()
        debug_msg.xpred_next = self.debug_xpred.tolist()
        debug_msg.tau_mpc = self.mpc_torque_debug.tolist()
        debug_msg.tau_ctrl = self.mpc_torque_debug.tolist()

        # Delan
        debug_msg.enc_input = np.copy(self.debug_enc_input).tolist()

        if len(self.logger.logged_data['tau_nom_pin']) > 0:
            debug_msg.tau_nom_pin = np.array(self.logger.logged_data['tau_nom_pin'][-1]).tolist()
            debug_msg.m_nom_pin = np.array(self.logger.logged_data['m_nom_pin'][-1]).tolist()
            debug_msg.c_nom_pin = np.array(self.logger.logged_data['c_nom_pin'][-1]).tolist()
            # debug_msg.g_nom_pin = np.array(self.logger.logged_data['g_nom_pin'][-1]).tolist()
            # print('debug joint')
            # print(np.copy(self.mavg_diff_tau))
            # print(np.array(debug_joint_state.tau_nom[:7]))
            debug_msg.g_nom_pin = np.array(debug_joint_state.tau_nom_filtered[:7]).tolist()
            debug_msg.diff_tau_g_nom_pin = (tau_read - np.array(debug_joint_state.tau_nom_filtered[:7])).tolist()

            # debug_msg.diff_tau_nom_pin = np.array(self.logger.logged_data['diff_tau_nom_pin'][-1]).tolist()
            debug_msg.diff_tau_nom_pin = np.copy(self.mavg_diff_tau).tolist()
            debug_msg.diff_tau_m_nom_pin = np.array(self.logger.logged_data['diff_tau_m_nom_pin'][-1]).tolist()
            debug_msg.diff_tau_c_nom_pin = np.array(self.logger.logged_data['diff_tau_c_nom_pin'][-1]).tolist()
            # debug_msg.diff_tau_g_nom_pin = np.array(self.logger.logged_data['diff_tau_g_nom_pin'][-1]).tolist()

        # if self.delan_model is not None and self.delan_model != 'KF':
        if self.inference_delan:

            # Get Nominal model
            H_nom = np.array(self.cs_robot_model.cpin_H_fn(q))
            nom_tau_cg = np.array(self.cs_robot_model.cpin_tau_cg_fn(q, qd)).reshape(-1)

            q_torch = torch.from_numpy(q).float().to(self.torch_delan_model.device).view(1, -1)
            qd_torch = torch.from_numpy(qd).float().to(self.torch_delan_model.device).view(1, -1)
            qdd_torch = torch.from_numpy(qdd).float().to(self.torch_delan_model.device).view(1, -1)
            lstm_input = np.concatenate((self.hist_q, self.hist_qd, self.hist_diff_tau_nom), axis=-1)
            lstm_input_torch = torch.from_numpy(lstm_input).float().to(self.torch_delan_model.device).view(1, self.hist_length, -1)

            delan_enc_input = self.torch_delan_model.lstm(lstm_input_torch).squeeze()
            delan_enc_input = delan_enc_input.cpu().detach().numpy()

            delan_enc_input_filtered = first_order_low_pass_filter(delan_enc_input, self.torch_enc_input_filtered_old, self.torch_enc_input_alpha)
            self.torch_enc_input_filtered_old = np.copy(delan_enc_input_filtered)

            debug_msg.enc_input = delan_enc_input_filtered.tolist()
            debug_msg.enc_input_raw = delan_enc_input.tolist()

            delan_tau_g = self.torch_delan_model.inv_dyn(q_torch, self.torch_zeros, self.torch_zeros, lstm_input_torch).squeeze()
            delan_tau_c = self.torch_delan_model.inv_dyn(q_torch, qd_torch, self.torch_zeros, lstm_input_torch).squeeze() - delan_tau_g
            delan_tau_m = self.torch_delan_model.inv_dyn(q_torch, self.torch_zeros, qdd_torch, lstm_input_torch).squeeze() - delan_tau_g

            delan_tau_g = delan_tau_g.cpu().detach().numpy()
            delan_tau_c = delan_tau_c.cpu().detach().numpy()
            delan_tau_m = delan_tau_m.cpu().detach().numpy()

            delan_tau_cg = delan_tau_c + delan_tau_g
            delan_tau = delan_tau_m + delan_tau_cg

            tau_total = H_nom @ qdd + nom_tau_cg + delan_tau

            debug_msg.tau_total_nom_delan = tau_total
            debug_msg.delan_tau = delan_tau
            debug_msg.delan_tau_m = delan_tau_m
            debug_msg.delan_tau_cg = delan_tau_cg
            debug_msg.delan_tau_c = delan_tau_c
            debug_msg.delan_tau_g = delan_tau_g

        time_values = [self.solver_time]
        time_values += [self.pub_cmd_time]
        time_values += [self.update_time]
        time_values += [self.enc_time]

        time_names = ['solver']
        time_names += ['pub']
        time_names += ['update']
        time_names += ['enc']

        debug_msg.times_values = time_values
        # debug_msg.time_names = time_names


        # Publish
        self.mpc_debug_pub.publish(debug_msg)
        
if __name__ == '__main__':
    rospy.init_node('mpc_node')
    node = CaDeLaCNode()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass