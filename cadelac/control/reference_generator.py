import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pinocchio as pin


from cadelac.control.pin_utils import *
from cadelac.control.math_utils import *
from cadelac.control.excitation_generator import obtain_valid_traj_param_simplified, generate_fourier_traj

import math

class ReferenceGenerator():    
    def __init__(self, nq = 7):
        self.nq = nq
        self.q1 = np.array([-0.477112,-0.160799,-0.275829,-2.56396,-0.0858327,2.43663,-1.5056])
        self.q2 = np.array([-0.476673,-0.31845,0.271152,-1.42044,0.136209,1.16319,-0.884341])
        self.q3 = np.array([-0.14389,0.31215,0.559452,-1.8845,-0.178117,2.22104,-0.26461])

    def infinity_symbol(self, amp_array, k):
        s = np.sin(k)
        c = np.cos(k)
        return np.array([amp_array[0] * s,
                         amp_array[1] * s * c,
                         amp_array[2] * s])

    def infinity_symbol_derivative(self, amp_array, k, omega):
        s = np.sin(k)
        c = np.cos(k)
        return np.array([amp_array[0] * omega * c,
                         amp_array[1] * omega * (c * c - s * s),
                         amp_array[2] * omega * c])

    def generate_half_infinity_cartesian(self, amp_array, theta_z, center_pos, freq, Ts, phase_offset = 0.0):
        omega = 2 * np.pi * freq
        rot_z = rotation_z(theta_z)
        pos_traj = []
        vel_traj = []
        time_traj = []

        t = 0
        k = omega * t - phase_offset

        end_phase = phase_offset + np.pi
        while k <= end_phase:
            k = omega * t - phase_offset
            raw_pos = self.infinity_symbol(amp_array, k)
            raw_vel = self.infinity_symbol_derivative(amp_array, k, omega)
            pos = rot_z @ raw_pos + center_pos
            vel = rot_z @ raw_vel
            pos_traj.append(pos)
            vel_traj.append(vel)
            time_traj.append(t)
            t += Ts

        len_single_traj = len(pos_traj)
        len_total_traj = len(pos_traj)
        num_traj = math.ceil(len_total_traj / len_single_traj)
        return np.array(time_traj), np.array(pos_traj), np.array(vel_traj)
    
    def generate_half_infinity_joint(self, model_pin, data_pin, params, q0 = None, return_ee_traj = False, Nrepeat = 1):
        amp_array = params['amp']
        theta = params['theta']
        center_pos = params['center_pos']
        freq = params['freq']
        Ts = params['Ts']
        phase_offset = params['phase_offset']

        # Generate trajectory in Cartesian space
        time_traj, pos_traj, vel_traj = self.generate_half_infinity_cartesian(amp_array, theta, center_pos, freq, Ts, phase_offset)
        
        if Nrepeat > 1:
            pos_traj = np.vstack([pos_traj[:-1]] * Nrepeat)
            vel_traj = np.vstack([vel_traj[:-1]] * Nrepeat)

        if q0 is None:
            q0 = pin.neutral(model_pin)
        # IK for joint trajectories
        q_traj, qd_traj = traj_inverse_kinematics(model_pin, data_pin, q0, pos_traj, vel_traj)

        if return_ee_traj:
            return time_traj, q_traj, qd_traj, pos_traj, vel_traj
        else:
            return time_traj, q_traj, qd_traj
        
    def generate_pick_and_place_traj(self, model_pin, data_pin, params, q0 = None, return_ee_traj = False):
        amp_array = params['amp']
        theta = params['theta']
        center_pos = params['center_pos']
        freq = params['freq']
        Ts = params['Ts']
        phase_offset = params['phase_offset']
        

        # Generate trajectory in Cartesian space
        time_traj, pos_traj, vel_traj = self.generate_full_infinity_cartesian(amp_array, theta, center_pos, freq, Ts)
        
        # if Nrepeat > 1:
        #     pos_traj = np.vstack([pos_traj[:-1]] * Nrepeat)
        #     vel_traj = np.vstack([vel_traj[:-1]] * Nrepeat)

        if q0 is None:
            q0 = pin.neutral(model_pin)
        # IK for joint trajectories
        q_traj, qd_traj = traj_inverse_kinematics(model_pin, data_pin, q0, pos_traj, vel_traj)

        first_part_t0 = (np.pi + phase_offset) / (2 * np.pi * freq)
        first_part_to_index = np.ceil(first_part_t0 * freq)
        q_traj_p0 = q_traj[-first_part_to_index:]
        qd_traj_p0 = qd_traj[-first_part_to_index:]
        pos_traj_p0 = pos_traj[-first_part_to_index:]
        vel_traj_p0 = vel_traj[-first_part_to_index:]

        full_q_traj = q_traj_p0
        full_qd_traj = qd_traj_p0
        full_pos_traj = pos_traj_p0
        full_vel_traj = vel_traj_p0
        time_traj = np.arange(len(full_q_traj)) * Ts

        if return_ee_traj:
            return time_traj, full_q_traj, full_qd_traj, full_pos_traj, full_vel_traj
        else:
            return time_traj,full_q_traj, full_qd_traj
    
    def generate_full_infinity_cartesian(self, amp_array, theta_z, center_pos, freq, Ts):
        omega = 2 * np.pi * freq
        rot_z = rotation_z(theta_z)
        pos_traj = []
        vel_traj = []
        time_traj = []

        t = 0
        k = omega * t

        end_phase = 2 * np.pi
        while k <= end_phase:
            k = omega * t
            raw_pos = self.infinity_symbol(amp_array, k)
            raw_vel = self.infinity_symbol_derivative(amp_array, k, omega)
            pos = rot_z @ raw_pos + center_pos
            vel = rot_z @ raw_vel
            pos_traj.append(pos)
            vel_traj.append(vel)
            time_traj.append(t)
            t += Ts

        len_single_traj = len(pos_traj)
        len_total_traj = len(pos_traj)
        num_traj = math.ceil(len_total_traj / len_single_traj)
        return np.array(time_traj), np.array(pos_traj), np.array(vel_traj)

    def generate_full_infinity_joint(self, model_pin, data_pin, params, q0 = None, return_ee_traj = False, Nrepeat = 1):
        amp_array = params['amp']
        theta = params['theta']
        center_pos = params['center_pos']
        freq = params['freq']
        Ts = params['Ts']

        # Generate trajectory in Cartesian space
        time_traj, pos_traj, vel_traj = self.generate_full_infinity_cartesian(amp_array, theta, center_pos, freq, Ts)
        
        if Nrepeat > 1:
            pos_traj = np.vstack([pos_traj[:-1]] * Nrepeat)
            vel_traj = np.vstack([vel_traj[:-1]] * Nrepeat)

        if q0 is None:
            q0 = pin.neutral(model_pin)
        # IK for joint trajectories
        q_traj, qd_traj = traj_inverse_kinematics(model_pin, data_pin, q0, pos_traj, vel_traj)

        if return_ee_traj:
            return time_traj, q_traj, qd_traj, pos_traj, vel_traj
        else:
            return time_traj, q_traj, qd_traj

    def generate_chirp_trajectory(self, A, B, duration, Ts, init_pos = None, init_vel = None):
        
        # A and B have shape [Aq0_0 Aq0_1, ... Aq0_order ; Aq1_0 ... Aq1_order; ....]
        nq, order = A.shape

        if init_pos is None:
            init_pos = np.zeros(nq)
        if init_vel is None:
            init_vel = np.zeros(nq)

        omega_f0 = 2 * np.pi / duration
        num_samples = int(duration / Ts)

        t = np.linspace(0, duration, num_samples+1)
        q = np.zeros((t.shape[0], nq)) + init_pos
        qd = np.zeros_like(q) + init_vel
        qdd = np.zeros_like(q)


        for k in range(1, order + 1):
            q += np.outer(np.sin(omega_f0 * k * t), A[k - 1] / (omega_f0 * k)) - np.outer(
                np.cos(omega_f0 * k * t), B[k - 1] / (omega_f0 * k)
            )
            qd += np.outer(np.cos(omega_f0 * k * t), A[k - 1]) + np.outer(
                np.sin(omega_f0 * k * t), B[k - 1]
            )
            qdd += -np.outer(np.sin(omega_f0 * k * t), A[k - 1] * (omega_f0 * k)) + np.outer(
                np.cos(omega_f0 * k * t), B[k - 1] * (omega_f0 * k)
            )
        return t, q, qd, qdd

    def exc_ref_init_config(self, duration):
        exc_fourier_config = {"order": 5,
                              "duration": duration}
        
        nq = self.nq
        upper_joint_pos_limits = np.array([2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
        lower_joint_pos_limits = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        joints_range = upper_joint_pos_limits - lower_joint_pos_limits
        joints_range[0] = 2.0
        joint_vel_limits = 2.0 * np.ones(7)

        exc_robot_config = {"njoints": nq,
                            "upper_joint_pos_limits": upper_joint_pos_limits,
                            "lower_joint_pos_limits": lower_joint_pos_limits,
                            "joint_vel_limits": joint_vel_limits,
                            "joints_range": joints_range,
                    }
        return exc_fourier_config, exc_robot_config


    def exc_ref_get_params(self, fourier_config, robot_config):
        _, _, _, _, params = obtain_valid_traj_param_simplified(fourier_config,
                                                                robot_config)
        return params
    
    def exc_ref_generate_full_traj(self, fourier_config, robot_config, params, Ts):
        fps = 1.0 / Ts
        t, q, qd, qdd = generate_fourier_traj(
                fourier_config['order'], 
                fourier_config['duration'],
                self.nq, 
                params, 
                robot_config['init_pos'], 
                robot_config['init_vel'],
                fps = fps,
            )
        return t, q, qd, qdd

    def generate_full_exc_traj(self, fourier_config, robot_config):
        t, q, qd, qdd, init_params = obtain_valid_traj_param_simplified(
            fourier_config, robot_config
        )

    def fit_6th_order_polynomial(self, t0, tm, tf, p0, pm, pf):
        """
        Fit a 6th-order polynomial trajectory that satisfies the given boundary conditions.
        Boundary conditions:
        - p(t0) = p0 (initial position)
        - p(tm) = pm (medium position)
        - p(tf) = pf (final position)
        - p'(t0) = 0 (initial velocity)
        - p'(tf) = 0 (final velocity)
        - p''(t0) = 0 (initial acceleration)
        - p''(tf) = 0 (final acceleration)
        - p'''(t0) = 0 (initial jerk)
        - p'''(tf) = 0 (final jerk)
        """
        # Set up the system of equations for a 6th order polynomial
        A = np.array([
            [t0**6, t0**5, t0**4, t0**3, t0**2, t0, 1],
            [tm**6, tm**5, tm**4, tm**3, tm**2, tm, 1],
            [tf**6, tf**5, tf**4, tf**3, tf**2, tf, 1],
            [6*t0**5, 5*t0**4, 4*t0**3, 3*t0**2, 2*t0, 1, 0],
            [6*tf**5, 5*tf**4, 4*tf**3, 3*tf**2, 2*tf, 1, 0],
            [30*t0**4, 20*t0**3, 12*t0**2, 6*t0, 2, 0, 0],
            [30*tf**4, 20*tf**3, 12*tf**2, 6*tf, 2, 0, 0],
        ])
        
        # Define the boundary conditions (position at t0, tm, tf, and zero velocity, acceleration, and jerk)
        B = np.array([p0, pm, pf, 0, 0, 0, 0])
        
        # Solve the system to get the polynomial coefficients
        coeffs = np.linalg.solve(A, B)
        
        return coeffs

    def compute_3d_trajectory(self, t0, tm, tf, p0, pm, pf, Ts):
        """
        Compute 3D position and velocity trajectories using 5th order polynomial.
        """
        t_values = np.arange(t0, tf, Ts)
        coeffs_x = self.fit_6th_order_polynomial(t0, tm, tf, p0[0], pm[0], pf[0])
        coeffs_y = self.fit_6th_order_polynomial(t0, tm, tf, p0[1], pm[1], pf[1])
        coeffs_z = self.fit_6th_order_polynomial(t0, tm, tf, p0[2], pm[2], pf[2])
        
        position = np.array([
            np.polyval(coeffs_x, t_values),
            np.polyval(coeffs_y, t_values),
            np.polyval(coeffs_z, t_values)
        ]).T
        
        velocity = np.array([
            np.polyval(np.polyder(coeffs_x), t_values),
            np.polyval(np.polyder(coeffs_y), t_values),
            np.polyval(np.polyder(coeffs_z), t_values)
        ]).T
        
        return t_values, position, velocity
    
    def compute_3d_trajectory_joint(self, model_pin, data_pin, params, q0 = None, return_ee_traj = False):
        t0 = 0.0
        tf = params['TotalTime']
        Ts = params['Ts']
        p0 = params['p0']
        pm = params['pm']
        pf = params['pf']
        
        time_traj, pos_traj, vel_traj = self.compute_3d_trajectory(t0, 0.5*tf, tf, p0, pm, pf, Ts)
        if q0 is None:
            q0 = pin.neutral(model_pin)
        q_traj, qd_traj = traj_inverse_kinematics(model_pin, data_pin, q0, pos_traj, vel_traj)

        if return_ee_traj:
            return time_traj, q_traj, qd_traj, pos_traj, vel_traj
        else:
            return time_traj, q_traj, qd_traj
        

    def compute_pick_and_place_traj(self, model_pin, data_pin, full_param, q0 = None, return_time_events = False):
    
        # inf_amp = full_param['amp']
        # freq = full_param['freq']
        # theta = full_param['theta']
        # ee_pos_init = full_param['center_pos']
        # phase_offset = full_param['phase_offset']
        # grasp_time = full_param['grasp_time']
        grasp_length = int(np.ceil(full_param['grasp_time'] / full_param['Ts']))
        # q0 = None

        # _, _, _, p0p1_pos_traj, p0p1_vel_traj = ref_gen.compute_3d_trajectory_joint(p01_params, return_ee_traj=True)
        # traj_p0p1 = {
        #     'ee_pos': p0p1_pos_traj.copy(),
        #     'ee_vel': p0p1_vel_traj.copy(),
        # }


        p1p2_param = copy.deepcopy(full_param)
        _, _, _, hinf_pos_traj_p1p2, hinf_vel_traj_p1p2 = self.generate_half_infinity_joint(model_pin, data_pin, 
                                                                                            p1p2_param, return_ee_traj=True,
                                                                                            q0 = q0)
        p1 = hinf_pos_traj_p1p2[0,:].copy()
        p2 = hinf_pos_traj_p1p2[-1,:].copy()
    
        ##
        # Traj to p1
        # Spline p0 to p1
        # Move to p1: spline to have enough opening
        p0 = full_param['init_pos']
        p0_mid = (p0 + p1) * 0.5
        p0_mid[2] = 0.75*p0[2] + 0.25*p1[2]
        p01_params = {
            'TotalTime': full_param['spline_time'],
            'Ts': full_param['Ts'],
            'p0': np.copy(p0),
            'pm': np.copy(p0_mid),
            'pf': np.copy(p1),
        }
        _, _, _, p0p1_pos_traj, p0p1_vel_traj = self.compute_3d_trajectory_joint(model_pin, data_pin, p01_params, q0, return_ee_traj=True)
        traj_p0p1 = {
            'ee_pos': p0p1_pos_traj.copy(),
            'ee_vel': p0p1_vel_traj.copy(),
        }


        # Holding position to grasp p1
        traj_p1 = {
            'ee_pos': np.tile(p1, (grasp_length, 1)),
            'ee_vel': np.zeros((grasp_length, 3)),
        }
        traj_p2 = {
            'ee_pos': np.tile(p2, (grasp_length, 1)),
            'ee_vel': np.zeros((grasp_length, 3)),
        }
        # Move from p1 to p2
        traj_p1p2 = {
            'ee_pos': hinf_pos_traj_p1p2.copy(),
            'ee_vel': hinf_vel_traj_p1p2.copy(),
        }

        # Spline p2 to p3
        # Move to p3: spline to have enough opening
        p3_offset = np.array([full_param['spline_x_diff'], full_param['spline_y_diff'], 0.0])
        # p3 = np.array([p2[0], p1[1], p2[2]]) + p3_offset
        p3 = np.array([p1[0], p2[1], p2[2]]) + p3_offset
        p3_mid = (p3 + p2) * 0.5
        p3_mid[2] = p3_mid[2] + full_param['spline_height']
        p23_params = {
            'TotalTime': full_param['spline_time'],
            'Ts': full_param['Ts'],
            'p0': np.copy(p2),
            'pm': np.copy(p3_mid),
            'pf': np.copy(p3),
        }

        _, _, _, p2p3_pos_traj, p2p3_vel_traj = self.compute_3d_trajectory_joint(model_pin, data_pin, p23_params, q0, return_ee_traj=True)
        traj_p2p3 = {
            'ee_pos': p2p3_pos_traj.copy(),
            'ee_vel': p2p3_vel_traj.copy(),
        }

        # half inf p3 to p4
        p3p4_param = copy.deepcopy(full_param)
        p3p4_param['center_pos'] += p3_offset
        _, _, _, hinf_pos_traj_p3p4, hinf_vel_traj_p3p4 = self.generate_half_infinity_joint(model_pin, data_pin, 
                                                                                            p3p4_param, return_ee_traj=True,
                                                                                            q0 = q0)
        # p3 = hinf_pos_traj_p3p4[0,:].copy()
        p4 = hinf_pos_traj_p3p4[-1,:].copy()

        traj_p3p4 = {
            'ee_pos': hinf_pos_traj_p3p4.copy(),
            'ee_vel': hinf_vel_traj_p3p4.copy(),
        }
        traj_p3 = {
            'ee_pos': np.tile(p3, (grasp_length, 1)),
            'ee_vel': np.zeros((grasp_length, 3)),
        }
        traj_p4 = {
            'ee_pos': np.tile(p4, (grasp_length, 1)),
            'ee_vel': np.zeros((grasp_length, 3)),
        }

        # Compute Return
        pf = np.copy(full_param['init_pos'])
        pf_mid = (p0 + p4) * 0.5
        pf_mid[2] = 0.75*pf[2] + 0.25*p4[2]
        p4f_params = {
            'TotalTime': full_param['spline_time'],
            'Ts': full_param['Ts'],
            'p0': np.copy(p4),
            'pm': np.copy(pf_mid),
            'pf': np.copy(pf),
        }
        _, _, _, p4pf_pos_traj, p4pf_vel_traj = self.compute_3d_trajectory_joint(model_pin, data_pin, p4f_params, q0, return_ee_traj=True)
        traj_p4pf = {
            'ee_pos': p4pf_pos_traj.copy(),
            'ee_vel': p4pf_vel_traj.copy(),
        }
        traj_pf = {
            'ee_pos': np.tile(pf, (grasp_length, 1)),
            'ee_vel': np.zeros((grasp_length, 3)),
        }


        traj_list = [traj_p0p1,
                     traj_p1, # Close
                     traj_p1p2, 
                     traj_p2, # Open
                     traj_p2p3,
                     traj_p3, # close
                     traj_p3p4,
                     traj_p4,  #open
                     traj_p4pf,
                     traj_pf,
                     ]
        concatenated_traj = {
            'ee_pos': np.concatenate([traj['ee_pos'] for traj in traj_list], axis=0),
            'ee_vel': np.concatenate([traj['ee_vel'] for traj in traj_list], axis=0)
        }
        full_time_traj = np.arange(concatenated_traj['ee_pos'].shape[0]) * full_param['Ts']

        # q0 = pin.neutral(model_pin)
        if q0 is None:
            q0 = pin.neutral(model_pin)
        full_q_traj, full_qd_traj = traj_inverse_kinematics(model_pin, data_pin, q0, concatenated_traj['ee_pos'], concatenated_traj['ee_vel'])
        ee_pos_check = traj_fwd_kinematics(model_pin, data_pin, full_q_traj.copy())

        if return_time_events:
            # Close to grasp Load 1 at P1
            list_p1_close = [traj_p0p1]
            traj_p1_close = np.concatenate([traj['ee_pos'] for traj in list_p1_close], axis=0)
            time_p1_close = traj_p1_close.shape[0] * full_param['Ts']

            # Open to release Load 1 at P2
            list_p2_open = [traj_p0p1,
                            traj_p1,
                            traj_p1p2]
            traj_p2_close = np.concatenate([traj['ee_pos'] for traj in list_p2_open], axis=0)
            time_p2_open = traj_p2_close.shape[0] * full_param['Ts']

            # Close to grasp Load 2 at P3
            list_p3_close = [traj_p0p1,
                            traj_p1,
                            traj_p1p2,
                            traj_p2,
                            traj_p2p3]
            traj_p3_close = np.concatenate([traj['ee_pos'] for traj in list_p3_close], axis=0)
            time_p3_close = traj_p3_close.shape[0] * full_param['Ts']

            # Open to release Load 2 at P4
            list_p4_open = [traj_p0p1,
                            traj_p1,
                            traj_p1p2,
                            traj_p2,
                            traj_p2p3,
                            traj_p3,
                            traj_p3p4]
            traj_p4_open = np.concatenate([traj['ee_pos'] for traj in list_p4_open], axis=0)
            time_p4_open = traj_p4_open.shape[0] * full_param['Ts']

            time_events = {
                'event': np.array([1, 0, 1, 0]),
                'times': np.array([time_p1_close, time_p2_open, time_p3_close, time_p4_open])
            }

            return full_time_traj, full_q_traj, full_qd_traj, concatenated_traj['ee_pos'], concatenated_traj['ee_vel'], time_events
        

        else:
            return full_time_traj, full_q_traj, full_qd_traj, concatenated_traj['ee_pos'], concatenated_traj['ee_vel'] 

if __name__ == "__main__":
    ref_gen = ReferenceGenerator()

    time_traj = []
    pos_traj = []
    vel_traj = []
    acc_traj = []

    # test_name = 'HALF_INF'
    # test_name = 'FULL_INF'
    # test_name = 'FULL_INF_PARAM'
    # test_name = 'HALF_INF_PARAM'
    # test_name = 'CHIRP'
    # test_name = 'PICK_AND_PLACE'
    test_name = 'PICK_AND_PLACE_PARAM'

    robot_name = 'panda'
    xml_path = (
            Path(__file__).resolve().parent.parent
            / "robots"
            / str(robot_name)
            / "panda.xml"
        ).as_posix()
    
    model_pin = pin.buildModelFromMJCF(xml_path)
    data_pin = model_pin.createData()
    q_init = np.array([0, -0.4, 0, -2.4, 0, 2.2, -0.7853])
    ee_id = model_pin.getFrameId("end_effector")
    ee_pos_init = pin_get_link_pos(model_pin, data_pin, q_init, ee_id)
    print(f'ee_pos_init: {ee_pos_init} | q_init {q_init}')

    q_neutral = pin.neutral(model_pin)
    ee_pos_neutral = pin_get_link_pos(model_pin, data_pin, q_neutral, ee_id)
    print(f'ee_pos_neutral: {ee_pos_neutral} | q_neutral {q_neutral}')


    if test_name == 'HALF_INF':
        amp_array = np.array([0, 0.2, 1.5])
        theta = -30/180*np.pi
        phase_offset = 50/180*np.pi
        # center_pos = np.zeros(3)
        center_pos = ee_pos_init
        print(center_pos)
        freq = 1
        Ts = 0.0001
        total_time = 5
        time_traj, pos_traj, vel_traj = ref_gen.generate_half_infinity_cartesian(amp_array, theta, center_pos, freq, Ts, phase_offset)

        vel_traj_finite_diff = []
        # Now calculate finite difference velocity
        for i in range(1, len(pos_traj)):
            # Compute finite difference: (position[i+1] - position[i]) / Ts
            vel_finite_diff = (pos_traj[i] - pos_traj[i-1]) / Ts
            vel_traj_finite_diff.append(vel_finite_diff)

        # We can use the last position's velocity from finite difference as a placeholder
        # or a zero velocity at the end if you want.
        vel_traj_finite_diff.append(vel_traj_finite_diff[-1])  # Optional: keep the last velocity constant
        vel_traj_finite_diff = np.array(vel_traj_finite_diff)
        # Extract the rotated x, y, z coordinates
        x, y, z = pos_traj[:, 0], pos_traj[:, 1], pos_traj[:, 2]

        print(f'traj len {pos_traj.shape}')
        print(f'r0 {pos_traj[0,:]}')
        print(f'r1 {pos_traj[-1,:]}')

        # Plot the trajectory
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, label='Figure-Eight Path')
        ax.plot(ee_pos_init[0], ee_pos_init[1], ee_pos_init[2], '*')

        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('End-Effector Trajectory - Phases Half Infinity')
        ax.legend()

        print(vel_traj_finite_diff.shape)

        ncor = 3
        fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
        for cor in range(ncor):
            axes[cor].plot(time_traj, pos_traj[:,cor], 'b', label='analy')
            # axes[cor].plot(time_traj, ee_pos_check[:,cor], '--r', label='Check from ik')
            axes[cor].grid()
        axes[0].legend()
        axes[0].set_title('End-effector Position')

        fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
        for cor in range(ncor):
            axes[cor].plot(time_traj, vel_traj[:,cor], 'b', label='analy')
            axes[cor].grid()
        axes[0].set_title('End-effector Velocity')

        plt.show()

    elif test_name == 'FULL_INF':
        amp_array = np.array([0, 0.3, 0.2])
        theta = -30/180*np.pi
        phase_offset = 0.0

        # center_pos = np.zeros(3)
        print(ee_pos_init)
        center_pos = ee_pos_init
        print(center_pos)
        
        freq = 0.1
        Ts = 0.01
        total_time = 5
        time_traj, pos_traj, vel_traj = ref_gen.generate_full_infinity_cartesian(amp_array, theta, center_pos, freq, Ts)

        vel_traj_finite_diff = []
        # Now calculate finite difference velocity
        for i in range(1, len(pos_traj)):
            # Compute finite difference: (position[i+1] - position[i]) / Ts
            vel_finite_diff = (pos_traj[i] - pos_traj[i-1]) / Ts
            vel_traj_finite_diff.append(vel_finite_diff)

        traj_time = np.arange(0, total_time, Ts)
        # We can use the last position's velocity from finite difference as a placeholder
        # or a zero velocity at the end if you want.
        vel_traj_finite_diff.append(vel_traj_finite_diff[-1])  # Optional: keep the last velocity constant
        vel_traj_finite_diff = np.array(vel_traj_finite_diff)
        # Extract the rotated x, y, z coordinates
        x, y, z = pos_traj[:, 0], pos_traj[:, 1], pos_traj[:, 2]



        ### Joint Trajectory
        q0 = q_init
        q_traj, qd_traj = traj_inverse_kinematics(model_pin, data_pin, q0, pos_traj, vel_traj)

        ee_pos_check = traj_fwd_kinematics(model_pin, data_pin, q_traj.copy())

        print(f'traj len {pos_traj.shape}')
        print(f'r0 {pos_traj[0,:]}')
        print(f'r1 {pos_traj[-1,:]}')

        # Plot the trajectory
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, label='Figure-Eight Path')
        ax.plot(ee_pos_init[0], ee_pos_init[1], ee_pos_init[2], '*')
        ax.plot(pos_traj[0,0], pos_traj[0,1], pos_traj[0,2], '*', label='r0')
        ax.plot(pos_traj[-1,0], pos_traj[-1,1], pos_traj[-1,2], '*', label='r1')

        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('End-Effector Trajectory - Full Infinity')
        ax.legend()

        print(vel_traj_finite_diff.shape)

        ncor = 3
        fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
        for cor in range(ncor):
            axes[cor].plot(time_traj, pos_traj[:,cor], 'b', label='analy')
            axes[cor].plot(time_traj, ee_pos_check[:,cor], '--r', label='Check from ik')
            axes[cor].grid()
        axes[0].legend()
        axes[0].set_title('End-effector Position')

        fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
        for cor in range(ncor):
            axes[cor].plot(time_traj, vel_traj[:,cor], 'b', label='analy')
            axes[cor].plot(time_traj, vel_traj_finite_diff[:,cor], '--r', label='diff')
            axes[cor].grid()
        axes[0].set_title('End-effector Velocity')

        ncor = 7
        fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
        for cor in range(ncor):
            axes[cor].plot(time_traj, q_traj[:,cor], 'b')
            axes[cor].grid()
            if cor == 0:
                axes[cor].set_title('Joint Position')

        fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
        for cor in range(ncor):
            axes[cor].plot(time_traj, qd_traj[:,cor], 'b')
            axes[cor].grid()
            if cor == 0:
                axes[cor].set_title('Joint Velocity')

        plt.show()

    elif test_name == 'FULL_INF_PARAM':
        TRAJ_PARAMS = {}
        TRAJ_PARAMS['amp'] = np.array([0, 0.40, 0.15])
        TRAJ_PARAMS['freq'] = 0.7
        TRAJ_PARAMS['theta'] = -0/180*np.pi
        TRAJ_PARAMS['center_pos'] = ee_pos_init
        TRAJ_PARAMS['Ts'] = 0.02

        time_traj, q_traj, qd_traj, pos_traj, vel_traj = ref_gen.generate_full_infinity_joint(model_pin, 
                                                                                              data_pin, 
                                                                                              TRAJ_PARAMS, 
                                                                                              return_ee_traj=True,
                                                                                              q0=q_init)

        ee_pos_check = traj_fwd_kinematics(model_pin, data_pin, q_traj.copy())


        # Plot the trajectory
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(pos_traj[:,0], pos_traj[:,1], pos_traj[:,2], label='Figure-Eight Path')
        ax.plot(ee_pos_init[0], ee_pos_init[1], ee_pos_init[2], '*')
        ax.plot(pos_traj[0,0], pos_traj[0,1], pos_traj[0,2], '*', label='r0')
        ax.plot(pos_traj[-1,0], pos_traj[-1,1], pos_traj[-1,2], '*', label='r1')

        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('End-Effector Trajectory - Full Infinity')
        ax.legend()

        ncor = 3
        fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
        for cor in range(ncor):
            axes[cor].plot(time_traj, pos_traj[:,cor], 'b', label='analy')
            axes[cor].plot(time_traj, ee_pos_check[:,cor], '--r', label='Check from ik')
            axes[cor].grid()
        axes[0].legend()
        axes[0].set_title('End-effector Position')

        fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
        for cor in range(ncor):
            axes[cor].plot(time_traj, vel_traj[:,cor], 'b', label='analy')
            axes[cor].grid()
        axes[0].set_title('End-effector Velocity')

        ncor = 7
        fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
        for cor in range(ncor):
            axes[cor].plot(time_traj, q_traj[:,cor], 'b')
            axes[cor].grid()
            if cor == 0:
                axes[cor].set_title('Joint Position')

        fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
        for cor in range(ncor):
            axes[cor].plot(time_traj, qd_traj[:,cor], 'b')
            axes[cor].grid()
            if cor == 0:
                axes[cor].set_title('Joint Velocity')

        plt.show()

    elif test_name == 'HALF_INF_PARAM':
        TRAJ_PARAMS = {}
        TRAJ_PARAMS['amp'] = np.array([0, 0.3, 0.2])
        TRAJ_PARAMS['freq'] = 0.5
        TRAJ_PARAMS['theta'] = -30/180*np.pi
        TRAJ_PARAMS['center_pos'] = ee_pos_init
        TRAJ_PARAMS['Ts'] = 0.01
        TRAJ_PARAMS['phase_offset'] = 50/180*np.pi

        q0 = np.array([-0.05, 2.5, 0.2, 1.6, 0.02, -3.1, 0.0])
        time_traj, q_traj, qd_traj, pos_traj, vel_traj = ref_gen.generate_half_infinity_joint(model_pin, data_pin, TRAJ_PARAMS, return_ee_traj=True,
                                                                                              q0 = q0)

        ee_pos_check = traj_fwd_kinematics(model_pin, data_pin, q_traj.copy())


        # Plot the trajectory
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(pos_traj[:,0], pos_traj[:,1], pos_traj[:,2], label='Figure-Eight Path')
        ax.plot(ee_pos_init[0], ee_pos_init[1], ee_pos_init[2], '*')
        ax.plot(pos_traj[0,0], pos_traj[0,1], pos_traj[0,2], '*', label='r0')
        ax.plot(pos_traj[-1,0], pos_traj[-1,1], pos_traj[-1,2], '*', label='r1')

        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('End-Effector Trajectory - HALF_INF_PARAM')
        ax.legend()

        ncor = 3
        fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
        for cor in range(ncor):
            axes[cor].plot(time_traj, pos_traj[:,cor], 'b', label='analy')
            axes[cor].plot(time_traj, ee_pos_check[:,cor], '--r', label='Check from ik')
            axes[cor].grid()
        axes[0].legend()
        axes[0].set_title('End-effector Position')

        fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
        for cor in range(ncor):
            axes[cor].plot(time_traj, vel_traj[:,cor], 'b', label='analy')
            axes[cor].grid()
        axes[0].set_title('End-effector Velocity')

        ncor = 7
        fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
        for cor in range(ncor):
            axes[cor].plot(time_traj, q_traj[:,cor], 'b')
            axes[cor].grid()
            if cor == 0:
                axes[cor].set_title('Joint Position')

        fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
        for cor in range(ncor):
            axes[cor].plot(time_traj, qd_traj[:,cor], 'b')
            axes[cor].grid()
            if cor == 0:
                axes[cor].set_title('Joint Velocity')

        plt.show()

    elif test_name == 'PICK_AND_PLACE':
        TRAJ_PARAMS = {}
        # TRAJ_PARAMS['amp'] = np.array([0, 0.4, 0.2])
        # TRAJ_PARAMS['freq'] = 0.5

        TRAJ_PARAMS['amp'] = np.array([0, 0.35, 0.4])
        TRAJ_PARAMS['freq'] = 0.2
        TRAJ_PARAMS['theta'] = -0*30/180*np.pi
        TRAJ_PARAMS['center_pos'] = ee_pos_init - np.array([0.15, 0.0, 0.0])
        TRAJ_PARAMS['Ts'] = 0.01
        TRAJ_PARAMS['phase_offset'] = 45/180*np.pi
        TRAJ_PARAMS['grasp_time'] = 2


        # q0 = np.array([-0.05, 2.5, 0.2, 1.6, 0.02, -3.1, 0.0])
        q0 = None
        time_traj, hinf_q_traj, hinf_qd_traj, hinf_pos_traj, hinf_vel_traj = ref_gen.generate_half_infinity_joint(model_pin, data_pin, TRAJ_PARAMS, return_ee_traj=True,
                                                                                              q0 = q_init)

        ee_pos_check = traj_fwd_kinematics(model_pin, data_pin, hinf_q_traj.copy())

        p1_data = {
            'q' : hinf_q_traj[0,:].copy(),
            'qd': hinf_qd_traj[0,:].copy(),
            'ee_pos' : hinf_pos_traj[0,:].copy(),
            'ee_vel' : hinf_vel_traj[0,:].copy(),
        }
        
        p2_data = {
            'q' : hinf_q_traj[-1,:].copy(),
            'qd': hinf_qd_traj[-1,:].copy(),
            'ee_pos' : hinf_pos_traj[-1,:].copy(),
            'ee_vel' : hinf_vel_traj[-1,:].copy(),
        }

        grasp_length = int(np.ceil(TRAJ_PARAMS['grasp_time'] / TRAJ_PARAMS['Ts']))

        # Holding position to grasp p1
        traj_1 = {
            'q' : np.tile(p1_data['q'], (grasp_length, 1)),
            'qd': np.zeros((grasp_length, 7)),
            'ee_pos': np.tile(p1_data['ee_pos'], (grasp_length, 1)),
            'ee_vel': np.zeros((grasp_length, 3)),
        }
        # Move from p1 to p2
        traj_2 = {
            'q' : hinf_q_traj.copy(),
            'qd': hinf_qd_traj.copy(),
            'ee_pos' : hinf_pos_traj.copy(),
            'ee_vel' : hinf_vel_traj.copy(),
        }
        # Position to graps p2
        traj_3 = {
            'q' : np.tile(p2_data['q'], (grasp_length, 1)),
            'qd': np.zeros((grasp_length, 7)),
            'ee_pos' : np.tile(p2_data['ee_pos'], (grasp_length, 1)),
            'ee_vel' : np.zeros((grasp_length, 3)),
        }
        # Move to p3: spline to have enough opening
        diff_p3_x = 0.3
        p1_ee_pos = p1_data['ee_pos']
        p2_ee_pos = p2_data['ee_pos']
        p3_ee_pos = np.array([p2_ee_pos[0] + diff_p3_x, p1_ee_pos[1], p2_ee_pos[2]])
        p3_mid = (p3_ee_pos + p2_ee_pos) * 0.5
        p3_mid[2] = p3_mid[2] + 0.2
        p23_params = {
            'TotalTime': 1.0,
            'Ts': TRAJ_PARAMS['Ts'],
            'p0': np.copy(p2_ee_pos),
            'pm': np.copy(p3_mid),
            'pf': np.copy(p3_ee_pos),
        }

        p23_time, p23_q_traj, p23_qd_traj, p23_pos_traj, p23_vel_traj = ref_gen.compute_3d_trajectory_joint(p23_params,
                                                                                                            return_ee_traj=True)
        # Move p2 to p3
        traj_4 = {
            'q' : p23_q_traj.copy(),
            'qd': p23_qd_traj.copy(),
            'ee_pos' : p23_pos_traj.copy(),
            'ee_vel' : p23_vel_traj.copy(),
        }

        TRAJ2_PARAMS = {}
        TRAJ2_PARAMS['amp'] = TRAJ_PARAMS['amp']
        TRAJ2_PARAMS['freq'] = TRAJ_PARAMS['freq']
        TRAJ2_PARAMS['theta'] = -0*30/180*np.pi
        diff_x = p2_ee_pos[0] - p1_ee_pos[0] + diff_p3_x
        TRAJ2_PARAMS['center_pos'] = ee_pos_init + np.array([diff_x, 0, 0])
        TRAJ2_PARAMS['Ts'] = 0.01
        TRAJ2_PARAMS['phase_offset'] = 45/180*np.pi
        TRAJ2_PARAMS['grasp_time'] = 2

        # q0 = np.array([-0.05, 2.5, 0.2, 1.6, 0.02, -3.1, 0.0])
        q0 = None
        time_traj_2, hinf_q_traj_2, hinf_qd_traj_2, hinf_pos_traj_2, hinf_vel_traj_2 = ref_gen.generate_half_infinity_joint(model_pin, data_pin, TRAJ2_PARAMS, return_ee_traj=True,
                                                                                              q0 = q0)
        p4_ee_pos = hinf_pos_traj_2[-1,:]
        # Move from p3 to p4
        traj_34 = {
            'q' : hinf_q_traj_2.copy(),
            'qd': hinf_qd_traj_2.copy(),
            'ee_pos' : hinf_pos_traj_2.copy(),
            'ee_vel' : hinf_vel_traj_2.copy(),
        }


        traj_list = [traj_1, traj_2, traj_3, traj_4, traj_34]
        ####
        concatenated_traj = {
            'q': np.concatenate([traj['q'] for traj in traj_list], axis=0),
            'qd': np.concatenate([traj['qd'] for traj in traj_list], axis=0),
            'ee_pos': np.concatenate([traj['ee_pos'] for traj in traj_list], axis=0),
            'ee_vel': np.concatenate([traj['ee_vel'] for traj in traj_list], axis=0)
        }
        full_time_traj = np.arange(concatenated_traj['q'].shape[0]) * TRAJ_PARAMS['Ts']

        q0 = pin.neutral(model_pin)
        full_q_traj, full_qd_traj = traj_inverse_kinematics(model_pin, data_pin, q0, concatenated_traj['ee_pos'], concatenated_traj['ee_vel'])
        ee_pos_check = traj_fwd_kinematics(model_pin, data_pin, full_q_traj.copy())


        # # Plot the trajectory
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(concatenated_traj['ee_pos'][:,0], concatenated_traj['ee_pos'][:,1], concatenated_traj['ee_pos'][:,2], label='Figure-Eight Path')
        ax.plot(p1_ee_pos[0], p1_ee_pos[1], p1_ee_pos[2], '*', color='orange', label='p1')
        ax.plot(p2_ee_pos[0], p2_ee_pos[1], p2_ee_pos[2], '*', color='blue', label='p2')
        ax.plot(p3_ee_pos[0], p3_ee_pos[1], p3_ee_pos[2], '*', color='black', label='p3')
        ax.plot(p4_ee_pos[0], p4_ee_pos[1], p4_ee_pos[2], '*', color='green', label='p4')
        ax.plot(p3_mid[0], p3_mid[1], p3_mid[2], '*', color='red', label='p3 mid')
        ax.plot(p3_mid[0], p3_mid[1], p3_mid[2], '*', color='red', label='p3 mid')
        # ax.plot(pos_traj[0,0], pos_traj[0,1], pos_traj[0,2], '*', label='r0')
        # ax.plot(pos_traj[-1,0], pos_traj[-1,1], pos_traj[-1,2], '*', label='r1')

        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('End-Effector Trajectory - PICK_AND_PLACE')
        ax.set_xlim([0.0, 0.9])
        ax.set_ylim([-0.25, 0.25])
        ax.set_zlim([0.0, 1.0])
        ax.legend()

        ncor = 3
        fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
        for cor in range(ncor):
            axes[cor].plot(full_time_traj, concatenated_traj['ee_pos'][:,cor], 'b', label='analy')
            axes[cor].plot(full_time_traj, ee_pos_check[:,cor], '--r', label='Check from ik')
            axes[cor].grid()
        axes[0].legend()
        axes[0].set_title('End-effector Position')

        fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
        for cor in range(ncor):
            axes[cor].plot(full_time_traj, concatenated_traj['ee_vel'][:,cor], 'b', label='analy')
            axes[cor].grid()
        axes[0].set_title('End-effector Velocity')

        ncor = 7
        fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
        for cor in range(ncor):
            axes[cor].plot(full_time_traj, concatenated_traj['q'][:,cor], 'b')
            axes[cor].plot(full_time_traj, full_q_traj[:,cor], '--r')
            axes[cor].grid()
            if cor == 0:
                axes[cor].set_title('Joint Position')

        fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
        for cor in range(ncor):
            axes[cor].plot(full_time_traj, concatenated_traj['qd'][:,cor], 'b')
            axes[cor].plot(full_time_traj, full_qd_traj[:,cor], '--r')
            axes[cor].grid()
            if cor == 0:
                axes[cor].set_title('Joint Velocity')

        plt.show()

    elif test_name == 'PICK_AND_PLACE_PARAM':
        TRAJ_PARAMS = {}
        TRAJ_PARAMS['amp'] = np.array([0, 0.35, 0.35])
        TRAJ_PARAMS['freq'] = 0.2
        # TRAJ_PARAMS['amp'] = np.array([0, 0.40, 0.25])
        # TRAJ_PARAMS['freq'] = 0.25
        TRAJ_PARAMS['theta'] = 90/180*np.pi
        TRAJ_PARAMS['init_pos'] = ee_pos_init + np.array([0.0, 0, 0])
        TRAJ_PARAMS['center_pos'] = ee_pos_init + np.array([0.1, -0.10, -0.1])
        TRAJ_PARAMS['Ts'] = 0.01
        TRAJ_PARAMS['phase_offset'] = 45/180*np.pi
        TRAJ_PARAMS['grasp_time'] = 2
        TRAJ_PARAMS['spline_height'] = 0.2
        TRAJ_PARAMS['spline_time'] = 1.5
        TRAJ_PARAMS['spline_x_diff'] = 0.0
        TRAJ_PARAMS['spline_y_diff'] = 0.20

        pp_params = {}
        pp_params['amp'] = np.array([0, 0.35, 0.35])
        pp_params['freq'] = 0.2
        pp_params['theta'] = 90/180*np.pi
        pp_params['init_pos'] = ee_pos_init
        pp_params['center_pos'] = ee_pos_init + np.array([0.1, -0.15, -0*0.05])
        pp_params['Ts'] = 0.02
        pp_params['phase_offset'] = 45/180*np.pi
        pp_params['grasp_time'] = 2
        pp_params['spline_height'] = 0.3
        pp_params['spline_time'] = 1.5
        pp_params['spline_x_diff'] = 0.0
        pp_params['spline_y_diff'] = 0.30
        TRAJ_PARAMS = pp_params

        full_time_traj, pp_q_traj, pp_qd_traj, pp_ee_pos_traj, pp_ee_vel_traj, time_events = ref_gen.compute_pick_and_place_traj(model_pin, data_pin,
                                                                                                                    TRAJ_PARAMS, q_init,
                                                                                                                    return_time_events=True)

        print(f'time_events {time_events}')
        ee_pos_check = traj_fwd_kinematics(model_pin, data_pin, pp_q_traj.copy())

        # # Plot the trajectory
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(pp_ee_pos_traj[:,0], pp_ee_pos_traj[:,1], pp_ee_pos_traj[:,2], label='Figure-Eight Path')
        # ax.plot(p1_ee_pos[0], p1_ee_pos[1], p1_ee_pos[2], '*', color='orange', label='p1')
        # ax.plot(p2_ee_pos[0], p2_ee_pos[1], p2_ee_pos[2], '*', color='blue', label='p2')
        # ax.plot(p3_ee_pos[0], p3_ee_pos[1], p3_ee_pos[2], '*', color='black', label='p3')
        # ax.plot(p4_ee_pos[0], p4_ee_pos[1], p4_ee_pos[2], '*', color='green', label='p4')
        # ax.plot(p3_mid[0], p3_mid[1], p3_mid[2], '*', color='red', label='p3 mid')
        # ax.plot(p3_mid[0], p3_mid[1], p3_mid[2], '*', color='red', label='p3 mid')
        # ax.plot(pos_traj[0,0], pos_traj[0,1], pos_traj[0,2], '*', label='r0')
        # ax.plot(pos_traj[-1,0], pos_traj[-1,1], pos_traj[-1,2], '*', label='r1')

        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('End-Effector Trajectory - PICK_AND_PLACE')
        ax.set_xlim([0.0, 0.9])
        ax.set_ylim([-0.25, 0.25])
        ax.set_zlim([0.0, 1.0])
        ax.legend()

        ncor = 3
        fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
        for cor in range(ncor):
            axes[cor].plot(full_time_traj, pp_ee_pos_traj[:,cor], label='org')
            axes[cor].plot(full_time_traj, ee_pos_check[:,cor], label='Check from ik')
            axes[cor].grid()
        axes[0].legend()
        axes[0].set_title('End-effector Position')

        fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
        for cor in range(ncor):
            axes[cor].plot(full_time_traj, pp_ee_vel_traj[:,cor], label='Check from ik')
            axes[cor].grid()
        axes[0].set_title('End-effector Velocity')

        ncor = 7
        fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
        for cor in range(ncor):
            axes[cor].plot(full_time_traj, pp_q_traj[:,cor])
            axes[cor].grid()
            if cor == 0:
                axes[cor].set_title('Joint Position')

        fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
        for cor in range(ncor):
            axes[cor].plot(full_time_traj, pp_qd_traj[:,cor])
            axes[cor].grid()
            if cor == 0:
                axes[cor].set_title('Joint Velocity')

        plt.show()

    elif test_name == 'CHIRP':
        # nq = 7
        # # A = 2 * np.ones((nq,1))
        # # B = 2 * np.ones((nq,1))
        # A = np.array([[1, 2, 3, 4, 5],
        #               [6, 7, 8, 9, 10],
        #               [11, 12, 13, 14, 15],
        #               [21, 22, 23, 24, 25],
        #               [31, 32, 13, 14, 15],
        #               [41, 42, 13, 14, 15],
        #               [51, 52, 13, 14, 15]])
        # B = 2 * np.random.rand(nq,5)

        # Ts = 0.001
        # duration = 20
        # time, q, qd, qdd  = ref_gen.generate_chirp_trajectory(A, B, duration, Ts)

        # ncor = nq
        # fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
        # for cor in range(ncor):
        #     axes[cor].plot(time, q[:,cor])
        #     axes[cor].grid()
        # # Show the plot
        # axes[0].set_title('Joint Position')

        # ncor = nq
        # fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
        # for cor in range(ncor):
        #     axes[cor].plot(time, qd[:,cor])
        #     axes[cor].grid()
        # # Show the plot
        # axes[0].set_title('Joint Velocity')
        # plt.show()
        for i in range(10):

            total_time = 10
            exc_fourier_config, exc_robot_config = ref_gen.exc_ref_init_config(total_time)
            
            nq = 7
            init_pos = np.array([0, -0.4, 0, -2.4, 0, 2.2, -0.7853])
            init_vel = np.zeros(7)

            exc_robot_config.update({"init_pos": init_pos,
                                    "init_vel": init_vel})

            params = ref_gen.exc_ref_get_params(exc_fourier_config,
                                                    exc_robot_config)
            
            Ts = 0.01
            exc_traj_data = ref_gen.exc_ref_generate_full_traj(exc_fourier_config,
                                                            exc_robot_config,
                                                            params,
                                                            Ts)
            exc_time_traj, exc_q_traj, exc_qd_traj, exc_qdd_traj = exc_traj_data
            ncor = nq
            fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
            for cor in range(ncor):
                axes[cor].plot(exc_time_traj, exc_q_traj[:,cor])
                axes[cor].grid()
            # Show the plot
            axes[0].set_title('Joint Position')

            ncor = nq
            fig, axes = plt.subplots(ncor, 1, figsize=(14, 8))
            for cor in range(ncor):
                axes[cor].plot(exc_time_traj, exc_qd_traj[:,cor])
                axes[cor].grid()
            # Show the plot
            axes[0].set_title('Joint Velocity')
            plt.show()