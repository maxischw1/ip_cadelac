import numpy as np
import mujoco
import pinocchio as pin

class Logger():
    def __init__(self, state_log = True, mj_log = True, 
                 pin_log = True, mpc_log = True, kf_log = True,
                 time_log = False, delan_log = True):
        
        self.nq = 7
        self.old_xpred_next = np.zeros(2*self.nq)

        self.state_log = state_log
        self.mj_log = mj_log
        self.pin_log = pin_log
        self.mpc_log = mpc_log
        self.kf_log = kf_log
        self.time_log = time_log
        self.delan_log = delan_log

    def reset_logger(self, run_name, init_xmpc = None):
        self.old_xpred_next = init_xmpc if init_xmpc is not None else np.zeros(2*self.nq)

        data_labels = ['run_name','t']

        self.logged_data = {}
        self.init_log_from_labels(data_labels)
        self.logged_data['labels'] = run_name

        if self.state_log: self.init_state_log()

        if self.mj_log: self.init_mj_log()

        if self.pin_log: self.init_pin_log()

        if self.mpc_log: self.init_mpc_log()

        if self.kf_log: self.init_kf_log()

        if self.time_log: self.init_time_log()

        if self.delan_log: self.init_delan_log()

        self.set_np_ignored_keys([''])

    def set_np_ignored_keys(self, keys):
        self.log_keys_np_ignore = keys

    def convert_log_data_to_np(self):
        for key in self.logged_data.keys():
            if key not in self.log_keys_np_ignore:
                self.logged_data[key] = np.array(self.logged_data[key])

    def log_new_value(self, key, value):
        if key in self.logged_data.keys():
            self.logged_data[key].append(np.copy(value))

    def init_log_from_labels(self, data_labels):
        for label in data_labels:
            self.logged_data[label] = []

    def init_state_log(self):
        data_labels = ['qp','qv','qa']
        data_labels += ['tau', 'tau_ctrl']
        data_labels += ['qp_ref','qv_ref']
        data_labels += ['ee_pos','ee_vel']
        data_labels += ['ee_pos_ref','ee_vel_ref']
        data_labels += ['p','pdot']
        data_labels += ['env_id', 'enc_input']
        self.init_log_from_labels(data_labels)

    def init_mj_log(self):
        data_labels = ['m','c','g','Mq']
        data_labels = ['tau_nom','m_nom','c_nom','Mq_nom']
        data_labels += ['diff_tau_nom','diff_tau_m_nom','diff_tau_c_nom','diff_tau_g_nom']
        self.init_log_from_labels(data_labels)

    def init_pin_log(self):
        data_labels = ['tau_nom_pin','m_nom_pin','c_nom_pin','g_nom_pin','Mq_nom_pin']
        data_labels += ['diff_tau_nom_pin','diff_tau_m_nom_pin','diff_tau_c_nom_pin','diff_tau_g_nom_pin']
        self.init_log_from_labels(data_labels)

    def init_kf_log(self):
        data_labels = ['tau_kf','fext_kf']
        self.init_log_from_labels(data_labels)

    def init_delan_log(self):
        data_labels = ['tau_delan']
        self.init_log_from_labels(data_labels)

    def init_mpc_log(self):
        data_labels = ['tau_mpc','diff_tau_mpc']
        data_labels += ['xpred_next','xpred_error']
        data_labels += ['xpred_horizon','u_horizon']
        data_labels += ['mpc_cost']
        data_labels += ['time_tot', 'time_lin', 'time_sim', 'time_qp']
        data_labels += ['time_reg', 'time_sim_ad', 'time_sim_la', 'time_solution_sens_lin']

        self.init_log_from_labels(data_labels)

    def init_time_log(self):
        data_labels = ['time_solver','time_pub']
        data_labels += ['time_update','time_enc']
        self.init_log_from_labels(data_labels)

    def add_state_data(self, q, qd, qdd, tau_ctrl):

        self.log_new_value('qp', q)
        self.log_new_value('qv', qd)
        self.log_new_value('qa', qdd)
        self.log_new_value('tau_ctrl', tau_ctrl)


    def add_mpc_data(self, x_mpc, u_horizon, xpred_horizon, mpc_cost, mpc_times):

        xpred_error = x_mpc - self.old_xpred_next
        u_horizon = u_horizon.reshape(-1)
        xpred_next = xpred_horizon[:,1]
        xpred_horizon = xpred_horizon.reshape(-1)

        self.log_new_value('xpred_next', xpred_next)
        self.log_new_value('old_xpred_next', self.old_xpred_next)
        self.log_new_value('xpred_error', xpred_error)
        self.log_new_value('xpred_horizon', xpred_horizon)
        self.log_new_value('u_horizon', u_horizon)
        self.log_new_value('mpc_cost', mpc_cost)
        for key in mpc_times:
            self.log_new_value(key, mpc_times[key])
        self.old_xpred_next = np.copy(xpred_next) # Update for next iteration

    def add_sim_models_data(self, q, qd, qdd, 
                            mj_model, mj_data, sim_mj_model, sim_mj_data, model_pin, data_pin):

        # Mujoco Simulation model - Consider model already update
        Mq = np.zeros((sim_mj_model.nv, sim_mj_model.nv))
        mujoco.mj_fullM(sim_mj_model, Mq, sim_mj_data.qM)

        tau_m = Mq @ qdd
        tau_cg = np.copy(sim_mj_data.qfrc_bias.reshape((sim_mj_model.nv,)))
        tau = tau_m + tau_cg

        self.log_new_value('tau', tau)
        self.log_new_value('m', tau_m)
        self.log_new_value('c', tau_cg)
        self.log_new_value('g', np.zeros(self.nq))
        self.log_new_value('Mq', Mq.reshape(-1))

        # Mujoco Nominal model - Consider model already update
        Mq_nom = np.zeros((mj_model.nv, mj_model.nv))
        mujoco.mj_fullM(mj_model, Mq_nom, mj_data.qM)
        tau_m_nom = Mq_nom @ qdd
        tau_cg_nom = np.copy(mj_data.qfrc_bias.reshape((mj_model.nv,)))
        tau_nom = tau_m_nom + tau_cg_nom

        self.log_new_value('tau_nom', tau_nom)
        self.log_new_value('m_nom', tau_m_nom)
        self.log_new_value('c_nom', tau_cg_nom)
        self.log_new_value('g_nom', np.zeros(self.nq))
        self.log_new_value('Mq_nom', Mq_nom.reshape(-1))

        self.log_new_value('diff_tau_nom', tau - tau_nom)
        self.log_new_value('diff_tau_m_nom', tau_m - tau_m_nom)
        self.log_new_value('diff_tau_c_nom', tau_cg - tau_cg_nom)
        self.log_new_value('diff_tau_g_nom', np.zeros(self.nq))

        ##
        # Pinocchio Model
        Mq_nom_pin = pin.crba(model_pin, data_pin, q)
        tau_m_nom_pin = (Mq_nom_pin @ qdd).reshape(self.nq)
        tau_cg_nom_pin = np.array(pin.nonLinearEffects(model_pin, data_pin, q, qd)).reshape(self.nq)
        tau_nom_pin = tau_m_nom_pin + tau_cg_nom_pin

        self.log_new_value('tau_nom_pin', tau_nom_pin)
        self.log_new_value('m_nom_pin', tau_m_nom_pin)
        self.log_new_value('c_nom_pin', tau_cg_nom_pin)
        self.log_new_value('g_nom_pin', np.zeros(self.nq))
        self.log_new_value('Mq_nom_pin', Mq_nom_pin.reshape(-1))

        self.log_new_value('diff_tau_nom_pin', tau - tau_nom_pin)
        self.log_new_value('diff_tau_m_nom_pin', tau_m - tau_m_nom_pin)
        self.log_new_value('diff_tau_c_nom_pin', tau_cg - tau_cg_nom_pin)
        self.log_new_value('diff_tau_g_nom_pin', np.zeros(self.nq))

        
    def add_pin_model_data(self, q, qd, qdd, tau_read, model_pin, data_pin):
        # Pinocchio Model
        Mq_nom_pin = pin.crba(model_pin, data_pin, q)
        tau_m_nom_pin = (Mq_nom_pin @ qdd).reshape(self.nq)
        tau_cg_nom_pin = np.array(pin.nonLinearEffects(model_pin, data_pin, q, qd)).reshape(self.nq)
        tau_nom_pin = tau_m_nom_pin + tau_cg_nom_pin

        self.log_new_value('tau_nom_pin', tau_nom_pin)
        self.log_new_value('m_nom_pin', tau_m_nom_pin)
        self.log_new_value('c_nom_pin', tau_cg_nom_pin)
        self.log_new_value('g_nom_pin', np.zeros(self.nq))
        self.log_new_value('Mq_nom_pin', Mq_nom_pin.reshape(-1))

        self.log_new_value('diff_tau_nom_pin', tau_read - tau_nom_pin)
        self.log_new_value('diff_tau_m_nom_pin', np.zeros(self.nq))
        self.log_new_value('diff_tau_c_nom_pin', np.zeros(self.nq))
        self.log_new_value('diff_tau_g_nom_pin', np.zeros(self.nq))

    def add_time_data(self, time_solver, time_pub, time_update, time_enc):
        self.log_new_value('time_solver', time_solver)
        self.log_new_value('time_pub', time_pub)
        self.log_new_value('time_update', time_update)
        self.log_new_value('time_enc', time_enc)

    def log_sim_data(self, time, q, qd, qdd, tau_ctrl,
                     ee_pos, ee_vel, 
                     tau_kf, fext_kf,
                     u_horizon, xpred_horizon,
                     env_id, enc_input,
                     mj_model, mj_data, sim_mj_model, sim_mj_data, model_pin, data_pin,
                     mpc_cost, mpc_times,
                     qp_ref = np.zeros(3), qv_ref = np.zeros(3),
                     ee_pos_ref = np.zeros(3), ee_vel_ref = np.zeros(3),
                     hist_data = np.zeros(3)):
        
        x_mpc = np.concatenate((qd, q))
        self.log_new_value('t', time)

        self.add_state_data(q, qd, qdd, tau_ctrl)
        self.add_sim_models_data(q, qd, qdd, mj_model, mj_data, sim_mj_model, sim_mj_data, model_pin, data_pin)
        self.add_mpc_data(x_mpc, u_horizon, xpred_horizon, mpc_cost, mpc_times)

        self.log_new_value('env_id', env_id)
        self.log_new_value('enc_input', enc_input)

        # KF
        self.log_new_value('fext_kf', fext_kf)
        self.log_new_value('tau_kf', tau_kf)

        # Reference
        self.log_new_value('qp_ref', qp_ref)
        self.log_new_value('qv_ref', qv_ref)

        # End-effector
        self.log_new_value('ee_pos', ee_pos)
        self.log_new_value('ee_vel', ee_vel)
        self.log_new_value('ee_pos_ref', ee_pos_ref)
        self.log_new_value('ee_vel_ref', ee_vel_ref)

        # MPC
        self.log_new_value('tau_mpc', tau_ctrl)
        self.log_new_value('diff_tau_mpc', tau_ctrl)

        self.log_new_value('hist_data', hist_data)

    def log_real_data(self, time, q, qd, qdd,
                      tau_read, tau_ctrl,
                      ee_pos, ee_vel,
                      tau_kf, fext_kf,
                      u_horizon, xpred_horizon,
                      env_id, enc_input,
                      model_pin, data_pin,
                      mpc_cost, mpc_times,
                      qp_ref = np.zeros(3), qv_ref = np.zeros(3),
                      ee_pos_ref = np.zeros(3), ee_vel_ref = np.zeros(3)):
        
        x_mpc = np.concatenate((qd, q))
        self.log_new_value('t', time)

        self.add_state_data(q, qd, qdd, tau_ctrl)
        self.add_pin_model_data(q, qd, qdd, tau_read, model_pin, data_pin)
        self.add_mpc_data(x_mpc, u_horizon, xpred_horizon, mpc_cost, mpc_times)

        self.log_new_value('env_id', env_id)
        self.log_new_value('enc_input', enc_input)

        self.log_new_value('tau', tau_read)

        # KF
        self.log_new_value('fext_kf', fext_kf)
        self.log_new_value('tau_kf', tau_kf)

        # Reference
        self.log_new_value('qp_ref', qp_ref)
        self.log_new_value('qv_ref', qv_ref)

        # End-effector
        self.log_new_value('ee_pos', ee_pos)
        self.log_new_value('ee_vel', ee_vel)
        self.log_new_value('ee_pos_ref', ee_pos_ref)
        self.log_new_value('ee_vel_ref', ee_vel_ref)

        # MPC
        self.log_new_value('tau_mpc', tau_ctrl)
        self.log_new_value('diff_tau_mpc', np.zeros(self.nq))

    def log_delan_data(self, tau_delan):
        self.log_new_value('tau_delan', tau_delan)