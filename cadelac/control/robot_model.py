import casadi as cs
from casadi import vertcat
import pinocchio as pin
import pinocchio.casadi as cpin

import numpy as np
from pathlib import Path
from enum import Enum

def SoftPlusCS(x, beta = 1.0):
    return 1.0 / beta * cs.log(1 + cs.exp(beta * x))

class RealtimeApprox(Enum):
    NO_APPROX = 0
    CASADI = 1
    EXTERNAL_EVAL = 2
    NO_APPROX_SHARED_LIB = 3

class CasadiModel():
    def __init__(self,
                 delan_model = None,
                 realtime_approx : RealtimeApprox = RealtimeApprox.NO_APPROX,
                 ):

        robot_name = 'panda'
        xml_path = (
                Path(__file__).resolve().parent.parent.parent
                / "robots"
                / str(robot_name)
                / "panda.xml"
            ).as_posix()
        
        self.model_pin = pin.buildModelFromMJCF(xml_path)
        self.data_pin = self.model_pin.createData()
        self.model_cs = cpin.Model(self.model_pin)
        self.data_cs = self.model_cs.createData()

        self.nq = self.model_pin.nq
        self.nx = 2*self.nq
        self.nu = self.nq

        # self.casadi_opts = {"cse": True, "post_expand": True, "jit": True, "compiler": "clang"}
        self.casadi_opts = {"cse": True, "post_expand": True, "jit": True}
        # self.casadi_opts = {}
        # self.casadi_opts = {"cse": True, "post_expand": True, "compiler": "shell", "jit": True, "jit_options": {"compiler": "gcc"}}

        # Delan Model
        self.realtime_approx = realtime_approx
        self.delan_model = delan_model
        if self.delan_model == 'KF':
            self.model_type = 'KFNominal'
            self.param_nn = cs.SX.sym("tau_kf", self.nq)
        elif self.delan_model == 'KFJac':
            self.model_type = 'KFJacNominal'
            self.param_nn = cs.SX.sym("fee_kf", 3)
        elif self.delan_model is not None:
            self.build_delan_nn()
            self.param_nn = self.build_delan_fn()
            self.model_type = 'DelanNominal'
        else:
            self.model_type = 'Nominal'
            self.param_nn = cs.vertcat([])

        self.index_list_param = [0, self.nq]
        self.index_list_param.append(self.index_list_param[-1]+self.nq*self.nq)
        self.index_list_param.append(self.index_list_param[-1]+self.nq*self.nq*self.nq)
        self.index_list_param.append(self.index_list_param[-1]+1)
        self.index_list_param.append(self.index_list_param[-1]+self.nq)


    def build_delan_nn(self):
        self.l_output_size = int((self.nq ** 2 + self.nq) / 2)
        self.l_diag_size = self.nq
        self.l_lower_size = self.l_output_size - self.nq

        self.idx_l_diag = np.diag_indices(self.nq)
        self.idx_l_off_diag = np.tril_indices(self.nq, -1)

        self.n_enc_input = self.delan_model.n_enc_input

        if self.realtime_approx == RealtimeApprox.NO_APPROX:
            self.inertia_nn = self.delan_model.inertia_net_naive_comp
            self.potential_nn = self.delan_model.potential_net_naive_comp

        # Default l4c does not work because pinocchio.casadi uses SX, which forces l4_casadi
        # to be evaluated as SX. However, casadi does not allow to evaluate SX as external library
        # else:
        elif self.realtime_approx == RealtimeApprox.NO_APPROX_SHARED_LIB:
            q_delan = cs.MX.sym("q_delan_nn", self.nq, 1)
            x_delan = q_delan
            if self.n_enc_input > 1:
                enc_delan = cs.MX.sym("enc_delan_nn", self.n_enc_input, 1)
                x_delan = cs.vertcat(x_delan, enc_delan)

            self.inertia_nn_raw = self.delan_model.l4c_inertia_net
            self.potential_nn_raw = self.delan_model.l4c_potential_net

            self.inertia_nn = cs.Function('cs_inertia_nn',
                        [x_delan],
                        [self.inertia_nn_raw(x_delan)])
            self.potential_nn = cs.Function('cs_potential_nn',
                        [x_delan],
                        [self.potential_nn_raw(x_delan)])


    def build_delan_fn(self):

        q_delan = cs.SX.sym("q_delan", self.nq, 1)
        x_delan = q_delan
        param = cs.vertcat([])
        if self.n_enc_input > 1:
            enc_delan = cs.SX.sym("enc_delan", self.n_enc_input, 1)
            x_delan = cs.vertcat(x_delan, enc_delan)
            # param = cs.vertcat(param, enc_delan)

        ## Casadi functions for NN
        q_eval_delan = cs.SX.sym("q_eval_delan", self.nq, 1)
        x_eval_delan = q_eval_delan
        if self.n_enc_input > 1:
            enc_eval_delan = cs.SX.sym("enc_eval_delan", self.n_enc_input, 1)
            x_eval_delan = cs.vertcat(x_eval_delan, enc_eval_delan)

        if self.realtime_approx == RealtimeApprox.NO_APPROX:
            self.cs_inertia_fn = cs.Function('cs_inertia_fn',
                            [x_eval_delan],
                            [self.delan_model.lower_tri_inertia_fn_cs(self.inertia_nn(x_eval_delan))],
                            self.casadi_opts)
            self.cs_inertia_jac_fn = cs.Function('cs_inertia_jac_fn',
                            [x_eval_delan],
                            [cs.jacobian(self.delan_model.lower_tri_inertia_fn_cs(self.inertia_nn(x_eval_delan)), q_eval_delan)],
                            self.casadi_opts)
            self.cs_potential_fn = cs.Function('cs_potential_fn',
                            [x_eval_delan],
                            [self.potential_nn(x_eval_delan)],
                            self.casadi_opts)
            self.cs_potential_jac_fn = cs.Function('cs_potential_jac_fn',
                            [x_eval_delan],
                            [cs.jacobian(self.potential_nn(x_eval_delan), q_eval_delan)],
                            self.casadi_opts)
            
        if self.realtime_approx == RealtimeApprox.NO_APPROX_SHARED_LIB:
            self.cs_inertia_fn = cs.Function('cs_inertia_fn',
                            [x_eval_delan],
                            [self.delan_model.lower_tri_inertia_fn_cs(self.inertia_nn(x_eval_delan))])
            self.cs_inertia_jac_fn = cs.Function('cs_inertia_jac_fn',
                            [x_eval_delan],
                            [cs.jacobian(self.delan_model.lower_tri_inertia_fn_cs(self.inertia_nn(x_eval_delan)), q_eval_delan)])
            self.cs_potential_fn = cs.Function('cs_potential_fn',
                            [x_eval_delan],
                            [self.potential_nn(x_eval_delan)])
            self.cs_potential_jac_fn = cs.Function('cs_potential_jac_fn',
                            [x_eval_delan],
                            [cs.jacobian(self.potential_nn(x_eval_delan), q_eval_delan)])

        if self.realtime_approx == RealtimeApprox.CASADI:
            q_aprox = cs.MX.sym("q_aprox", self.nq, 1)
            x_aprox  = q_aprox
            if self.n_enc_input > 1:
                enc_aprox = cs.MX.sym("enc_aprox", self.n_enc_input, 1)
                x_aprox = cs.vertcat(x_aprox, enc_aprox)

            # Inertia
            yh_jac_approx = self.cs_inertia_jac_fn(x_aprox)
            yh_approx = self.cs_inertia_fn(x_aprox) + yh_jac_approx @ (q_delan - q_aprox)

            self.res_inertia_dyn_d = cs.Function('res_inertia_d',
                                    [x_aprox],
                                    [yh_jac_approx],
                                    {'is_diff_in': [0]} | self.casadi_opts)

            self.res_inertia_dyn = cs.Function('res_inertia',
                                    [x_delan, q_aprox],
                                    [yh_approx],
                                    self.casadi_opts)

            # Potential
            yv_jac_approx = self.cs_potential_jac_fn(x_aprox)
            yv_approx = self.cs_potential_fn(x_aprox) + yv_jac_approx @ (q_delan - q_aprox)

            self.res_potential_dyn_d = cs.Function('res_potential_d',
                                    [x_aprox],
                                    [yv_jac_approx],
                                    {'is_diff_in': [0]} | self.casadi_opts)

            self.res_potential_dyn = cs.Function('res_potential',
                                    [x_delan, q_aprox],
                                    [yv_approx],
                                    self.casadi_opts)


        elif self.realtime_approx == RealtimeApprox.EXTERNAL_EVAL:
            q_tay = cs.SX.sym("q_tay", self.nq, 1)
            x_tay = q_tay
            if self.n_enc_input > 1:
                enc_tay = cs.SX.sym("enc_tay", self.n_enc_input, 1)
                x_tay = cs.vertcat(x_tay, enc_tay)

            yh_q_tay = cs.SX.sym("yh_q_tay", self.nq*self.nq, 1)
            yh_jac_q_tay = cs.SX.sym("yh_jac_q_tay", self.nq*self.nq*self.nq, 1)
            yv_q_tay = cs.SX.sym("yv_q_tay", 1, 1)
            yv_jac_q_tay = cs.SX.sym("yv_jac_q_tay", self.nq, 1)

            # Inertia
            yh_tay_approx = yh_q_tay + yh_jac_q_tay.reshape((-1, self.nq)) @ (q_delan - q_tay)
            # yh_tay_approx = yh_q_tay
            self.res_inertia_dyn_d = cs.Function('res_inertia_d',
                                    [yh_jac_q_tay],
                                    [yh_jac_q_tay.reshape((-1, self.nq))],
                                    {'cse': True, 'jit': False, 'post_expand': True})

            self.res_inertia_dyn = cs.Function('res_inertia',
                                    [q_delan, q_tay, yh_q_tay, yh_jac_q_tay],
                                    [yh_tay_approx],
                                    {'cse': True, 'jit': False, 'post_expand': True})

            # Potential
            yv_tay_approx = yv_q_tay + yv_jac_q_tay.reshape((1, self.nq))  @ (q_delan - q_tay)
            # yv_tay_approx = yv_q_tay
            self.res_potential_dyn_d = cs.Function('res_potential_d',
                                    [yv_jac_q_tay],
                                    [yv_jac_q_tay],
                                    {'is_diff_in': [0], 'cse': True, 'jit': False, 'post_expand': True})

            self.res_potential_dyn = cs.Function('res_potential',
                                    [q_delan, q_tay, yv_q_tay, yv_jac_q_tay],
                                    [yv_tay_approx],
                                    {'cse': True, 'jit': False, 'post_expand': True})

            param = cs.vertcat(q_tay, yh_q_tay, yh_jac_q_tay, yv_q_tay, yv_jac_q_tay)

        # No realtime aprox
        else:
            yh_jac_true = self.cs_inertia_jac_fn(x_delan)
            yh_true = self.cs_inertia_fn(x_delan)
            self.res_inertia_dyn_d = cs.Function('res_inertia_d',
                                    [x_delan],
                                    [yh_jac_true],
                                    {'cse': True})
            self.res_inertia_dyn = cs.Function('res_inertia',
                                    [x_delan],
                                    [yh_true])

            yv_jac_true = self.cs_potential_jac_fn(x_delan)
            yv_true = self.cs_potential_fn(x_delan)
            self.res_potential_dyn_d = cs.Function('res_potential_d',
                                    [x_delan],
                                    [yv_jac_true],
                                    {'cse': True})
            self.res_potential_dyn = cs.Function('res_potential',
                                    [x_delan],
                                    [yv_true])

            if self.n_enc_input > 1:
                param = cs.vertcat(param, enc_delan)

        return param

    def model(self):
        q = cs.SX.sym("q", self.nq)
        qd = cs.SX.sym("qd", self.nq)
        tau = cs.SX.sym("tau", self.nq, 1)

        x = vertcat(qd, q)
        u = tau

        # Derivate for forward dynamics
        q_dot = cs.SX.sym("q_dot", self.nq, 1)
        qd_dot = cs.SX.sym("qd_dot", self.nq, 1)
        x_dot = vertcat(qd_dot, q_dot)

        param = vertcat([])
        # if self.realtime_approx == RealtimeApprox.EXTERNAL_EVAL:
        if self.param_nn.is_empty() is False:
            param = vertcat(param, self.param_nn)

        # Inverse Dynamics
        tau_inv_dyn_cpin = cpin.rnea(self.model_cs, self.data_cs, q, qd, qd_dot)
        self.tau_nom_inv_dyn_fn = cs.Function("tau_inv_dyn", [q, qd, qd_dot], [tau_inv_dyn_cpin], self.casadi_opts)

        # Forward Dynamics
        qdd_cpin = cpin.aba(self.model_cs, self.data_cs, q, qd, tau)
        self.acc_fn = cs.Function("acc", [q, qd, tau], [qdd_cpin], self.casadi_opts)

        acc_imp = cpin.crba(self.model_cs, self.data_cs, q) @ qd_dot
        acc_imp += cpin.nonLinearEffects(self.model_cs, self.data_cs, q, qd)
        acc_imp -= tau
        self.acc_imp_fn = cs.Function("acc_imp", [q, qd, qd_dot, tau], [acc_imp], self.casadi_opts)

        cpin_H = cpin.crba(self.model_cs, self.data_cs, q)
        cpin_tau_cg = cpin.nonLinearEffects(self.model_cs, self.data_cs, q, qd)
        self.cpin_H_fn = cs.Function("cpin_H", [q], [cpin_H], self.casadi_opts)
        self.cpin_tau_cg_fn = cs.Function("cpin_tau_cg", [q, qd], [cpin_tau_cg], self.casadi_opts)

        # Define Linear End effector Jacobian
        ee_id = self.model_cs.getFrameId("end_effector")
        cpin_Jee = cpin.computeFrameJacobian(self.model_cs, self.data_cs, q, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
        self.cpin_Jee_fn = cs.Function("cpin_Jee", [q], [cpin_Jee], self.casadi_opts)
        self.cpin_Jee_T_fn = cs.Function("cpin_Jee_T", [q], [cpin_Jee.T], self.casadi_opts)

        if self.delan_model is not None and self.delan_model != 'KF' and self.delan_model != 'KFJac':
            H_delan, tau_c_delan, tau_g_delan = self.eval_delan_terms(q, qd, param)
            self.delan_output_fn = cs.Function("delan_ouput_fn", [q, qd, param], [H_delan, tau_c_delan, tau_g_delan], self.casadi_opts)

        fwd_dyn = self.forward_dynamics(x, u, param)
        imp_fwd_dyn = self.imp_forward_dynamics(x_dot, x, u, param)

        # Store in Model Struct
        model = cs.types.SimpleNamespace()
        model.x = x
        model.xdot = x_dot
        model.u = u
        model.z = vertcat([])
        model.f_expl = fwd_dyn
        model.f_impl_expr = imp_fwd_dyn
        model.x_start = np.zeros((self.nx,))
        model.constraints = vertcat([])
        model.name = "panda_cs"
        model.p = param
        model.parameter_values = np.zeros(param.shape[0])
        
        return model

    def eval_delan_nn(self, x_delan, param = None):

        if self.realtime_approx == RealtimeApprox.NO_APPROX or self.realtime_approx == RealtimeApprox.NO_APPROX_SHARED_LIB:
            res_inertia = self.res_inertia_dyn(x_delan)
            res_inertia_d = self.res_inertia_dyn_d(x_delan)

            res_potential = self.res_potential_dyn(x_delan)
            res_potential_d = self.res_potential_dyn_d(x_delan)

        elif self.realtime_approx == RealtimeApprox.EXTERNAL_EVAL:
            # index_list_param = [0, self.nq]
            # index_list_param.append(index_list_param[-1]+self.nq*self.nq)
            # index_list_param.append(index_list_param[-1]+self.nq*self.nq*self.nq)
            # index_list_param.append(index_list_param[-1]+1)
            # index_list_param.append(index_list_param[-1]+self.nq)
            # q_tay, yh_q_tay, yh_jac_q_tay, yv_q_tay, yv_jac_q_tay = cs.vertsplit(param, index_list_param)
            q_tay, yh_q_tay, yh_jac_q_tay, yv_q_tay, yv_jac_q_tay = cs.vertsplit(param, self.index_list_param)

            res_inertia = self.res_inertia_dyn(x_delan, q_tay, yh_q_tay, yh_jac_q_tay)
            # res_inertia_d = self.res_inertia_dyn_d(yh_q_tay)
            res_inertia_d = yh_jac_q_tay.reshape((-1, self.nq))

            res_potential = self.res_potential_dyn(x_delan, q_tay, yv_q_tay, yv_jac_q_tay)
            # res_potential_d = self.res_potential_dyn_d(yv_jac_q_tay)
            res_potential_d = yv_jac_q_tay.reshape((-1, self.nq))

        return res_inertia, res_inertia_d, res_potential, res_potential_d

    def eval_delan_torque(self, q, qd, qdd, param):
        H, tau_c, tau_g = self.eval_delan_terms(q, qd, param)
        return H @ qdd + tau_c + tau_g

    def eval_delan_terms(self, q, qd, param = None):

        x_delan = q
        # Get enc_input only if exist and if NN was not evaluated externally
        if self.n_enc_input > 1 and self.realtime_approx != RealtimeApprox.EXTERNAL_EVAL:
            enc_input, param = cs.vertsplit(param, [0, self.n_enc_input, param.shape[0]])
            x_delan = cs.vertcat(x_delan, enc_input)

        delan_inertia, delan_inertia_d, delan_potential, delan_potential_d = self.eval_delan_nn(x_delan, param)

        l = delan_inertia.reshape((self.nq,self.nq))
        for i in range(self.nq):
            # l[i,i] = SoftPlusCS(l[i,i]) + 0.1
            l[i,i] = cs.fmax(0,l[i,i]) + 0.1
        dldq = delan_inertia_d

        V = delan_potential
        dVdq = delan_potential_d

        # Compute gravity
        tau_g = dVdq.reshape((self.nq,1))

        # Compute H
        lT = l.T
        H = l @ lT

        # Get propoer dldq1 and dldq2
        dldq_all = []
        for i in range(self.nq):
            dldq_all.append(dldq[:,i].reshape((self.nq,self.nq)))

        #  Compute dH/dt
        dldt = cs.SX.zeros(self.nq, self.nq)
        for i in range(self.nq):
            dldt += dldq_all[i] @ qd[i]
        dHdt = l @ dldt.T + dldt @ lT

        # Compute Coriolis
        quad_dq = cs.SX.zeros(self.nq)
        for i in range(self.nq):
            dHdqi = dldq_all[i] @ lT + l @ dldq_all[i].T
            quad_dq[i] = qd.T @ dHdqi @ qd

        dHdt_qd = dHdt @ qd
        tau_c = dHdt_qd - 1. / 2. * quad_dq

        return H, tau_c, tau_g

    def forward_dynamics(self, state: np.ndarray, input: np.ndarray, param: np.ndarray) -> cs.SX:
        qd, q = cs.vertsplit(state, [0, self.nq, 2*self.nq])
        tau = input

        if self.model_type == 'Nominal':
            qdd = self.acc_fn(q, qd, tau)
        elif self.model_type == 'KFNominal':
            qdd = self.acc_fn(q, qd, tau + param)
        elif self.model_type == 'KFJacNominal':
            qdd = self.acc_fn(q, qd, tau + self.cpin_Jee_T_fn(q) @ param)
        elif self.model_type == 'DelanNominal':
            # Get Nominal model
            H = self.cpin_H_fn(q)
            tau_cg = self.cpin_tau_cg_fn(q, qd)
            # H_delan, tau_delan_c, tau_delan_g = self.eval_delan_terms(q, qd, param)
            H_delan, tau_delan_c, tau_delan_g = self.delan_output_fn(q, qd, param)
            Hinv = cs.inv(H + H_delan)
            # Hinv = cs.solve(H + H_delan, cs.SX.eye(H.size1()))
            qdd = Hinv @ (tau - tau_cg - tau_delan_c - tau_delan_g)

        return vertcat(qdd, qd)

    def imp_forward_dynamics(self, state_dot, state, input, param) -> cs.SX:
        qd, q = cs.vertsplit(state, [0, self.nq, 2*self.nq])
        qd_dot, q_dot = cs.vertsplit(state_dot, [0, self.nq, 2*self.nq])
        tau = input

        # Get Nominal model
        H = self.cpin_H_fn(q)
        tau_cg = self.cpin_tau_cg_fn(q, qd)

        if self.model_type == 'Nominal':
            tau_total = H @ qd_dot + tau_cg - tau
        elif self.model_type == 'KFNominal':
            tau_total = H @ qd_dot + tau_cg - tau - param
        elif self.model_type == 'KFJacNominal':
            tau_total = H @ qd_dot + tau_cg - tau - self.cpin_Jee_T_fn(q) @ param
        elif self.model_type == 'DelanNominal':
            # H_delan, tau_delan_c, tau_delan_g = self.eval_delan_terms(q, qd, param)
            H_delan, tau_delan_c, tau_delan_g = self.delan_output_fn(q, qd, param)
            tau_total = (H + H_delan) @ qd_dot + tau_cg + tau_delan_c + tau_delan_g - tau

        return vertcat(tau_total, q_dot - qd)