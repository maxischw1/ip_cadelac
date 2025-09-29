from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, AcadosOcpOptions

import casadi as cs
import numpy as np
import scipy.linalg
import copy
import os

from cadelac.control.casadi_model import CasadiModel, RealtimeApprox

class AcadosMPC:
    def __init__(self,
                  N_horizon,
                  T_horizon,
                  delan_model = None,
                  expl_dyn = True,
                  name_suffix = '',
                  realtime_approx = RealtimeApprox.NO_APPROX,
                  RTI_mode = False,
                ):
        self.N_horizon = N_horizon
        self.T_horizon = T_horizon
        self.period_ctrl = T_horizon / N_horizon
        self.name_suffix = '_Ny_' + str(N_horizon) + name_suffix
        self.robot_model = CasadiModel(delan_model, realtime_approx)
        self.model = self.robot_model.model()
        self.expl_dyn = expl_dyn
        self.RTI_mode = RTI_mode

        self.tau_max = np.array([87, 87, 87, 87, 12, 12, 12])

        # Dimensions
        self.nq = self.robot_model.nq
        self.nx = self.model.x.rows()
        self.nu = self.model.u.rows()
        self.ny = self.nx + self.nu

        self.lib_dir = None
        # Not using this option: the default l4c is incompatible because pinocchio.casadi uses SX,
        # which forces l4_casadi to be evaluated as SX. However, CasADi does not support evaluating
        # SX expressions through external libraries.
        if delan_model is not None and delan_model != 'KF' and realtime_approx != RealtimeApprox.NO_APPROX:
            self.lib_dir = delan_model.l4c_inertia_net.shared_lib_dir
            self.lib_name = delan_model.l4c_inertia_net.name
            self.lib_name += ' -l' + delan_model.l4c_potential_net.name

        rl_mpc_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        compiled_controller = rl_mpc_folder + '/c_generated_code/'
        compiled_controller += 'libacados_ocp_solver_panda_cs' + self.name_suffix + '.so'

        if os.path.exists(compiled_controller):
            self.solver = AcadosOcpSolver(self.ocp(), build=False)
            print(f'Controller found at {compiled_controller}')
        else:
            print('Controller not found. Compiling it.')
            self.solver = AcadosOcpSolver(self.ocp())
    
    def ocp(self):
        # model = self.model
        T_horizon = self.T_horizon
        N = self.N_horizon

        # Get model
        model_ac = self.acados_model(model=self.model)

        # Create OCP object
        ocp = AcadosOcp()
        ocp.model = model_ac
        ocp.model.name = ocp.model.name + self.name_suffix
        ocp.dims.N = N
        ocp.dims.nx = self.nx
        ocp.dims.nu = self.nu
        ocp.dims.ny = self.ny
        ocp.solver_options.tf = T_horizon

        if model_ac.p.shape[0] > 0.0:
            ocp.parameter_values = model_ac.parameter_values

        # Set initial state constraint
        ocp.constraints.x0 = np.zeros(self.nx)

        ## Set Cost
        Q_mat, R_mat = self.get_weight_mat()

        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        ny_e = self.nx

        ocp.cost.W_e = Q_mat
        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)

        ocp.cost.Vx = np.zeros((self.ny, self.nx))
        ocp.cost.Vx[:self.nx, :self.nx] = np.eye(self.nx)

        Vu = np.zeros((self.ny, self.nu))
        Vu[self.nx : self.nx + self.nu, 0:self.nu] = np.eye(self.nu)
        ocp.cost.Vu = Vu

        ocp.cost.Vx_e = np.eye(self.nx)

        ocp.cost.yref = np.zeros((self.ny,))
        ocp.cost.yref_e = np.zeros((ny_e,))

        ## Set Constraints

        # Torque Limits
        ocp.constraints.lbu = -self.tau_max
        ocp.constraints.ubu = self.tau_max
        ocp.constraints.idxbu = np.arange(self.nu)

        # Solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        # ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        # ocp.solver_options.hessian_approx = 'EXACT'
        if self.expl_dyn:
            ocp.solver_options.integrator_type = 'ERK'
        else:
            ocp.solver_options.integrator_type = 'IRK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI' if self.RTI_mode else 'SQP'
        ocp.solver_options.nlp_solver_max_iter = 10
        ocp.solver_options.nlp_solver_tol_comp = 1e-5
        ocp.solver_options.nlp_solver_tol_eq = 1e-5
        ocp.solver_options.nlp_solver_tol_ineq = 1e-5


        if self.RTI_mode:
            ocp.solver_options.nlp_solver_warm_start_first_qp = True
            ocp.solver_options.qp_solver_warm_start = True

            ocp.solver_options.ext_fun_compile_flags = '-O3 -ffast-math -march=native'
            # ocp.solver_options.hpipm_mode = 'SPEED'

        ocp.solver_options.sens_forw = True
        ocp.solver_options.sim_method_jac_reuse = False
        ocp.solver_options.ext_fun_expand = True

        # Unecessary beucase of SX vs MX conflict between pinocchi and external library call
        if self.lib_dir is not None:
            ocp.solver_options.model_external_shared_lib_dir = self.lib_dir
            ocp.solver_options.model_external_shared_lib_name = self.lib_name

        return ocp


    def acados_model(self, model):
        model_ac = AcadosModel()
        model_ac.f_impl_expr = model.f_impl_expr
        model_ac.f_expl_expr = model.f_expl
        model_ac.x = model.x
        model_ac.xdot = model.xdot
        model_ac.u = model.u
        model_ac.name = model.name
        model_ac.p = model.p
        if model.p.shape[0] > 0.0:
            model_ac.parameter_values = model.parameter_values
        return model_ac
    

    def get_weight_mat(self):
        Qpos = 5*np.array([200, 200, 200, 200, 100, 100, 100])
        Qvel = np.array([100, 100, 100, 100, 50, 50, 50])
        Q_mat = np.diag(np.concatenate((Qvel, Qpos)))

        R_mat = np.diag(np.array([0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0]))

        return Q_mat, R_mat

    def init_solver(self, x0, u0):
        # Initialize solver
        for i in range(self.N_horizon + 1):
            self.solver.set(i, "x", x0)
        for i in range(self.N_horizon):
            self.solver.set(i, "u", u0)

    def get_mpc_solution_full_horizon(self):
        x_full_horizon = np.zeros((self.nx, self.N_horizon + 1))
        u_horizon = np.zeros((self.nu, self.N_horizon))
        for i in range(self.N_horizon + 1):
            x_full_horizon[:,i] = self.solver.get(i,'x')
        for i in range(self.N_horizon):
            u_horizon[:,i] = self.solver.get(i,'u')
        return u_horizon, x_full_horizon

    def set_mpc_ref(self, y_ref, u_ref):
        y_ref_with_input = np.vstack((y_ref[:,:self.N_horizon], u_ref))
        for i in range(self.N_horizon):
            self.solver.set(i, "yref", y_ref_with_input[:,i].T)
        self.solver.set(self.N_horizon, "yref", y_ref[:,self.N_horizon].T)

    def set_mpc_param(self, param):
        if param is not None:
            for i in range(self.N_horizon):
                if param.shape[1] > 1:
                    self.solver.set(i, "p", copy.deepcopy(param[i]))
                else:
                    self.solver.set(i, "p", copy.deepcopy(param))

    def iterate_solver_warm_start(self):
        for i in range(self.N_horizon-1):
            x_old = copy.deepcopy(self.solver.get(i+1,'x'))
            u_old = copy.deepcopy(self.solver.get(i+1,'u'))
            self.solver.set(i,'x',x_old)
            self.solver.set(i,'u',u_old)

    def get_solver_times(self):
        times_dict = {}
        times_dict['time_tot'] = self.solver.get_stats('time_tot')
        times_dict['time_lin'] = self.solver.get_stats('time_lin')
        times_dict['time_sim'] = self.solver.get_stats('time_sim')
        times_dict['time_qp'] = self.solver.get_stats('time_qp')
        times_dict['time_reg'] = self.solver.get_stats('time_reg')
        times_dict['time_sim_ad'] = self.solver.get_stats('time_sim_ad')
        times_dict['time_sim_la'] = self.solver.get_stats('time_sim_la')
        times_dict['time_solution_sens_lin'] = self.solver.get_stats('time_solution_sens_lin')
        return times_dict

    def print_solver_times(self):
        print(f"Total {self.solver.get_stats('time_tot')}")
        print(f"Lin {self.solver.get_stats('time_lin')}")
        print(f"Sim {self.solver.get_stats('time_sim')}")
        print(f"QP {self.solver.get_stats('time_qp')}")
        print(f"Reg {self.solver.get_stats('time_reg')}")
        print(f"Ext {self.solver.get_stats('time_sim_ad')}")
        print(f"Alg {self.solver.get_stats('time_sim_la')}")
        print(f"Prev {self.solver.get_stats('time_solution_sens_lin')}")