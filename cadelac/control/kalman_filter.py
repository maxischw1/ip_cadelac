import numpy as np
import pinocchio as pin

class KalmanFilter():
    def __init__(self, nx, nu, ny, nd_known, nd, Ts):

        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.nd_known = nd_known
        self.nd = nd
        self.Ts = Ts

        self.Ad = np.zeros((self.nx, self.nx))
        self.Bd = np.zeros((self.nx, self.nu))
        self.Cd = np.zeros((self.ny, self.ny))
        self.Ed = np.zeros((self.nx, self.nd_known))
        self.Dd = np.zeros((self.nx, self.nd))

        self.Q = np.eye(self.nx)
        self.R = np.eye(self.ny)

        self.Pcovar = 1000*np.eye(self.nx)

    def set_Q(self, Q_new):
        self.Q = Q_new

    def set_R(self, R_new):
        self.R = R_new

    def set_Pcovar(self, Pcovar_new):
        self.Pcovar = Pcovar_new

    def set_model(self, Ac_new, Bc_new, Ec_new, Dc_new, Cc_new = None):
        self.Ad = np.eye(self.nx) + self.Ts * Ac_new
        self.Bd = self.Ts * Bc_new
        self.Dd = self.Ts * Dc_new
        self.Ed = self.Ts * Ec_new
        if Cc_new is not None:
            self.Cd = Cc_new

    def compute_prediction(self, x_old, u_old, dist_known_old):
        x_new = self.Ad @ x_old + self.Bd @ u_old + self.Ed @ dist_known_old
        return self.Cd @ x_new, x_new

    def compute_K(self, Pcovar_old):
        return Pcovar_old @ self.Cd.T @ np.linalg.inv(self.Cd @ Pcovar_old @ self.Cd.T + self.R)

    def update_estimation(self, x_pred, K, y_error, Pcovar_old):
        x_est = x_pred + K @ y_error
        Pcovar_est = (np.eye(self.nx) - K @ self.Cd) @ Pcovar_old
        return x_est, Pcovar_est

    def update(self, y_new, x_est_old, u_old, dist_known_old, Pcovar_old):
        # Update Prediction
        y_pred, x_pred = self.compute_prediction(x_est_old, u_old, dist_known_old)
        y_pred_error = y_new - y_pred

        # Update
        K = self.compute_K(Pcovar_old)
        x_est, Pcovar_est = self.update_estimation(x_pred, K, y_pred_error, Pcovar_old)

        # Update Error
        y_est_error = y_new - self.Cd @ x_est

        return x_est, Pcovar_est, y_est_error


class KFForceEE(KalmanFilter):
    def __init__(self, model_pin, data_pin, Ts, 
                 x_init = None, u_init = None):
        self.model_pin = model_pin
        self.data_pin = data_pin

        self.nq = 7
        nx = 2*self.nq # Linear and Angular wrench
        nu = 2*self.nq
        ny = nx
        nd_known = 3
        nd = 3

        super().__init__(nx, nu, ny, nd_known, nd, Ts)

        self.init_model(Ts)
        if x_init is None:
            self.x_old = np.zeros((self.nx, 1))
            self.y_old = np.zeros((self.ny, 1))
            self.x_est_old = np.zeros((self.nx, 1))
        else:
            self.x_old = x_init
            self.y_old = x_init
            self.x_est_old = x_init

        # if u_init is None:
        #     self.tau_old = np.zeros((self.nu, 1))
        # else:
        #     self.tau_old = u_init

        self.dist_known_old = np.zeros((self.nd_known, 1))

        self.eye3 = np.eye(3)        
        # self.old_robot_state = initial_state

        R = 1e3*np.diag(np.concatenate((1e-5*np.ones(self.nq), 1e-5*np.ones(self.nq))))
        Q = 1e2*np.diag(np.concatenate((1e1*np.ones(self.nq), 1e-2*np.ones(self.nq))))
        P = Q
        self.set_Q(Q)
        self.set_Pcovar(P)
        self.set_R(R)

        self.fext_est_max = 10*9.81

        self.ee_id = self.model_pin.getFrameId("end_effector")
        self.nfee = nd

    def init_model(self, Ts):
        self.Ad = np.eye(self.nx)
        self.Ad[self.nq:,:self.nq] = Ts * np.eye(self.nq)
        self.Bd = np.vstack((Ts * np.eye(self.nq), np.zeros((self.nq, self.nq))))
        self.Ed = -self.Bd
        self.Cd = np.eye(self.nx)
        # print(f'new Ad {self.Ad}')

    def compute_prediction(self, x_old, tau_old, Pcovar):
        qd_old = x_old[:self.nq]
        q_old = x_old[self.nq:]

        # Update Model
        inertia_mat = np.array(pin.crba(self.model_pin, self.data_pin, q_old))
        tau_known = np.array(pin.nonLinearEffects(self.model_pin, self.data_pin, q_old, qd_old))
        self.Bd[:self.nq,:] = self.Ts * np.linalg.inv(inertia_mat)
        self.Ed = -self.Bd

        x_pred = self.Ad @ x_old + self.Bd @ tau_old + self.Ed @ tau_known
        y_pred = self.Cd @ x_pred

        Pcovar_pred = self.Ad @ Pcovar @ self.Ad.T + self.Q

        return y_pred, x_pred, Pcovar_pred

    def get_observation(self, q, qd):
        x_new = np.concatenate((
            qd,
            q
        ))
        y_new = self.Cd @ x_new
        return y_new, x_new
    
    def get_force_ee_est(self, x_old, K, y_pred_error):
        q_old = x_old[self.nq:]
        inertia_mat = np.array(pin.crba(self.model_pin, self.data_pin, q_old))
        
        Jee = pin.computeFrameJacobian(self.model_pin, self.data_pin, q_old,
                                            self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
        Jee_T = np.array(Jee.T)

        # Compute mapping matrix
        Dd = np.zeros((self.nx, self.nfee))
        Dd[:self.nq,:] = self.Ts * np.linalg.inv(inertia_mat) @ Jee_T
        D_inv = np.linalg.pinv(Dd)

        force_ee_est = D_inv @ K @ y_pred_error
        force_ee_est = np.clip(force_ee_est, -self.fext_est_max, self.fext_est_max)

        tau_ee_est = Jee_T @ force_ee_est
        return force_ee_est, tau_ee_est


    def update(self, q, qd, tau_old):
        # Get observation
        y_new, x_new = self.get_observation(q, qd)

        # Update Prediction
        # y_pred, x_pred = self.compute_prediction(self.x_old, tau_old)
        y_pred, x_pred, Pcovar = self.compute_prediction(self.x_est_old, tau_old, self.Pcovar)
        y_pred_error = y_new - y_pred

        # Update
        K = self.compute_K(Pcovar)
        x_est, Pcovar_est = self.update_estimation(x_pred, K, y_pred_error, Pcovar)

        # Update Error
        y_est_error = y_new - self.Cd @ x_est
        # print(f'y new {y_new}')
        # print(f'y pred {y_pred}')
        # print(f'y pred error {y_pred_error}')
        # print(f'y est error {y_est_error}')
        # print(f'Pcovar_est {Pcovar_est}')

        # Compute force ext
        force_ee_est, qtau_fee_est = self.get_force_ee_est(self.x_old, K, y_pred_error)
        # force_ee_est, qtau_fee_est = self.get_force_ee_est(self.x_old, K, y_est_error)

        # Store data for next iteration
        self.x_est_old = np.copy(x_est)
        self.x_old = np.copy(x_new)
        self.Pcovar = Pcovar_est

        return force_ee_est, qtau_fee_est