import numpy as np
import pinocchio as pin
from cadelac.control.kalman_filter import KalmanFilter

class KFStateFee(KalmanFilter):
    def __init__(self, model_pin, data_pin, Ts, 
                 q_init = None, qd_init = None, 
                 Fee_init = None, u_init = None):
        self.model_pin = model_pin
        self.data_pin = data_pin
        self.ee_id = self.model_pin.getFrameId("end_effector")

        self.nq = 7
        self.nFee = 3
        nx = 2*self.nq + self.nFee
        nu = self.nq
        ny = 2*self.nq
        nd_known = 3
        nd = 2*self.nq + self.nFee
        self.old_Jee = np.zeros((self.nFee, self.nq))

        super().__init__(nx, nu, ny, nd_known, nd, Ts)

        q_init = q_init if q_init is not None else np.zeros((self.nq))
        qd_init = qd_init if qd_init is not None else np.zeros((self.nq))
        Fee_init = Fee_init if Fee_init is not None else np.zeros((self.nFee))
        u_init = u_init if u_init is not None else np.zeros((self.nq))

        self.init_model(q_init)

        # Initial Statess
        self.x_old = np.concatenate((qd_init,
                                     q_init,
                                     Fee_init))
        self.x_est_old = np.copy(self.x_old)

        self.eye3 = np.eye(3)

        R = np.diag(np.concatenate((1e-2*np.ones(self.nq), 1e-2*np.ones(self.nq))))
        # Order qd, q, F
        Q = np.diag(np.concatenate((1e0*np.ones(self.nq), 1e-1*np.ones(self.nq), 1e2*np.ones(self.nFee))))
        P = Q
        self.set_Q(Q)
        self.set_Pcovar(P)
        self.set_R(R)

        self.fext_est_max = 10*9.81


    def init_model(self, q_init):
        
        # Get Inetia matrix
        inertia_mat = np.array(pin.crba(self.model_pin, self.data_pin, q_init))
        inertia_mat_inv = np.linalg.inv(inertia_mat)
        
        # Get Linear Jee
        Jee = pin.computeFrameJacobian(self.model_pin, self.data_pin, q_init,
                                        self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:self.nFee, :]

        # Build continuos state space 
        Ac = np.zeros((self.nx, self.nx))
        Ac[self.nq:2*self.nq, self.nq:2*self.nq] = np.eye(self.nq)
        Ac[:self.nq, 2*self.nq:] = inertia_mat_inv @ (Jee.T)

        Bc = np.zeros((self.nx, self.nu))
        Bc[:self.nq,:] = inertia_mat_inv

        Ec = np.zeros((self.nx, self.nu))
        Ec[:self.nq,:] = -inertia_mat_inv

        Dc = np.eye(self.nd)

        Cc = np.eye(self.ny,self.nx)

        self.Ad = np.eye(self.nx) + self.Ts * Ac
        self.Bd = self.Ts * Bc
        self.Ed = self.Ts * Ec
        self.Dd = Dc
        self.Cd = Cc

    def compute_prediction(self, x_old, tau_old, Pcovar):
        qd_old, q_old, _ = self.get_data_from_state(x_old)

        # Update Model
        inertia_mat = np.array(pin.crba(self.model_pin, self.data_pin, q_old))
        inertia_mat_inv = np.linalg.inv(inertia_mat)

        # Get Linear Jee
        Jee = pin.computeFrameJacobian(self.model_pin, self.data_pin, q_old,
                                        self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:self.nFee, :]

        tau_known = np.array(pin.nonLinearEffects(self.model_pin, self.data_pin, q_old, qd_old))

        ## Update State space
        Ac = np.zeros((self.nx, self.nx))
        Ac[self.nq:2*self.nq, self.nq:2*self.nq] = np.eye(self.nq)
        Ac[:self.nq, 2*self.nq:] = inertia_mat_inv @ (Jee.T)

        Bc = np.zeros((self.nx, self.nu))
        Bc[:self.nq,:] = inertia_mat_inv

        Ec = np.zeros((self.nx, self.nu))
        Ec[:self.nq,:] = -inertia_mat_inv

        Dc = np.eye(self.nd)

        self.Ad = np.eye(self.nx) + self.Ts * Ac
        self.Bd = self.Ts * Bc
        self.Ed = self.Ts * Ec
        self.Dd = Dc

        x_pred = self.Ad @ x_old + self.Bd @ tau_old + self.Ed @ tau_known
        y_pred = self.Cd @ x_pred

        Pcovar_pred = self.Ad @ Pcovar @ self.Ad.T + self.Dd @ self.Q @ self.Dd.T

        return y_pred, x_pred, Pcovar_pred

    def get_observation(self, q, qd):
        y_new = np.concatenate((
            qd,
            q
        ))
        return y_new
    
    def get_data_from_state(self, x_state):
        qd, q, Fee = x_state[:self.nq], x_state[self.nq:2*self.nq], x_state[2*self.nq:]
        return qd, q, Fee
    
    def get_force_ee_est(self, x_new, x_old):
        qd_old, q_old, _ = self.get_data_from_state(x_old)
        
        Jee = pin.computeFrameJacobian(self.model_pin, self.data_pin, q_old,
                                       self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:self.nFee, :]
        Jee_T = np.array(Jee.T)

        _, _, force_ee_est = self.get_data_from_state(x_new)

        force_ee_est = np.clip(force_ee_est, -self.fext_est_max, self.fext_est_max)

        tau_ee_est = Jee_T @ force_ee_est
        return force_ee_est, tau_ee_est

    def update(self, q_new, qd_new, tau_old):
        # Get observation
        y_new = self.get_observation(q_new, qd_new)

        # Update Prediction
        y_pred, x_pred, Pcovar = self.compute_prediction(self.x_est_old, tau_old, self.Pcovar)
        y_pred_error = y_new - y_pred

        # Update
        K = self.compute_K(Pcovar)
        x_est, Pcovar_est = self.update_estimation(x_pred, K, y_pred_error, Pcovar)

        # Update Error
        y_est_error = y_new - self.Cd @ x_est
 
        # Compute force ext
        force_ee_est, qtau_fee_est = self.get_force_ee_est(x_est, self.x_est_old)

        # Store data for next iteration
        self.x_est_old = np.copy(x_est)
        self.Pcovar = Pcovar_est

        return force_ee_est, qtau_fee_est