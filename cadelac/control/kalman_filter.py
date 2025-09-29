import numpy as np

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
