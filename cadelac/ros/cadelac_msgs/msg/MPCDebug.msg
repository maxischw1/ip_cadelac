# Joint
float32[] q
float32[] qd
float32[] qdd
float32[] qdd_est
float32[] tau_read
float32[] tau_ctrl

float32[] q_ref
float32[] qd_ref

# End-effector
float32[] ee_pos
float32[] ee_vel

float32[] ee_pos_ref
float32[] ee_vel_ref


# MPC
float32[] mpc_time
float32[] x_mpc
float32[] xpred_next
float32[] tau_mpc

# Kalman Filter
float32[] tau_kf
float32[] fext_kf

float32[] enc_input
float32[] enc_input_raw

# Model
float32[] tau_nom_pin
float32[] m_nom_pin
float32[] c_nom_pin
float32[] g_nom_pin
float32[] diff_tau_nom_pin
float32[] diff_tau_m_nom_pin
float32[] diff_tau_c_nom_pin
float32[] diff_tau_g_nom_pin

# Delan
float64[] tau_total_nom_delan

float64[] delan_tau
float64[] delan_tau_m
float64[] delan_tau_cg
float64[] delan_tau_c
float64[] delan_tau_g

# Times
float32[] times_values
std_msgs/String[] time_names