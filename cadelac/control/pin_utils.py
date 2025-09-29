import pinocchio as pin
import numpy as np

def pin_get_frame_jacobian(pin_model, pin_data, q, frame_id, compute_rot_jac=False):
    jacobian = pin.computeFrameJacobian(pin_model, pin_data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)
    lin_jac = jacobian[:3,:]

    if compute_rot_jac:
        rot_jac = jacobian[3:6,:]

    if compute_rot_jac:
        return lin_jac, rot_jac
    else:
        return np.copy(lin_jac)

def pin_get_link_pos(pin_model, pin_data, q, frame_id):
    pin.framesForwardKinematics(pin_model, pin_data, q)
    return np.copy(pin_data.oMf[frame_id].translation)

def inverse_kinematics(pin_model, pin_data, q0, ee_ref, rot_ref = None, ee_id = None):
    """Compute inverse kinematics using an iterative approach."""
    max_iter = 10_000
    eps = 1e-4
    DT = 1e-3
    damp = 1e-2

    JOINT_ID = 7
    q_min = pin_model.lowerPositionLimit
    q_max = pin_model.upperPositionLimit

    q = q0.copy()
    if ee_id is None:
        ee_id = pin_model.getFrameId("end_effector")

    success = False
    for _ in range(max_iter):
        pin.forwardKinematics(pin_model, pin_data, q)
        pin.updateFramePlacements(pin_model, pin_data)
        ee_pos = pin_data.oMf[ee_id].translation
        error = ee_ref - ee_pos
        if np.linalg.norm(error) < eps:
            success = True
            return q
        J = pin.computeFrameJacobian(pin_model, pin_data, q, ee_id, pin.LOCAL_WORLD_ALIGNED)[:3, :]

        dq = np.linalg.pinv(J) @ error
        q += 0.5*dq
        
        q = np.clip(q, q_min, q_max)

    if not success:
        print(
            "\nWarning: the iterative algorithm has not reached convergence to "
            "the desired precision"
        )


    return q

def traj_inverse_kinematics(pin_model, pin_data, q0, ee_pos_traj, ee_vel_traj):

    traj_length = len(ee_pos_traj)
    q_old = q0.copy()
    q_traj = []
    qd_traj = []
    ee_id = pin_model.getFrameId("end_effector")

    rot_ref = None

    for i in range(traj_length):
        q_new = inverse_kinematics(pin_model, pin_data, q_old, ee_pos_traj[i,:], rot_ref)

        J = pin.computeFrameJacobian(pin_model, pin_data, q_new, ee_id, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        qd_new = np.linalg.pinv(J) @ ee_vel_traj[i,:]

        q_traj.append(q_new)
        qd_traj.append(qd_new)
        q_old = q_new.copy()
    
    return np.array(q_traj), np.array(qd_traj)

def traj_fwd_kinematics(pin_model, pin_data, q_traj):

    traj_length = len(q_traj)
    ee_traj = []
    ee_id = pin_model.getFrameId("end_effector")

    for i in range(traj_length):
        new_ee_pos = pin_get_link_pos(pin_model, pin_data, q_traj[i,:], ee_id)
        ee_traj.append(new_ee_pos)

    return np.array(ee_traj)