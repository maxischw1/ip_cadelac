import pinocchio as pin
import numpy as np
from numpy.linalg import norm, solve
from scipy.spatial.transform import Rotation

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


import numpy as np
import pinocchio as pin

def inverse_kinematics_with_orientation(pin_model, pin_data, q_init, ee_id, ee_ref, R_ref, 
                                       eps=1e-4, max_iter=100, alpha=0.5):
    """
    Inverse kinematics solving for both position and orientation.

    Parameters:
    - pin_model: Pinocchio robot model
    - pin_data: Pinocchio data structure
    - q_init: Initial joint positions
    - ee_id: End-effector frame index
    - ee_ref: Desired end-effector position (3D vector)
    - R_ref: Desired end-effector rotation matrix (3x3)
    - q_min, q_max: Joint limits
    - eps: Convergence tolerance
    - max_iter: Maximum number of iterations
    - alpha: Damping factor (reduces large jumps)

    Returns:
    - q: Joint configuration solving IK
    - success: True if solution found, False otherwise
    """
    
    q = np.copy(q_init)  # Start with initial guess
    success = False
    damp = 1e-2
    DT = 1e-3

    for _ in range(max_iter):
        # Forward kinematics
        pin.forwardKinematics(pin_model, pin_data, q)
        pin.updateFramePlacements(pin_model, pin_data)

        # Get current position and orientation of end-effector
        ee_pose = pin_data.oMf[ee_id]  # SE(3) transformation matrix
        ee_pos = ee_pose.translation  # Extract position
        ee_rot = ee_pose.rotation  # Extract orientation

        # Compute position error
        pos_error = ee_ref - ee_pos  # 3x1 vector

        # Compute orientation error using rotation matrix log map (axis-angle)
        R_error = ee_rot.T @ R_ref  # Rotation error: R_e.T * R_des
        angle_axis_error = pin.log3(R_error)  # Convert rotation matrix difference to axis-angle
        ori_error = -ee_rot @ angle_axis_error  # Convert to base frame

        # Stack position and orientation errors into a 6x1 vector
        error = np.hstack((pos_error, ori_error))

        # Convergence check
        if np.linalg.norm(error) < eps:
            success = True
            break  # Solution found

        # Compute full 6D Jacobian (Position + Orientation)
        J = pin.computeFrameJacobian(pin_model, pin_data, q, ee_id) #, pin.LOCAL_WORLD_ALIGNED)

        # # Solve for joint updates using damped least squares (DLS)
        # J_damped = J.T @ np.linalg.inv(J @ J.T + damp * np.eye(6))  # Moore-Penrose pseudo-inverse with damping
        # dq = alpha * J_damped @ error  # Scale step size to prevent overshooting

        # # Update joint positions
        # q += dq

        v = J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), error))
        q = pin.integrate(pin_model, q, v * DT)



    if not success:
        print(
            "\nWarning: the iterative algorithm has not reached convergence to "
            "the desired precision"
        )


    return q


def inverse_kinematics_v2(pin_model, pin_data, q0, ee_ref, rot_ref = None, ee_id = None):

    eps = 1e-4
    IT_MAX = 2000
    DT = 1e-3
    damp = 1e-2

    # oMdes = pin.SE3(np.eye(3), ee_ref)
    oMdes = pin.SE3(np.eye(3), ee_ref)
    q = q0

    q_min = pin_model.lowerPositionLimit
    q_max = pin_model.upperPositionLimit

    JOINT_ID = 7
    i = 0
    while True:
        pin.forwardKinematics(pin_model, pin_data, q)
        iMd = pin_data.oMi[JOINT_ID].actInv(oMdes)

        err = pin.log(iMd).vector  # in joint frame
        if norm(err) < eps:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break
        J = pin.computeJointJacobian(pin_model, pin_data, q, JOINT_ID)
        J = pin.Jlog6(iMd.inverse()) @ J
        v = J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
        q = pin.integrate(pin_model, q, v * DT)

        q = np.clip(q, q_min, q_max)

        # if not i % 10:
            # print("%d: error = %s" % (i, err.T))
        i += 1
    return q

def inverse_kinematics_v3(pin_model, pin_data, q0, ee_ref, rot_ref=None, ee_id=None):
    """
    Computes inverse kinematics while enforcing joint limits.

    Parameters:
    - pin_model: Pinocchio robot model.
    - pin_data: Pinocchio data structure.
    - q0: Initial joint configuration.
    - ee_ref: Desired end-effector position (3D vector).
    - rot_ref: Desired end-effector rotation (3x3 matrix). If None, assumes identity.
    - ee_id: End-effector frame ID.

    Returns:
    - q: Optimized joint configuration within limits.
    - success: True if convergence was achieved, False otherwise.
    """
    eps = 1e-4  # Convergence tolerance
    IT_MAX = 1000  # Max iterations
    DT = 1e-1  # Step size
    damp = 1e-12  # Regularization factor

    # Get joint limits
    q_min = pin_model.lowerPositionLimit
    q_max = pin_model.upperPositionLimit

    # Default to identity rotation if none is provided
    if rot_ref is None:
        # rot_ref = np.eye(3)
        rot_ref = Rotation.from_euler('x', 180, degrees=True).as_matrix()

    # Desired transformation (position + rotation)
    oMdes = pin.SE3(rot_ref, ee_ref)
    
    # Initialize joint configuration
    q = q0

    i = 0
    while True:
        # Forward kinematics
        pin.forwardKinematics(pin_model, pin_data, q)
        pin.updateFramePlacements(pin_model, pin_data)

        # Compute the transformation error
        iMd = pin_data.oMf[ee_id].actInv(oMdes)  # End-effector frame error

        # Compute error as a 6D vector (position + orientation)
        err = pin.log(iMd).vector  

        # Convergence check
        if norm(err) < eps:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break

        # Compute the full 6D Jacobian (position + orientation)
        J = pin.computeFrameJacobian(pin_model, pin_data, q, ee_id, pin.LOCAL_WORLD_ALIGNED)
        
        # Adjust Jacobian using log-space transformation
        J = pin.Jlog6(iMd.inverse()) @ J
        
        # Compute joint update using damped least squares
        v = J.T @ solve(J @ J.T + damp * np.eye(6), err)

        # Update joint configuration
        q = pin.integrate(pin_model, q, v * DT)

        # Enforce joint limits 
        q = np.clip(q, q_min, q_max)  # Ensures q stays within the limits

        i += 1  # Increment iteration counter


    if not success:
        print(
            "\nWarning: the iterative algorithm has not reached convergence to "
            "the desired precision"
        )

    return q

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
        
        # J_damped = J.T @ np.linalg.inv(J @ J.T + damp * np.eye(3))
        # dq = J_damped @ error
        # q += dq
        
        # if q > 2*np.pi:
        # q = q % (np.pi)

        # J = pin.computeJointJacobian(pin_model, pin_data, JOINT_ID)
        # v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(3), error))
        # q = pin.integrate(pin_model, q, v * DT)

    # if rot_ref is None:
    #     oMdes = pin.SE3(np.eye(3), ee_ref)
    # else:
    #     oMdes = pin.SE3(rot_ref, ee_ref)

    # it = 0
    # while True:
    #     pin.forwardKinematics(pin_model, pin_data, q)
    #     iMd = pin_data.oMi[JOINT_ID].actInv(oMdes)
    #     err = iMd.translation
    #     # err = pin.log(iMd).vector

    #     # pin.updateFramePlacements(pin_model, pin_data)
    #     # iMd = pin_data.oMf[ee_id]
    #     # ee_pos = pin_data.oMf[ee_id].translation
    #     # err = pin.log(iMd).vector
        
    #     if norm(err) < eps:
    #         success = True
    #         break
    #     if it >= max_iter:
    #         success = False
    #         break

    #     J = pin.computeJointJacobian(pin_model, pin_data, q, JOINT_ID)  # in joint frame
    #     J = -J[:3, :]  # linear part of the Jacobian
    #     v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(3), err))

    #     # J = pin.computeJointJacobian(pin_model, pin_data, q, JOINT_ID)  # in joint frame
    #     # J = -np.dot(pin.Jlog6(iMd.inverse()), J)
    #     # v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))


    #     q = pin.integrate(pin_model, q, v * DT)
    #     it += 1

    # # if success:
    #     print("Convergence achieved!")
    # else:
    #     print(
    #         "\nWarning: the iterative algorithm has not reached convergence to "
    #         "the desired precision"
    #     )

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

    # pin.forwardKinematics(pin_model, pin_data, q0)
    # rot_ref = pin_data.oMi[7].rotation

    # pin.updateFramePlacements(pin_model, pin_data)
    # rot_ref = pin_data.oMf[ee_id].rotation
    rot_ref = None

    # pin.forwardKinematics(pin_model, pin_data, q0)
    # pin.updateFramePlacements(pin_model, pin_data)
    # rot_ref = pin_data.oMf[ee_id].rotation

    for i in range(traj_length):
        q_new = inverse_kinematics(pin_model, pin_data, q_old, ee_pos_traj[i,:], rot_ref)
        # rot_ref = np.eye(3)
        # q_new = inverse_kinematics_with_orientation(pin_model, pin_data, q_old, ee_id, ee_pos_traj[i,:], rot_ref)

        # q_new = inverse_kinematics_v2(pin_model, pin_data, q_old, ee_pos_traj[i,:])
        # q_new = inverse_kinematics_v3(pin_model, pin_data, q_old, ee_pos_traj[i,:], ee_id=ee_id, rot_ref=rot_ref)

        J = pin.computeFrameJacobian(pin_model, pin_data, q_new, ee_id, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        qd_new = np.linalg.pinv(J) @ ee_vel_traj[i,:]
        # J = pin.computeFrameJacobian(pin_model, pin_data, q_new, ee_id, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        # qd_new = np.zeros_like(q_new)
        # qd_new = J.T @ ee_vel_traj[i,:]
        # if i == 0:
        #     qd_new = np.zeros_like(q_old)
        # else:
        #     qd_new = (q_new - q_traj[-1]) / Ts

        q_traj.append(q_new)
        qd_traj.append(qd_new)
        q_old = q_new.copy()

    VEL_THRESHOLD = 1e-6
    # qd_traj = np.array(qd_traj)
    # qd_traj[np.abs(qd_traj) < VEL_THRESHOLD] = 0.0
    
    return np.array(q_traj), np.array(qd_traj)

def traj_fwd_kinematics(pin_model, pin_data, q_traj):

    traj_length = len(q_traj)
    ee_traj = []
    ee_id = pin_model.getFrameId("end_effector")

    for i in range(traj_length):
        new_ee_pos = pin_get_link_pos(pin_model, pin_data, q_traj[i,:], ee_id)
        ee_traj.append(new_ee_pos)

    return np.array(ee_traj)