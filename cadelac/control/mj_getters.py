import numpy as np

import mujoco

class MjGetters():
    def get_joint_pos(self, mj_data):
        return mj_data.qpos[self.joint_ids].squeeze()
    
    def get_joint_vel(self, mj_data):
        return mj_data.qvel[self.joint_ids].squeeze()
    
    def get_joint_acc(self, mj_data):
        return mj_data.qacc[self.joint_ids].squeeze()
    
    def get_joint_torques(self, mj_data):
        return mj_data.qfrc_actuator[self.joint_ids].squeeze()
    
    def get_ee_pos(self, mj_data):
        return np.copy(mj_data.xpos[self.ee_id])
    
    def get_ee_vel(self, mj_data):
        return np.copy(mj_data.cvel[self.ee_id][3:])

    def get_box_pos(self, mj_data):
        return np.copy(mj_data.xpos[self.box_id])
    
    def get_box_vel(self, mj_data):
        return np.copy(mj_data.cvel[self.box_id][3:])
    
    def get_Mq(self, mj_model, mj_data):
        Mq = np.zeros((mj_model.nq,mj_model.nq))
        mujoco.mj_fullM(mj_model, Mq, mj_data.qM)
        return np.copy(Mq[np.ix_(self.joint_ids, self.joint_ids)])
    
    def get_Mq_inv(self, mj_model, mj_data):
        M_inv = np.zeros((mj_model.nq,mj_model.nq))
        mujoco.mj_solveM(mj_model,mj_data, M_inv, np.eye(mj_model.nv))
        return np.copy(M_inv[np.ix_(self.joint_ids, self.joint_ids)])
    
    def get_model_ids(self, mujoco_model: mujoco.MjModel, act_blacklist = []):
        joint_ids = []
        for i in range(mujoco_model.actuator_actnum.shape[0]):
            act_name = mujoco_model.actuator(i).name
            if act_name not in act_blacklist:
                joint_name = act_name.replace('actuator','joint') 
                joint_ids.append(mujoco_model.joint(joint_name).id)

        ee_id = mujoco_model.body('end_effector').id

        if 'ee_box' in str(mujoco_model.names):
            box_id = mujoco_model.body('ee_box').id
        else:
            box_id = None

        return joint_ids, ee_id, box_id

    def get_joint_ranges(self, mujoco_model: mujoco.MjModel, joint_ids):
        ranges = np.zeros((len(joint_ids),2))
        for i in joint_ids:
            ranges[i,0] = mujoco_model.joint(i).range[0]
            ranges[i,1] = mujoco_model.joint(i).range[1]
        return ranges

    def get_body_jac(self, mj_model, mj_data, body_id, rot_jac = False):
        nv = mj_model.nv
        jac_lin = np.zeros((3,nv))
        jac_rot = np.zeros((3,nv)) if rot_jac else None
        mujoco.mj_jacBody(mj_model, mj_data, jac_lin, jac_rot, body_id)

        if rot_jac:
            return jac_lin[:,self.joint_ids], jac_rot[:,self.joint_ids]
        else:
            return jac_lin[:,self.joint_ids]
