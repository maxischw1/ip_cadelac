import dill as pickle

filename = '/home/schulze/projects/cadelac/cadelac/learning/data/datasets/panda/env_0_env_10_ref_exc_joint_init3_panda_box_rand_mass_rand_envs_101_box_pos_rand_1_box_mass_rand_nominal_1_kf_comp_0_samples_1040300_sim_time_10_dataset_lqr_freq_015.pkl'
with open(filename, 'rb') as f:
    data = pickle.load(f)

print(data.keys())

# print(data['ee_vel_ref'][0].shape)
# print(data['ee_vel'])