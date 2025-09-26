import dill as pickle
import numpy as np
import torch
import copy

from cadelac.learning.data_scripts.utils_panda import add_historical_data

def init_env(args):

    # Set the NumPy Formatter:
    np.set_printoptions(suppress=True, precision=2, linewidth=500,
                        formatter={'float_kind': lambda x: "{0:+08.2f}".format(x)})

    # Read the parameters:
    seed, cuda_id, cuda_flag = args.s[0], args.i[0], args.c[0]
    render, load_model, save_model = bool(args.r[0]), bool(args.l[0]), bool(args.m[0])

    cuda_flag = cuda_flag and torch.cuda.is_available()

    # Set the seed:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set CUDA Device:
    if torch.cuda.device_count() > 1:
        assert cuda_id < torch.cuda.device_count()
        torch.cuda.set_device(cuda_id)

    return seed, cuda_flag, render, load_model, save_model

def add_historical_data(hist_length, data,
                        list_key = ['qp', 'qv', 'tau', 'diff_tau_nom']):
    
    new_data = copy.deepcopy(data)

    data_hist = {}
    for key in list_key:
        data_hist[key] = [[] for _ in range(len(data[key]))]

    for key in data.keys():
        if key in list_key:
            for index, run_data in enumerate(data[key]):
                hist_values = []
                for i in range(len(run_data)-hist_length):
                    hist_values.append(run_data[i:(i+hist_length),:])
                data_hist[key][index] = np.array(hist_values)

        # Delete data before hist_length
        if key != 'labels':
            for index, run_data in enumerate(data[key]):
                new_data[key][index] = new_data[key][index][hist_length:]
    
    return new_data, data_hist


def load_dataset(filename="data/character_data.pickle", test_label=("run_0","run_1","run_2"),
                     full_model = True, sample_offset = 0, dataset_use = 1.0,
                     n_dof = 2, hist_length = 0, hist_labels = ['qp', 'qv', 'tau', 'diff_tau_nom'],
                     return_ref = False, dt = None, add_noise = False):

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    if add_noise:
        var_qp = np.zeros(7)
        var_qv = 0.001 * np.ones(7)
        var_qa = [0.05, 0.05, 0.15, 0.03, 0.15, 0.25, 0.65]
        var_tau_read = [0.5, 0.1, 0.5, 0.3, 0.01, 0.01, 0.01]
        var_diff_tau_nom_pin = [1.0, 0.5, 1.0, 0.4, 0.02, 0.03, 0.02]

        for run in range(len(data['qp'])):
            data["qp"][run] = data["qp"][run] + np.random.normal(0, np.sqrt(var_qp), data["qp"][run].shape)
            data["qv"][run] = data["qv"][run] + np.random.normal(0, np.sqrt(var_qv), data["qv"][run].shape)
            data["qa"][run] = data["qa"][run] + np.random.normal(0, np.sqrt(var_qa), data["qa"][run].shape)
            data["tau"][run] = data["tau"][run] + np.random.normal(0, np.sqrt(var_tau_read), data["qv"][run].shape)
            data["diff_tau_nom"][run] = data["diff_tau_nom"][run] + np.random.normal(0, np.sqrt(var_diff_tau_nom_pin), data["diff_tau_nom"][run].shape)
            data["diff_tau_nom_pin"][run] = data["diff_tau_nom_pin"][run] + np.random.normal(0, np.sqrt(var_diff_tau_nom_pin), data["diff_tau_nom_pin"][run].shape)

        print('Noise added')

    if hist_length > 0:
        data, data_hist = add_historical_data(hist_length, data, hist_labels)

    # Split the dataset in train and test set:
    if len(test_label) == 0:
        test_rate = 0.1
        n_test_label = test_rate * len(data["labels"])
        # test_label = data["labels"][:int(n_test_label)]
        test_label = np.random.choice(data["labels"], size=int(n_test_label), replace=False)

    test_idx = [data["labels"].index(x) for x in test_label]

    dt = np.concatenate([data["t"][idx][1:] - data["t"][idx][:-1] for idx in test_idx])
    dt_mean, dt_var = np.mean(dt), np.var(dt)
    assert dt_var < 1.e-12

    train_labels, test_labels = [], []
    train_qp, train_qv, train_qa, train_tau = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))
    train_p, train_pd = np.zeros((0, n_dof)), np.zeros((0, n_dof))

    test_qp, test_qv, test_qa, test_tau = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))
    test_m, test_c, test_g = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))
    test_p, test_pd = np.zeros((0, n_dof)), np.zeros((0, n_dof))

    train_qp_ref, train_qv_ref, test_qp_ref, test_qv_ref = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))

    if hist_length > 0:
        train_hist_qp, train_hist_qv, train_hist_tau, train_hist_diff_tau_nom = np.zeros((0, hist_length, n_dof)), np.zeros((0, hist_length, n_dof)), np.zeros((0, hist_length, n_dof)), np.zeros((0, hist_length, n_dof))
        test_hist_qp, test_hist_qv, test_hist_tau, test_hist_diff_tau_nom = np.zeros((0, hist_length, n_dof)), np.zeros((0, hist_length, n_dof)), np.zeros((0, hist_length, n_dof)), np.zeros((0, hist_length, n_dof))

    divider = [0, ]   # Contains idx between characters for plotting

    env_ids = np.array(data['env_id'], dtype=np.int32)
    unique_env_ids = np.unique(env_ids)
    # one_hot_targets = np.eye(len(unique_env_ids))[env_ids]
    one_hot_targets = np.eye(len(unique_env_ids))[env_ids].squeeze()

    if not full_model:
        train_one_hot, test_one_hot = np.zeros((0, len(unique_env_ids))), np.zeros((0, len(unique_env_ids)))
    else:
        train_one_hot, test_one_hot = np.zeros((0, one_hot_targets.shape[1])), np.zeros((0, one_hot_targets.shape[1]))

    for i in range(int(dataset_use*len(data["labels"]))):

        if i in test_idx:
            test_labels.append(data["labels"][i])
            test_qp = np.vstack((test_qp, data["qp"][i][sample_offset:]))
            test_qv = np.vstack((test_qv, data["qv"][i][sample_offset:]))
            test_qa = np.vstack((test_qa, data["qa"][i][sample_offset:]))
            test_qp_ref = np.vstack((test_qp_ref, data["qp_ref"][i][sample_offset:]))
            test_qv_ref = np.vstack((test_qv_ref, data["qv_ref"][i][sample_offset:]))

            if full_model:
                test_tau = np.vstack((test_tau, data["tau"][i][sample_offset:]))

                test_m = np.vstack((test_m, data["m"][i][sample_offset:]))
                test_c = np.vstack((test_c, data["c"][i][sample_offset:]))
                test_g = np.vstack((test_g, data["g"][i][sample_offset:]))
            else:
                test_tau = np.vstack((test_tau, data["diff_tau_nom_pin"][i][sample_offset:]))

                test_m = np.vstack((test_m, data["diff_tau_m_nom_pin"][i][sample_offset:]))
                test_c = np.vstack((test_c, data["diff_tau_c_nom_pin"][i][sample_offset:]))
                test_g = np.vstack((test_g, data["diff_tau_g_nom_pin"][i][sample_offset:])) # Copy the same value since it isn't used

            test_one_hot = np.vstack((test_one_hot, one_hot_targets[i][sample_offset:]))

            if hist_length > 0:
                test_hist_qp = np.vstack((test_hist_qp, data_hist["qp"][i][sample_offset:]))
                test_hist_qv = np.vstack((test_hist_qv, data_hist["qv"][i][sample_offset:]))
                test_hist_tau = np.vstack((test_hist_tau, data_hist["tau"][i][sample_offset:]))
                test_hist_diff_tau_nom = np.vstack((test_hist_diff_tau_nom, data_hist["diff_tau_nom_pin"][i][sample_offset:]))

            test_p = np.vstack((test_p, data["p"][i][sample_offset:]))
            test_pd = np.vstack((test_pd, data["pdot"][i][sample_offset:]))
            divider.append(test_qp.shape[0])

        else:
            train_labels.append(data["labels"][i])
            train_qp = np.vstack((train_qp, data["qp"][i][sample_offset:]))
            train_qv = np.vstack((train_qv, data["qv"][i][sample_offset:]))
            train_qa = np.vstack((train_qa, data["qa"][i][sample_offset:]))
            train_qp_ref = np.vstack((train_qp_ref, data["qp_ref"][i][sample_offset:]))
            train_qv_ref = np.vstack((train_qv_ref, data["qv_ref"][i][sample_offset:]))

            if full_model:
                train_tau = np.vstack((train_tau, data["tau"][i][sample_offset:]))
            else:
                train_tau = np.vstack((train_tau, data["diff_tau_nom_pin"][i][sample_offset:]))

            train_one_hot = np.vstack((train_one_hot, one_hot_targets[i][sample_offset:]))

            if hist_length > 0:
                train_hist_qp = np.vstack((train_hist_qp, data_hist["qp"][i][sample_offset:]))
                train_hist_qv = np.vstack((train_hist_qv, data_hist["qv"][i][sample_offset:]))
                train_hist_tau = np.vstack((train_hist_tau, data_hist["tau"][i][sample_offset:]))
                train_hist_diff_tau_nom = np.vstack((train_hist_diff_tau_nom, data_hist["diff_tau_nom_pin"][i][sample_offset:]))

            train_p = np.vstack((train_p, data["p"][i][sample_offset:]))
            train_pd = np.vstack((train_pd, data["pdot"][i][sample_offset:]))

    if hist_length > 0:
        train_data = (train_labels, train_qp, train_qv, train_qa, train_p, train_pd, train_tau, train_one_hot, \
                     train_hist_qp, train_hist_qv, train_hist_tau, train_hist_diff_tau_nom)
        test_data = (test_labels, test_qp, test_qv, test_qa, test_p, test_pd, test_tau, test_m, test_c, test_g, test_one_hot, \
                     test_hist_qp, test_hist_qv, test_hist_tau, test_hist_diff_tau_nom)       
    else:
        train_data = (train_labels, train_qp, train_qv, train_qa, train_p, train_pd, train_tau, train_one_hot)
        test_data = (test_labels, test_qp, test_qv, test_qa, test_p, test_pd, test_tau, test_m, test_c, test_g, test_one_hot)

    if return_ref:
        ref_data = (train_qp_ref, train_qv_ref, test_qp_ref, test_qv_ref)
        return train_data, test_data, divider, dt_mean, ref_data
    else:
        return train_data, test_data, divider, dt_mean

def parition_params(module_name, name, value, key):
    return module_name.split("/")[0] == key
