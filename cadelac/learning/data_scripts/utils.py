import dill as pickle
import numpy as np
import torch

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


def load_dataset(n_characters=3, filename="data/character_data.pickle", test_label=("e", "q", "v")):

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    n_dof = 2

    # Split the dataset in train and test set:

    # Random Test Set:
    # test_idx = np.random.choice(len(data["labels"]), n_characters, replace=False)

    # Specified Test Set:
    # test_char = ["e", "q", "v"]
    test_idx = [data["labels"].index(x) for x in test_label]

    dt = np.concatenate([data["time"][idx][1:] - data["time"][idx][:-1] for idx in test_idx])
    dt_mean, dt_var = np.mean(dt), np.var(dt)
    assert dt_var < 1.e-12

    train_labels, test_labels = [], []
    train_qp, train_qv, train_qa, train_tau = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))
    train_p, train_pd = np.zeros((0, n_dof)), np.zeros((0, n_dof))

    test_qp, test_qv, test_qa, test_tau = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))
    test_m, test_c, test_g = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))
    test_p, test_pd = np.zeros((0, n_dof)), np.zeros((0, n_dof))

    divider = [0, ]   # Contains idx between characters for plotting

    for i in range(len(data["labels"])):

        if i in test_idx:
            test_labels.append(data["labels"][i])
            test_qp = np.vstack((test_qp, data["qp"][i]))
            test_qv = np.vstack((test_qv, data["qv"][i]))
            test_qa = np.vstack((test_qa, data["qa"][i]))
            test_tau = np.vstack((test_tau, data["tau"][i]))

            test_m = np.vstack((test_m, data["m"][i]))
            test_c = np.vstack((test_c, data["c"][i]))
            test_g = np.vstack((test_g, data["g"][i]))

            test_p = np.vstack((test_p, data["p"][i]))
            test_pd = np.vstack((test_pd, data["pdot"][i]))
            divider.append(test_qp.shape[0])

        else:
            train_labels.append(data["labels"][i])
            train_qp = np.vstack((train_qp, data["qp"][i]))
            train_qv = np.vstack((train_qv, data["qv"][i]))
            train_qa = np.vstack((train_qa, data["qa"][i]))
            train_tau = np.vstack((train_tau, data["tau"][i]))

            train_p = np.vstack((train_p, data["p"][i]))
            train_pd = np.vstack((train_pd, data["pdot"][i]))

    return (train_labels, train_qp, train_qv, train_qa, train_p, train_pd, train_tau), \
           (test_labels, test_qp, test_qv, test_qa, test_p, test_pd, test_tau, test_m, test_c, test_g),\
           divider, dt_mean

def load_dataset_mpc(filename="data/character_data.pickle", test_label=("run_0","run_1","run_2"),
                     full_model = True, sample_offset = 1, dataset_use = 1.0, one_hot_input = False,
                     n_dof = 2, seq_length = 0):

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    if seq_length > 0:
        data, data_hist = add_historical_data(seq_length, data)
        # sample_offset = seq_length + 1

    # Split the dataset in train and test set:
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

    divider = [0, ]   # Contains idx between characters for plotting

    if one_hot_input:
        env_ids = np.array(data['env_id'], dtype=np.int32)
    else:
        env_ids = np.array(data['env_id'], dtype=np.int32)
    unique_env_ids = np.unique(env_ids)
    one_hot_targets = np.eye(len(unique_env_ids))[env_ids]

    train_one_hot, test_one_hot = np.zeros((0, len(unique_env_ids))), np.zeros((0, len(unique_env_ids)))

    for i in range(int(dataset_use*len(data["labels"]))):

        if i in test_idx:
            test_labels.append(data["labels"][i])
            test_qp = np.vstack((test_qp, data["qp"][i][sample_offset:]))
            test_qv = np.vstack((test_qv, data["qv"][i][sample_offset:]))
            test_qa = np.vstack((test_qa, data["qa"][i][sample_offset:]))

            if full_model:
                test_tau = np.vstack((test_tau, data["tau"][i][sample_offset:]))

                test_m = np.vstack((test_m, data["m"][i][sample_offset:]))
                test_c = np.vstack((test_c, data["c"][i][sample_offset:]))
                test_g = np.vstack((test_g, data["g"][i][sample_offset:]))
            else:
                test_tau = np.vstack((test_tau, data["diff_tau_nom"][i][sample_offset:]))

                test_m = np.vstack((test_m, data["diff_tau_m_nom"][i][sample_offset:]))
                test_c = np.vstack((test_c, data["diff_tau_c_nom"][i][sample_offset:]))
                test_g = np.vstack((test_g, data["diff_tau_g_nom"][i][sample_offset:])) # Copy the same value since it isn't used

            test_one_hot = np.vstack((test_one_hot, one_hot_targets[i][sample_offset:]))

            test_p = np.vstack((test_p, data["p"][i][sample_offset:]))
            test_pd = np.vstack((test_pd, data["pdot"][i][sample_offset:]))
            divider.append(test_qp.shape[0])

        else:
            train_labels.append(data["labels"][i])
            train_qp = np.vstack((train_qp, data["qp"][i][sample_offset:]))
            train_qv = np.vstack((train_qv, data["qv"][i][sample_offset:]))
            train_qa = np.vstack((train_qa, data["qa"][i][sample_offset:]))

            if full_model:
                train_tau = np.vstack((train_tau, data["tau"][i][sample_offset:]))
            else:
                train_tau = np.vstack((train_tau, data["diff_tau_nom"][i][sample_offset:]))

            train_one_hot = np.vstack((train_one_hot, one_hot_targets[i][sample_offset:]))

            train_p = np.vstack((train_p, data["p"][i][sample_offset:]))
            train_pd = np.vstack((train_pd, data["pdot"][i][sample_offset:]))

    # if one_hot_input:
    return (train_labels, train_qp, train_qv, train_qa, train_p, train_pd, train_tau, train_one_hot), \
        (test_labels, test_qp, test_qv, test_qa, test_p, test_pd, test_tau, test_m, test_c, test_g, test_one_hot),\
        divider, dt_mean

def parition_params(module_name, name, value, key):
    return module_name.split("/")[0] == key
