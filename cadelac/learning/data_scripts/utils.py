import dill as pickle
import numpy as np
import torch
import copy

import matplotlib as mp
import matplotlib.pyplot as plt
import os

def init_env(args):

    # Set the NumPy Formatter:
    np.set_printoptions(suppress=True, precision=2, linewidth=500,
                        formatter={'float_kind': lambda x: "{0:+08.2f}".format(x)})

    # Read the parameters:
    seed, cuda_id, cuda_flag = args.s[0], args.i[0], args.c[0]
    render, load_model, save_model, full_model = bool(args.r[0]), int(args.l[0]), bool(args.m[0]), bool(args.f[0])

    cuda_flag = cuda_flag and torch.cuda.is_available()

    # Set the seed:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set CUDA Device:
    if torch.cuda.device_count() > 1:
        assert cuda_id < torch.cuda.device_count()
        torch.cuda.set_device(cuda_id)

    return seed, cuda_flag, render, load_model, save_model, full_model

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
                     add_noise = False):

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
            data["diff_tau"][run] = data["diff_tau"][run] + np.random.normal(0, np.sqrt(var_diff_tau_nom_pin), data["diff_tau"][run].shape)

        print("\n################################################")
        print('Real robot noise added to data.')
        print("################################################")

    if hist_length > 0:
        data, data_hist = add_historical_data(hist_length, data, hist_labels)

    # Split the dataset in train and test set:
    if len(test_label) == 0:
        test_rate = 0.1
        n_test_label = test_rate * len(data["labels"])
        test_label = np.random.choice(data["labels"], size=int(n_test_label), replace=False)

    test_idx = [data["labels"].index(x) for x in test_label]

    dt = np.concatenate([data["t"][idx][1:] - data["t"][idx][:-1] for idx in test_idx])
    dt_mean, dt_var = np.mean(dt), np.var(dt)
    assert dt_var < 1.e-12

    train_labels, test_labels = [], []
    train_qp, train_qv, train_qa, train_tau = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))

    test_qp, test_qv, test_qa, test_tau = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))
    test_m, test_c, test_g = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))

    if hist_length > 0:
        train_hist_qp, train_hist_qv, train_hist_tau, train_hist_diff_tau_nom = np.zeros((0, hist_length, n_dof)), np.zeros((0, hist_length, n_dof)), np.zeros((0, hist_length, n_dof)), np.zeros((0, hist_length, n_dof))
        test_hist_qp, test_hist_qv, test_hist_tau, test_hist_diff_tau_nom = np.zeros((0, hist_length, n_dof)), np.zeros((0, hist_length, n_dof)), np.zeros((0, hist_length, n_dof)), np.zeros((0, hist_length, n_dof))

    divider = [0, ]   # Contains idx between characters for plotting

    for i in range(int(dataset_use*len(data["labels"]))):

        if i in test_idx:
            test_labels.append(data["labels"][i])
            test_qp = np.vstack((test_qp, data["qp"][i][sample_offset:]))
            test_qv = np.vstack((test_qv, data["qv"][i][sample_offset:]))
            test_qa = np.vstack((test_qa, data["qa"][i][sample_offset:]))

            if full_model:
                test_tau = np.vstack((test_tau, data["tau"][i][sample_offset:]))

                test_m = np.vstack((test_m, data["tau_m"][i][sample_offset:]))
                test_c = np.vstack((test_c, data["tau_c"][i][sample_offset:]))
                test_g = np.vstack((test_g, data["tau_g"][i][sample_offset:]))
            else:
                test_tau = np.vstack((test_tau, data["diff_tau"][i][sample_offset:]))

                test_m = np.vstack((test_m, data["diff_tau_m"][i][sample_offset:]))
                test_c = np.vstack((test_c, data["diff_tau_c"][i][sample_offset:]))
                test_g = np.vstack((test_g, data["diff_tau_g"][i][sample_offset:]))

            if hist_length > 0:
                test_hist_qp = np.vstack((test_hist_qp, data_hist["qp"][i][sample_offset:]))
                test_hist_qv = np.vstack((test_hist_qv, data_hist["qv"][i][sample_offset:]))
                test_hist_tau = np.vstack((test_hist_tau, data_hist["tau"][i][sample_offset:]))
                test_hist_diff_tau_nom = np.vstack((test_hist_diff_tau_nom, data_hist["diff_tau"][i][sample_offset:]))

            divider.append(test_qp.shape[0])

        else:
            train_labels.append(data["labels"][i])
            train_qp = np.vstack((train_qp, data["qp"][i][sample_offset:]))
            train_qv = np.vstack((train_qv, data["qv"][i][sample_offset:]))
            train_qa = np.vstack((train_qa, data["qa"][i][sample_offset:]))

            if full_model:
                train_tau = np.vstack((train_tau, data["tau"][i][sample_offset:]))
            else:
                train_tau = np.vstack((train_tau, data["diff_tau"][i][sample_offset:]))

            if hist_length > 0:
                train_hist_qp = np.vstack((train_hist_qp, data_hist["qp"][i][sample_offset:]))
                train_hist_qv = np.vstack((train_hist_qv, data_hist["qv"][i][sample_offset:]))
                train_hist_tau = np.vstack((train_hist_tau, data_hist["tau"][i][sample_offset:]))
                train_hist_diff_tau_nom = np.vstack((train_hist_diff_tau_nom, data_hist["diff_tau"][i][sample_offset:]))

    if hist_length > 0:
        train_data = (train_labels, train_qp, train_qv, train_qa, train_tau, \
                     train_hist_qp, train_hist_qv, train_hist_tau, train_hist_diff_tau_nom)
        test_data = (test_labels, test_qp, test_qv, test_qa, test_tau, test_m, test_c, test_g, \
                     test_hist_qp, test_hist_qv, test_hist_tau, test_hist_diff_tau_nom)       
    else:
        train_data = (train_labels, train_qp, train_qv, train_qa, train_tau)
        test_data = (test_labels, test_qp, test_qv, test_qa, test_tau, test_m, test_c, test_g)

    return train_data, test_data, divider, dt_mean

def parition_params(module_name, name, value, key):
    return module_name.split("/")[0] == key


def plot_torques(test_tau, test_m, test_c, test_g, delan_tau, delan_m, delan_c, delan_g, test_labels, divider, fig_dir, render = True):


    print("\n################################################")
    print("Plotting Performance:")

    n_dof = test_tau.shape[-1]

    # Alpha of the graphs:
    plot_alpha = 0.8

    # Plot the performance:
    y_t_low = np.clip(1.2 * np.min(np.vstack((test_tau, delan_tau)), axis=0), -np.inf, -0.01)
    y_t_max = np.clip(1.5 * np.max(np.vstack((test_tau, delan_tau)), axis=0), 0.01, np.inf)

    y_m_low = np.clip(1.2 * np.min(np.vstack((test_m, delan_m)), axis=0), -np.inf, -0.01)
    y_m_max = np.clip(1.2 * np.max(np.vstack((test_m, delan_m)), axis=0), 0.01, np.inf)

    y_c_low = np.clip(1.2 * np.min(np.vstack((test_c, delan_c)), axis=0), -np.inf, -0.01)
    y_c_max = np.clip(1.2 * np.max(np.vstack((test_c, delan_c)), axis=0), 0.01, np.inf)

    y_g_low = np.clip(1.2 * np.min(np.vstack((test_g, delan_g)), axis=0), -np.inf, -0.01)
    y_g_max = np.clip(1.2 * np.max(np.vstack((test_g, delan_g)), axis=0), 0.01, np.inf)
    if n_dof % 2 == 1:
        y_t_low = np.concatenate((y_t_low, -10*np.ones(1)))
        y_t_max = np.concatenate((y_t_max, 10*np.ones(1)))
        y_m_low = np.concatenate((y_m_low, -10*np.ones(1)))
        y_m_max = np.concatenate((y_m_max, 10*np.ones(1)))
        y_c_low = np.concatenate((y_c_low, -10*np.ones(1)))
        y_c_max = np.concatenate((y_c_max, 10*np.ones(1)))
        y_g_low = np.concatenate((y_g_low, -10*np.ones(1)))
        y_g_max = np.concatenate((y_g_max, 10*np.ones(1)))

    color_i = ["r", "b", "g", "k"]

    ticks = np.array(divider)
    ticks = (ticks[:-1] + ticks[1:]) / 2

    for i in range(0, n_dof, 2):

        fig = plt.figure(figsize=(24.0/1.54, 8.0/1.54), dpi=100)
        fig.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=0.95, wspace=0.3, hspace=0.2)

        legend = [mp.patches.Patch(color=color_i[0], label="DeLaN"),
                mp.patches.Patch(color="k", label="Ground Truth")]

        # Plot Torque
        ax0 = fig.add_subplot(2, 4, 1)
        ax0.set_title('Torque')
        ax0.text(s=f'Joint {i}', x=-0.35, y=.5, fontsize=12, fontweight="bold", rotation=90, horizontalalignment="center", verticalalignment="center", transform=ax0.transAxes)
        ax0.set_ylabel("Torque [Nm]")
        ax0.get_yaxis().set_label_coords(-0.2, 0.5)
        ax0.set_ylim(y_t_low[i+0], y_t_max[i+0])
        ax0.set_xticks(ticks)
        ax0.set_xticklabels(test_labels)
        ax0.vlines(divider, y_t_low[i+0], y_t_max[i+0], linestyles='--', lw=0.5, alpha=1.)
        ax0.set_xlim(divider[0], divider[-1])

        ax1 = fig.add_subplot(2, 4, 5)
        ax1.text(s=f'Joint {i+1}', x=-.35, y=0.5, fontsize=12, fontweight="bold", rotation=90,
                horizontalalignment="center", verticalalignment="center", transform=ax1.transAxes)

        ax1.text(s=r"$\mathbf{(a)}$", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
                verticalalignment="center", transform=ax1.transAxes)

        ax1.set_ylabel("Torque [Nm]")
        ax1.get_yaxis().set_label_coords(-0.2, 0.5)
        ax1.set_ylim(y_t_low[i+1], y_t_max[i+1])
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(test_labels)
        ax1.vlines(divider, y_t_low[i+1], y_t_max[i+1], linestyles='--', lw=0.5, alpha=1.)
        ax1.set_xlim(divider[0], divider[-1])

        ax0.legend(handles=legend, bbox_to_anchor=(0.0, 1.0), loc='upper left', ncol=1, framealpha=1.)

        # Plot Ground Truth Torque:
        ax0.plot(test_tau[:, i+0], color="k")
        if i+1 < n_dof:
            ax1.plot(test_tau[:, i+1], color="k")

        # Plot DeLaN Torque:
        ax0.plot(delan_tau[:, i+0], color=color_i[0], alpha=plot_alpha)
        if i+1 < n_dof:
            ax1.plot(delan_tau[:, i+1], color=color_i[0], alpha=plot_alpha)

        # Plot Mass Torque
        ax0 = fig.add_subplot(2, 4, 2)
        ax0.set_title(r"$\mathbf{H}(\mathbf{q}) \ddot{\mathbf{q}}$")
        ax0.set_ylabel("Torque [Nm]")
        ax0.set_ylim(y_m_low[i+0], y_m_max[i+0])
        ax0.set_xticks(ticks)
        ax0.set_xticklabels(test_labels)
        ax0.vlines(divider, y_m_low[i+0], y_m_max[i+0], linestyles='--', lw=0.5, alpha=1.)
        ax0.set_xlim(divider[0], divider[-1])

        ax1 = fig.add_subplot(2, 4, 6)
        ax1.text(s=r"$\mathbf{(b)}$", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
                verticalalignment="center", transform=ax1.transAxes)

        ax1.set_ylabel("Torque [Nm]")
        ax1.set_ylim(y_m_low[i+1], y_m_max[i+1])
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(test_labels)
        ax1.vlines(divider, y_m_low[i+1], y_m_max[i+1], linestyles='--', lw=0.5, alpha=1.)
        ax1.set_xlim(divider[0], divider[-1])

        # Plot Ground Truth Inertial Torque:
        ax0.plot(test_m[:, i+0], color="k")
        if i+1 < n_dof:
            ax1.plot(test_m[:, i+1], color="k")

        # Plot DeLaN Inertial Torque:
        ax0.plot(delan_m[:, i+0], color=color_i[0], alpha=plot_alpha)
        if i+1 < n_dof:
            ax1.plot(delan_m[:, i+1], color=color_i[0], alpha=plot_alpha)

        # Plot Coriolis Torque
        ax0 = fig.add_subplot(2, 4, 3)
        ax0.set_title(r"$\mathbf{c}(\mathbf{q}, \dot{\mathbf{q}})$")
        ax0.set_ylabel("Torque [Nm]")
        ax0.set_ylim(y_c_low[i+0], y_c_max[i+0])
        ax0.set_xticks(ticks)
        ax0.set_xticklabels(test_labels)
        ax0.vlines(divider, y_c_low[i+0], y_c_max[i+0], linestyles='--', lw=0.5, alpha=1.)
        ax0.set_xlim(divider[0], divider[-1])

        ax1 = fig.add_subplot(2, 4, 7)
        ax1.text(s=r"$\mathbf{(c)}$", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
                verticalalignment="center", transform=ax1.transAxes)

        ax1.set_ylabel("Torque [Nm]")
        ax1.set_ylim(y_c_low[i+1], y_c_max[i+1])
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(test_labels)
        ax1.vlines(divider, y_c_low[i+1], y_c_max[i+1], linestyles='--', lw=0.5, alpha=1.)
        ax1.set_xlim(divider[0], divider[-1])

        # Plot Ground Truth Coriolis Torque:
        ax0.plot(test_c[:, i+0], color="k")
        if i+1 < n_dof:
            ax1.plot(test_c[:, i+1], color="k")

        # Plot DeLaN Coriolis Torque:
        ax0.plot(delan_c[:, i+0], color=color_i[0], alpha=plot_alpha)
        if i+1 < n_dof:
            ax1.plot(delan_c[:, i+1], color=color_i[0], alpha=plot_alpha)

        # Plot Gravity
        ax0 = fig.add_subplot(2, 4, 4)
        ax0.set_title(r"$\mathbf{g}(\mathbf{q})$")
        ax0.set_ylabel("Torque [Nm]")
        ax0.set_ylim(y_g_low[i+0], y_g_max[i+0])
        ax0.set_xticks(ticks)
        ax0.set_xticklabels(test_labels)
        ax0.vlines(divider, y_g_low[i+0], y_g_max[i+0], linestyles='--', lw=0.5, alpha=1.)
        ax0.set_xlim(divider[0], divider[-1])

        ax1 = fig.add_subplot(2, 4, 8)
        ax1.text(s=r"$\mathbf{(d)}$", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
                verticalalignment="center", transform=ax1.transAxes)

        ax1.set_ylabel("Torque [Nm]")
        ax1.set_ylim(y_g_low[i+1], y_g_max[i+1])
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(test_labels)
        ax1.vlines(divider, y_g_low[i+1], y_g_max[i+1], linestyles='--', lw=0.5, alpha=1.)
        ax1.set_xlim(divider[0], divider[-1])

        # Plot Ground Truth Gravity Torque:
        ax0.plot(test_g[:, i+0], color="k")
        if i+1 < n_dof:
            ax1.plot(test_g[:, i+1], color="k")

        # Plot DeLaN Gravity Torque:
        ax0.plot(delan_g[:, i+0], color=color_i[0], alpha=plot_alpha)
        if i+1 < n_dof:
            ax1.plot(delan_g[:, i+1], color=color_i[0], alpha=plot_alpha)

        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)
        fig.savefig(f"{fig_dir}/joints_{i}_{i+1}.pdf", format="pdf")
        fig.savefig(f"{fig_dir}/joints_{i}_{i+1}.png", format="png")

    if render:
        plt.show()

    print("\n################################################\n\n\n")