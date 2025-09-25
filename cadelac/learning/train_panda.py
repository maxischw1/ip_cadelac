import argparse
import torch
import numpy as np
import time
import os

import matplotlib as mp

if os.getenv("DISPLAY"):
    try:
        mp.use("Qt5Agg")
        mp.rc('text', usetex=True)
        mp.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    except ImportError:
        pass

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from cadelac.learning.models.DeLaN_model import DeepLagrangianNetwork
from cadelac.learning.models.context_aware_delan import ContextAwareDeLaN

from cadelac.learning.data_scripts.replay_memory import PyTorchReplayMemory
from cadelac.learning.data_scripts.utils import init_env
from cadelac.learning.data_scripts.utils_panda import load_dataset_panda_mpc


if __name__ == "__main__":

    # Read Command Line Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", nargs=1, type=int, required=False, default=[True, ], help="Training using CUDA.")
    parser.add_argument("-i", nargs=1, type=int, required=False, default=[0, ], help="Set the CUDA id.")
    parser.add_argument("-s", nargs=1, type=int, required=False, default=[0, ], help="Set the random seed")
    parser.add_argument("-r", nargs=1, type=int, required=False, default=[1, ], help="Render the figure")
    parser.add_argument("-l", nargs=1, type=int, required=False, default=[0, ], help="Load the DeLaN model")
    parser.add_argument("-m", nargs=1, type=int, required=False, default=[1, ], help="Save the DeLaN model")
    seed, cuda, render, load_model, save_model = init_env(parser.parse_args())

    # Construct Hyperparameters:
    nn_id = "ContextAware"
    if nn_id == "default":
        nn_type = DeepLagrangianNetwork
    elif nn_id == "ContextAware":
        nn_type = ContextAwareDeLaN

    # Read the dataset:
    box_mass = 1.0
    box_pos = np.array([0.1, 0.15])
    dataset_use = 1.0
    samples = 100000
    tau_max = 100
    sim_total_time = 20
    embedding_dim = 3
    shared_embedding_layer = 1
    minibatch = 1024
    # minibatch = 512
    loss_power = False

    n_dof = 7
    one_hot_input = False
    full_model = False
    flag_normalize_tau = True
    sample_offset = 0
    save_checkpoint_model = False
    log_period = 5

    add_tau_noise = False # old flag
    add_noise = False # old flag
    add_noise_to_load_data = False


    ## LSTM parameters
    hist_length = 15 if (nn_id == "ContextAware" and not full_model) else 0
    hist_labels = ['qp', 'qv', 'tau', 'diff_tau_nom_pin']
    n_lstm_input = n_dof * 3
    n_lstm_hidden = 10
    n_lstm_output = 10
    n_lstm_depth = 5

    if hist_length > 0:

        filename = 'env_0_env_10_ref_exc_joint_init3_panda_box_rand_mass_rand_envs_101_box_pos_rand_1_box_mass_rand_nominal_1_kf_comp_0_samples_1040300_sim_time_10_dataset_lqr_freq_015'
        filename_short = '10_envs_ref_exec_init3_rand_envs_nom_101_kf_0_samples_1040300_stime_10'

    else:

        filename = 'env_0_ref_exc_joint_init3_panda_box_rand_mass_rand_envs_101_box_pos_rand_1_box_mass_rand_nominal_1_kf_comp_0_samples_1040300_sim_time_10_dataset_lqr_freq_015'
        filename_short = 'env_0_ref_exec_init3_rand_envs_nom_101_kf_0_samples_1040300_stime_10'

    if full_model:
        test_label = ['env_0_run_1','env_0_run_2']
        test_label = []
        model_type_folder = 'full_model/panda/' + nn_id
    else:
        if hist_length > 0:
           test_label = []
        else:
            test_label = ['env_0_run_1','env_0_run_2']
        model_type_folder = 'res_model/panda/' + nn_id
    filename_full = 'learning/data/panda/' + filename + '.pkl'

    if hist_length > 0:
        filename_full = filename_full
    else:
        embedding_dim = 0

    train_data, test_data, divider, dt_mean = load_dataset_panda_mpc(filename=filename_full, test_label=test_label,
                                                                    full_model=full_model, sample_offset=sample_offset,
                                                                    dataset_use=dataset_use, 
                                                                    n_dof=n_dof, hist_length=hist_length, 
                                                                    hist_labels=hist_labels,
                                                                    add_noise=add_noise_to_load_data)

    if hist_length == 0:
        train_labels, train_qp, train_qv, train_qa, train_p, train_pd, train_tau, train_one_hot = train_data
        test_labels, test_qp, test_qv, test_qa, test_p, test_pd, test_tau, test_m, test_c, test_g, test_one_hot = test_data
        # n_enc_input = train_one_hot.shape[-1]
        n_enc_input = 1
        train_lstm_input = np.ones_like(train_one_hot)
        test_lstm_input = np.ones_like(test_one_hot)
    else:
        train_labels, train_qp, train_qv, train_qa, train_p, train_pd, train_tau, train_one_hot, \
                     tain_hist_qp, tain_hist_qv, tain_hist_tau, tain_hist_diff_tau_nom = train_data
        test_labels, test_qp, test_qv, test_qa, test_p, test_pd, test_tau, test_m, test_c, test_g, test_one_hot, \
                     test_hist_qp, test_hist_qv, test_hist_tau, test_hist_diff_tau_nom = test_data
        
        train_lstm_input = np.concatenate((tain_hist_qp, tain_hist_qv, tain_hist_diff_tau_nom), axis=-1)
        test_lstm_input = np.concatenate((test_hist_qp, test_hist_qv, test_hist_diff_tau_nom), axis=-1)
        n_enc_input = n_lstm_output

    # add_tau_noise = False
    if add_tau_noise:
        train_tau = train_tau + np.random.normal(0, 0.5, train_tau.shape)

    if add_noise:
        # train_qp = train_qp + np.random.normal(0, 0.15, train_qp.shape)
        # train_qv = train_qv + np.random.normal(0, 0.20, train_qv.shape)
        # train_qa = train_qa + np.random.normal(0, 4, train_qa.shape)
        # train_tau = train_tau + np.random.normal(0, 3, train_tau.shape)
        train_qp = train_qp + np.random.normal(0, 0.1, train_qp.shape)
        train_qv = train_qv + np.random.normal(0, 0.10, train_qv.shape)
        train_qa = train_qa + np.random.normal(0, 2, train_qa.shape)
        train_tau = train_tau + np.random.normal(0, 1, train_tau.shape)

    print("\n\n################################################")
    print("Characters:")
    print("   Test Characters = {0}".format(test_labels))
    print("  Train Characters = {0}".format(train_labels))
    print("# Training Samples = {0:05d}".format(int(train_qp.shape[0])))
    print("")

    # Training Parameters:
    print("\n################################################")
    print("Training Deep Lagrangian Networks (DeLaN):")

    # Construct Hyperparameters:
    hyper = {
             'n_width_inertia': 32,
             'n_depth_inertia': 2,
             'n_width_pot': 32,
             'n_depth_pot': 2,
             'diagonal_epsilon': 0.1,
             'activation': 'Tanh',
             'net_arch_inertia': [30, 20],
             'net_arch_pot': [30, 20],
             'b_init': 1.e-4,
             'b_diag_init': 0.001,
             'w_init': 'xavier_normal',
             'gain_hidden': np.sqrt(2.),
             'gain_output': 0.1,
             'n_minibatch': minibatch,
             'learning_rate': 5.e-04,
             'lr_scheduler': 'No',
             'lr_final': 5.e-05,
             'weight_decay': 1.e-4,
             'init_tf': True,
             'n_enc_input': n_enc_input,
             'embedding_dim': embedding_dim,
             'shared_embedding_layer_flag': shared_embedding_layer,
             'n_lstm_hidden': n_lstm_hidden,
             'n_lstm_input': n_lstm_input,
             'n_lstm_depth': n_lstm_depth,
             'hist_length': hist_length,
             'act_ld': 'Softplus',
             'max_epoch': 10
            }

    model_name = 'epochs_' + str(hyper['max_epoch'])
    
    if hyper['net_arch_inertia'] is None:
        model_name += '_nw_inertia_' + str(hyper['n_width_inertia']) + '_nd_inertia_' + str(hyper['n_depth_inertia'])
    else:
        model_name += '_nw_inertia_' + '_'.join(str(x) for x in hyper['net_arch_inertia'])

    if hyper['net_arch_pot'] is None:
        model_name += '_nw_pot_' + str(hyper['n_width_pot']) + '_nd_pot_' + str(hyper['n_depth_pot'])
    else:
        model_name += '_nw_pot_' + '_'.join(str(x) for x in hyper['net_arch_pot'])

    model_name += '_pin'
    if add_noise_to_load_data:
        model_name += '_noise_'
    model_name += '_mb_' + str(minibatch) + '_norm_tau_' + str(int(flag_normalize_tau)) + '_'
    if add_noise:
        model_name += '_noise2_'
    model_name += filename_short + '.torch'
    if nn_id == "ContextAware":
        model_name = 'hist_' + str(hist_length) + '_lstm_in_' + str(n_lstm_input) + '_h_' + str(n_lstm_hidden) + '_out_' + str(n_lstm_output) + '_d_' + str(n_lstm_depth) + '_act_ld_' + str(hyper['act_ld']) + '_' + model_name
        
        if hyper['lr_scheduler'] is not None:
            model_name = 'lr_sch_' + hyper['lr_scheduler'] + '_' + model_name
    else:
        model_name = nn_id + '_' + model_name

    if n_enc_input > 1 and nn_id != "ContextAware":
        model_name = str(n_enc_input) + '_envs' + '_embed_' + str(embedding_dim) + '_' + model_name


    # Load existing model parameters:
    if load_model:
        # model_name = 'hist_10_input_21_hidden_10_output_10_depth_5_epochs_2000_n_width_32_n_depth_2_sample_off_0_minibatch_512_loss_pwr_0_dataset_use_1.0_norm_tau_1_panda_box_mass_disc_rand_envs_5_box_pos_rand_0_box_mass_rand_0_samples_200000_sim_time_20_dataset_lqr.torch'
        # model_name = 'hist_4_input_21_hidden_5_output_5_depth_5_epochs_100_n_width_32_n_depth_2_sample_off_0_minibatch_1024_loss_pwr_0_dataset_use_1.0_norm_tau_1_panda_box_mass_disc_rand_envs_5_box_pos_rand_0_box_mass_rand_0_samples_200000_sim_time_20_dataset_lqr.torch'
        load_file = f"data/trained_models/{model_type_folder}/{model_name}"
        state = torch.load(load_file)

        delan_model = nn_type(n_dof, **state['hyper'])
        delan_model.load_state_dict(state['state_dict'])
        delan_model = delan_model.cuda() if cuda else delan_model.cpu()

    else:
        # Construct DeLaN:
        delan_model = nn_type(n_dof, **hyper)
        delan_model = delan_model.cuda() if cuda else delan_model.cpu()

    # Generate & Initialize the Optimizer:
    optimizer = torch.optim.Adam(delan_model.parameters(),
                                 lr=hyper["learning_rate"],
                                 weight_decay=hyper["weight_decay"],
                                 amsgrad=False)

    # Generate Replay Memory:
    if not full_model:
        lstm_input_shape = (hist_length, n_lstm_input, ) if hist_length > 0 else (1, )
        mem_dim = ((n_dof, ), (n_dof, ), (n_dof, ), (n_dof, ), (test_one_hot.shape[-1], ), lstm_input_shape)
        mem = PyTorchReplayMemory(train_qp.shape[0], hyper["n_minibatch"], mem_dim, cuda)
        mem.add_samples([train_qp, train_qv, train_qa, train_tau, train_one_hot, train_lstm_input])

    else:
        train_one_hot = train_one_hot.reshape(train_qp.shape[0], -1)
        train_lstm_input = train_lstm_input.reshape(train_qp.shape[0], -1)
        mem_dim = ((n_dof, ), (n_dof, ), (n_dof, ), (n_dof, ), (train_one_hot.shape[-1], ), (train_lstm_input.shape[-1], ))
        mem = PyTorchReplayMemory(train_qp.shape[0], hyper["n_minibatch"], mem_dim, cuda)
        mem.add_samples([train_qp, train_qv, train_qa, train_tau, train_one_hot, train_lstm_input])

    # Start Training Loop:
    t0_start = time.perf_counter()

    if flag_normalize_tau:
        norm_tau = torch.from_numpy(np.var(train_tau,axis=0))
    else:
        norm_tau = torch.ones(n_dof)
    norm_tau = norm_tau.cuda() if cuda else norm_tau.cpu()
    norm_tau_np = norm_tau.detach().cpu().numpy()

    init_param_dict = delan_model.state_dict()
    # sum_param = 0.0
    # for key in init_param_dict.keys():
    #     sum_param += init_param_dict[key].reshape(-1).shape[0]
    # print(f'Number of parameters {sum_param}')
    sum_param = 0.0
    lstm_param = 0.0
    for key in init_param_dict.keys():
        sum_param += init_param_dict[key].reshape(-1).shape[0]
        if 'lstm' in key:
            lstm_param += init_param_dict[key].reshape(-1).shape[0]
    print(f'Number of parameters {sum_param} | LSTM {lstm_param} | MLPs {sum_param - lstm_param}')


    if hyper['lr_scheduler'] == 'Exp':
        lr_final = hyper['lr_final']
        lr_initial = hyper["learning_rate"]
        lr_gamma = (lr_final / lr_initial) ** (1 / hyper['max_epoch'])
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    else:
        lr_scheduler = None

    epoch_i = 0
    t_avg_epoch = 0.0
    while epoch_i < hyper['max_epoch'] and not load_model:
        l_mem_mean_inv_dyn, l_mem_var_inv_dyn = 0.0, 0.0
        l_mem_mean_dEdt, l_mem_var_dEdt = 0.0, 0.0
        l_mem, n_batches = 0.0, 0.0
        t0_epoch = time.perf_counter()

        if save_checkpoint_model:
            if epoch_i > 0 and (epoch_i % 500) == 0:
                print(f'Saving checkpoint model epoch: {epoch_i}')
                torch.save({"epoch": epoch_i,
                            "hyper": hyper,
                            "state_dict": delan_model.state_dict()},
                            f"data/trained_models/{model_type_folder}/checkpoint/{model_name}_{epoch_i}") 

        for q, qd, qdd, tau, enc_input, lstm_input in mem:
            t0_batch = time.perf_counter()

            # Reset gradients:
            optimizer.zero_grad()

            if embedding_dim > 1:
                enc_input = enc_input.long()

            # Compute the Rigid Body Dynamics Model:
            if hist_length == 0:
                if n_enc_input == 1:
                    tau_hat, dEdt_hat = delan_model(q, qd, qdd)
                else:
                    tau_hat, dEdt_hat = delan_model(q, qd, qdd, enc_input)
            else:
                tau_hat, dEdt_hat = delan_model(q, qd, qdd, lstm_input)

            # Compute the loss of the Euler-Lagrange Differential Equation:
            err_inv = torch.sum((tau_hat - tau) ** 2 / norm_tau, dim=1)
            l_mean_inv_dyn = torch.mean(err_inv)
            l_var_inv_dyn = torch.var(err_inv)

            # Compute the loss of the Power Conservation:
            dEdt = torch.matmul(qd.view(-1, n_dof, 1).transpose(dim0=1, dim1=2), tau.view(-1, n_dof, 1)).view(-1)
            err_dEdt = (dEdt_hat - dEdt) ** 2
            l_mean_dEdt = torch.mean(err_dEdt)
            l_var_dEdt = torch.var(err_dEdt)

            # Compute gradients & update the weights:
            if loss_power:
                loss = l_mean_inv_dyn + l_mean_dEdt
            else:
                loss = l_mean_inv_dyn
            loss.backward()
            optimizer.step()
            if hasattr(delan_model, 'shared_embedding_layer'):
                if delan_model.shared_embedding_layer:
                    delan_model.shared_embedding_layer.normalize_weights()

            # Update internal data:
            n_batches += 1
            l_mem += loss.item()
            l_mem_mean_inv_dyn += l_mean_inv_dyn.item()
            l_mem_var_inv_dyn += l_var_inv_dyn.item()
            l_mem_mean_dEdt += l_mean_dEdt.item()
            l_mem_var_dEdt += l_var_dEdt.item()

            t_batch = time.perf_counter() - t0_batch

        # Update Epoch Loss & Computation Time:
        l_mem_mean_inv_dyn /= float(n_batches)
        l_mem_var_inv_dyn /= float(n_batches)
        l_mem_mean_dEdt /= float(n_batches)
        l_mem_var_dEdt /= float(n_batches)
        l_mem /= float(n_batches)
        epoch_i += 1

        if lr_scheduler is not None:
            lr_scheduler.step()

        t_epoch = time.perf_counter() - t0_epoch
        t_avg_epoch = t_avg_epoch + t_epoch

        if epoch_i == 1 or np.mod(epoch_i, log_period) == 0:
            print("Epoch {0:05d}: ".format(epoch_i), end=" ")
            print("Time = {0:05.1f}s".format(time.perf_counter() - t0_start), end=", ")
            print("Time/Epoch = {0:05.4f}s".format(t_avg_epoch / log_period), end=", ")
            print("Loss = {0:.3e}".format(l_mem), end=", ")
            print("Inv Dyn = {0:.3e} \u00B1 {1:.3e}".format(l_mem_mean_inv_dyn, 1.96 * np.sqrt(l_mem_var_inv_dyn)), end=", ")
            print("Power Con = {0:.3e} \u00B1 {1:.3e}".format(l_mem_mean_dEdt, 1.96 * np.sqrt(l_mem_var_dEdt)))
            t_avg_epoch = 0.0

    # Save the Model:
    if save_model and not load_model:
        print(f'Saving model: {model_name}')
        torch.save({"epoch": epoch_i,
                    "hyper": hyper,
                    "state_dict": delan_model.state_dict()},
                    f"data/trained_models/{model_type_folder}/{model_name}")

    print("\n################################################")
    print("Evaluating DeLaN:")

    # Compute the inertial, centrifugal & gravitational torque using batched samples
    t0_batch = time.perf_counter()

    debug_test = False
    if debug_test:
        zeros_np = np.zeros((7,1))
        ones_np = np.ones((7,1))
        # q_test = ones_np
        # qd_test = ones_np
        # qdd_test = zeros_np

        q_test = np.array([[1.5], [-0.2], [-0.5], [1.8], [2.0], [0.4], [1.0]])
        qd_test = np.array([[0.3], [0.4], [-0.35], [2.8], [0.5], [-1.4], [1.2]])
        qdd_test = np.array([[-0.2], [1.2], [1.5], [-1.2], [0.0], [-0.2], [-0.2]])

        # q_test = np.array([[0.1], [0.2], [0.3], [0.4], [-0.7], [0.1], [-2.5]])
        # q = np.array([[0.1, 0.2, 0.3, 0.4, -0.7, 0.1, -2.5]])

        # q_test = np.array([[2.2], [1.45]])
        # qd_test = np.array([[-0.7], [0.1]])
        # qdd_test = np.array([[0.5], [-0.25]])
        # q_test = np.array([[0.4], [-0.87]])
        # qd_test = np.array([[1.2], [-0.7]])
        # qdd_test = np.array([[5], [-7]])
        q_test = torch.from_numpy(q_test.T).float().to(delan_model.device)
        qd_test = torch.from_numpy(qd_test.T).float().to(delan_model.device)
        qdd_test = torch.from_numpy(qdd_test.T).float().to(delan_model.device)
        enc_test =  torch.from_numpy(np.array([0,1,0]).reshape((1,3))).float().to(delan_model.device)

        if hist_length == 0:
            if n_enc_input == 1:
                tau_test = delan_model.inv_dyn(q_test, qd_test, qdd_test).detach().cpu().numpy().squeeze()
            else:
                tau_test = delan_model.inv_dyn(q_test, qd_test, qdd_test, enc_test).detach().cpu().numpy().squeeze()
        else:
            lstm_test = np.ones((hist_length, n_lstm_input))
            lstm_test = torch.from_numpy(lstm_test).float().to(delan_model.device).view(1, hist_length, -1)
            tau_test = delan_model.inv_dyn(q_test, qd_test, qdd_test, lstm_test).detach().cpu().numpy().squeeze()

        print(f'debug tau {tau_test}')

    # Convert NumPy samples to torch:
    q = torch.from_numpy(test_qp).float().to(delan_model.device)
    qd = torch.from_numpy(test_qv).float().to(delan_model.device)
    qdd = torch.from_numpy(test_qa).float().to(delan_model.device)
    enc_input = torch.from_numpy(test_one_hot).float().to(delan_model.device)
    lstm_input = torch.from_numpy(test_lstm_input).float().to(delan_model.device)
    zeros = torch.zeros_like(q).float().to(delan_model.device)

    # Compute the torque decomposition:
    with torch.no_grad():
        if hist_length == 0:
            if n_enc_input == 1:
                delan_g = delan_model.inv_dyn(q, zeros, zeros).cpu().numpy().squeeze()
                delan_c = delan_model.inv_dyn(q, qd, zeros).cpu().numpy().squeeze() - delan_g
                delan_m = delan_model.inv_dyn(q, zeros, qdd).cpu().numpy().squeeze() - delan_g
            else:
                delan_g = delan_model.inv_dyn(q, zeros, zeros, enc_input).cpu().numpy().squeeze()
                delan_c = delan_model.inv_dyn(q, qd, zeros, enc_input).cpu().numpy().squeeze() - delan_g
                delan_m = delan_model.inv_dyn(q, zeros, qdd, enc_input).cpu().numpy().squeeze() - delan_g
            delan_output = delan_model(q, qd, qdd, enc_input)
        else:
            delan_g = delan_model.inv_dyn(q, zeros, zeros, lstm_input).cpu().numpy().squeeze()
            delan_c = delan_model.inv_dyn(q, qd, zeros, lstm_input).cpu().numpy().squeeze() - delan_g
            delan_m = delan_model.inv_dyn(q, zeros, qdd, lstm_input).cpu().numpy().squeeze() - delan_g
            delan_output = delan_model(q, qd, qdd, lstm_input)
        delan_tau = delan_output[0].cpu().numpy()
        delan_dEdt = delan_output[1].cpu().numpy()
    t_batch = (time.perf_counter() - t0_batch) / (3. * float(test_qp.shape[0]))

    # Move model to the CPU:
    delan_model.cpu()

    # Compute the joint torque using single samples on the CPU. The results is done using only single samples to
    # imitate the online control-loop. These online computation are performed on the CPU as this is faster for single
    # samples.

    # delan_tau, delan_dEdt = np.zeros(test_qp.shape), np.zeros((test_qp.shape[0], 1))
    # t0_evaluation = time.perf_counter()
    # for i in range(test_qp.shape[0]):

    #     with torch.no_grad():

    #         # Convert NumPy samples to torch:
    #         q = torch.from_numpy(test_qp[i]).float().view(1, -1)
    #         qd = torch.from_numpy(test_qv[i]).float().view(1, -1)
    #         qdd = torch.from_numpy(test_qa[i]).float().view(1, -1)
    #         enc_input = torch.from_numpy(test_one_hot[i]).float().view(1, -1)

    #         # Compute predicted torque:
    #         if hist_length == 0:
    #             if n_enc_input == 1:
    #                 out = delan_model(q, qd, qdd)
    #             else:
    #                 out = delan_model(q, qd, qdd, enc_input)
    #         else:
    #             lstm_input = torch.from_numpy(test_lstm_input[i]).float().view(1, hist_length, -1)
    #             out = delan_model(q, qd, qdd, lstm_input)
    #         delan_tau[i] = out[0].cpu().numpy().squeeze()
    #         delan_dEdt[i] = out[1].cpu().numpy()

    # t_eval = (time.perf_counter() - t0_evaluation) / float(test_qp.shape[0])

    # Compute Errors:
    test_dEdt = np.sum(test_tau * test_qv, axis=1).reshape((-1, 1))
    err_g = 1. / float(test_qp.shape[0]) * np.sum((delan_g - test_g) ** 2 / norm_tau_np)
    err_m = 1. / float(test_qp.shape[0]) * np.sum((delan_m - test_m) ** 2 / norm_tau_np)
    err_c = 1. / float(test_qp.shape[0]) * np.sum((delan_c - test_c) ** 2 / norm_tau_np)
    err_cg = 1. / float(test_qp.shape[0]) * np.sum((delan_c + delan_g - test_c - test_g) ** 2 / norm_tau_np)
    err_tau = 1. / float(test_qp.shape[0]) * np.sum((delan_tau - test_tau) ** 2 / norm_tau_np)
    err_dEdt = 1. / float(test_qp.shape[0]) * np.sum((delan_dEdt - test_dEdt) ** 2 / norm_tau_np)

    print("\nPerformance:")
    print("                Torque MSE = {0:.3e}".format(err_tau))
    print("              Inertial MSE = {0:.3e}".format(err_m))
    print("Coriolis & Centrifugal MSE = {0:.3e}".format(err_c))
    print("         Gravitational MSE = {0:.3e}".format(err_g))
    print("      Cor & Cen & Grav MSE = {0:.3e}".format(err_cg))
    print("    Power Conservation MSE = {0:.3e}".format(err_dEdt))
    print("      Comp Time per Sample = {0:.3e}s / {1:.1f}Hz".format(t_batch, 1./t_batch))

    print("\n################################################")
    print("Plotting Performance:")

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

    plt.rc('text', usetex=True)
    color_i = ["r", "b", "g", "k"]

    ticks = np.array(divider)
    ticks = (ticks[:-1] + ticks[1:]) / 2

    for i in range(0,n_dof, 2):

        fig = plt.figure(figsize=(24.0/1.54, 8.0/1.54), dpi=100)
        fig.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=0.95, wspace=0.3, hspace=0.2)
        # fig.canvas.manage.set_window_title('Seed = {0}'.format(seed))

        legend = [mp.patches.Patch(color=color_i[0], label="DeLaN"),
                mp.patches.Patch(color="k", label="Ground Truth")]

        # Plot Torque
        ax0 = fig.add_subplot(2, 4, 1)
        # ax0.set_title(r"$\boldsymbol{\tau}$")
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

        ax1.text(s=r"\textbf{(a)}", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
                verticalalignment="center", transform=ax1.transAxes)

        ax1.set_ylabel("Torque [Nm]")
        ax1.get_yaxis().set_label_coords(-0.2, 0.5)
        ax1.set_ylim(y_t_low[i+1], y_t_max[1])
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(test_labels)
        ax1.vlines(divider, y_t_low[i+1], y_t_max[1], linestyles='--', lw=0.5, alpha=1.)
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
        ax0.set_title(r"$\displaystyle\mathbf{H}(\mathbf{q}) \ddot{\mathbf{q}}$")
        ax0.set_ylabel("Torque [Nm]")
        ax0.set_ylim(y_m_low[i+0], y_m_max[i+0])
        ax0.set_xticks(ticks)
        ax0.set_xticklabels(test_labels)
        ax0.vlines(divider, y_m_low[i+0], y_m_max[i+0], linestyles='--', lw=0.5, alpha=1.)
        ax0.set_xlim(divider[0], divider[-1])

        ax1 = fig.add_subplot(2, 4, 6)
        ax1.text(s=r"\textbf{(b)}", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
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
        ax0.set_title(r"$\displaystyle\mathbf{c}(\mathbf{q}, \dot{\mathbf{q}}) + \displaystyle\mathbf{g}(\mathbf{q})$")
        ax0.set_ylabel("Torque [Nm]")
        ax0.set_ylim(y_c_low[i+0], y_c_max[i+0])
        ax0.set_xticks(ticks)
        ax0.set_xticklabels(test_labels)
        ax0.vlines(divider, y_c_low[i+0], y_c_max[i+0], linestyles='--', lw=0.5, alpha=1.)
        ax0.set_xlim(divider[0], divider[-1])

        ax1 = fig.add_subplot(2, 4, 7)
        ax1.text(s=r"\textbf{(c)}", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
                verticalalignment="center", transform=ax1.transAxes)

        ax1.set_ylabel("Torque [Nm]")
        ax1.set_ylim(y_c_low[i+1], y_c_max[i+1])
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(test_labels)
        ax1.vlines(divider, y_c_low[i+1], y_c_max[i+1], linestyles='--', lw=0.5, alpha=1.)
        ax1.set_xlim(divider[0], divider[-1])

        # Plot Ground Truth Coriolis & Centrifugal Torque:
        ax0.plot(test_c[:, i+0] + 0*test_g[:, i+0], color="k")
        if i+1 < n_dof:
            ax1.plot(test_c[:, i+1] + 0*test_g[:, i+1], color="k")

        # Plot DeLaN Coriolis & Centrifugal Torque:
        ax0.plot(delan_c[:, i+0] + delan_g[:, i+0], color=color_i[0], alpha=plot_alpha)
        if i+1 < n_dof:
            ax1.plot(delan_c[:, i+1] + delan_g[:, i+1], color=color_i[0], alpha=plot_alpha)

        # Plot Gravity
        ax0 = fig.add_subplot(2, 4, 4)
        ax0.set_title(r"$\displaystyle\mathbf{g}(\mathbf{q})$")
        ax0.set_ylabel("Torque [Nm]")
        ax0.set_ylim(y_g_low[i+0], y_g_max[i+0])
        ax0.set_xticks(ticks)
        ax0.set_xticklabels(test_labels)
        ax0.vlines(divider, y_g_low[i+0], y_g_max[i+0], linestyles='--', lw=0.5, alpha=1.)
        ax0.set_xlim(divider[0], divider[-1])

        ax1 = fig.add_subplot(2, 4, 8)
        ax1.text(s=r"\textbf{(d)}", x=.5, y=-0.25, fontsize=12, fontweight="bold", horizontalalignment="center",
                verticalalignment="center", transform=ax1.transAxes)

        ax1.set_ylabel("Torque [Nm]")
        ax1.set_ylim(y_g_low[i+1], y_g_max[i+1])
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(test_labels)
        ax1.vlines(divider, y_g_low[i+1], y_g_max[i+1], linestyles='--', lw=0.5, alpha=1.)
        ax1.set_xlim(divider[0], divider[-1])

        # Plot Ground Truth Gravity Torque:
        ax0.plot(-1.0*test_g[:, i+0], color="k")
        if i+1 < n_dof:
            ax1.plot(-1.0*test_g[:, i+1], color="k")

        # Plot DeLaN Gravity Torque:
        ax0.plot(delan_g[:, i+0], color=color_i[0], alpha=plot_alpha)
        if i+1 < n_dof:
            ax1.plot(delan_g[:, i+1], color=color_i[0], alpha=plot_alpha)

        fig_dir = f"figures/mpc_DeLaN_Performance/{model_type_folder}/{model_name}"
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)
        fig.savefig(f"{fig_dir}/joints_{i}_{i+1}.pdf", format="pdf")
        fig.savefig(f"{fig_dir}/joints_{i}_{i+1}.png", format="png")

    if render:
        plt.show()

    print("\n################################################\n\n\n")

