import torch
import numpy as np
import pickle
import time

from cadelac.control.panda_sim import build_models
from cadelac.control.learned_dynamics.new_double_DeLaN_model import L4CDoubleDeLaN
from cadelac.control.main_mpc import PandaMPCSim
from cadelac.control.casadi_model import RealtimeApprox

import os
from pathlib import Path

def compute_rms(error):
    return np.sqrt(np.mean(np.square(error), axis=0))

if __name__ == "__main__":

    # Model Parameteres
    n_dof = 7
    use_delan = False
    kf_filter = True
    hist_length = 15
    n_lstm_output = 10
    n_enc_input = n_lstm_output
    nx = 2 * n_dof
    RTI_mode = True

    name_suffix = '_kf' if kf_filter else '_nominal'

    # Evaluation parameters
    n_train_envs = 100
    n_eval_envs = 30
    n_runs = 20
    Tsim_run = 10
    # n_runs = 2

    n_rand_envs = n_train_envs + n_eval_envs
    box_pos = None
    box_inertia_flag = True
    box_mass = None
    seed = 0

    ## Build xml handles
    sim_xml_handles, xml_handles, _ = build_models(num_rand_envs=n_rand_envs,
                                                box_pos=box_pos,
                                                box_inertia_flag=box_inertia_flag,
                                                box_mass=box_mass,
                                                collision=False,
                                                seed=seed)
    
    ## Load Delan
    CONTROL_DIR = str(Path(__file__).resolve().parents[0])
    model_folder = CONTROL_DIR + '/learned_dynamics/models/res_model/'
    filename = 'epochs_3000_nw_inertia_30_20_nw_pot_30_20_pin_noise_rand_envs_nom_101_kf_0_samples_1040300.torch'
    inference_suffix = '_delan_v4_envs_101_noise_kf_0_new_QR_repeat'

    load_file = model_folder + filename
    print(f'Loading file {load_file}')

    torch_model = torch.load(load_file, map_location=torch.device('cpu'), weights_only=False)
    if 'n_width' not in torch_model['hyper'].keys():
        torch_model['hyper']['n_width'] = torch_model['hyper']['n_width_inertia']
        torch_model['hyper']['n_depth'] = torch_model['hyper']['n_depth_inertia']
    l4c_delan_inference = L4CDoubleDeLaN(torch_model, n_dof=n_dof, n_enc_input=n_enc_input,
                                fwd_embedding=False, device='cpu')



    if use_delan:
        l4c_delan = l4c_delan_inference
        name_suffix = inference_suffix
    elif kf_filter:
        l4c_delan = 'KF'
    else:
        l4c_delan = None

    ## Init MPC
    panda_mpc = PandaMPCSim(
                        delan_model=l4c_delan,
                        use_viewer=False,
                        expl_dyn=False if not RTI_mode else True,
                        realtime_approx=RealtimeApprox.NO_APPROX,
                        RTI_mode=RTI_mode,
                        name_suffix=name_suffix,
                        ref_type='EXC',
                        hist_length = hist_length,
                        delan_model_inference=l4c_delan_inference,
                        inference_suffix=inference_suffix,
                        sim_xml_handles=sim_xml_handles,
                        xml_handles=xml_handles
                        )
    

    Nsim = int(Tsim_run / panda_mpc.joint_ctrl_period)
    N_horizon = panda_mpc.mpc.N_horizon


    results = {'env_id': [],
               'q': [], 'qd': [], 'qdd': [], 'q_ref': [], 'qd_ref': [], 'enc_input': [],
               'xpred_next': [], 'xpred_error': [], 'xpred_horizon': [], 'u_horizon': [],
               'rms_q_track_error': [], 'rms_qd_track_error': [], 
               'avg_step_time': [], 'avg_solver_time': [], 'avg_param_time': [],
               'rms_q_pred_error': [], 'rms_qd_pred_error': [],
               'rms_q_pred_error_horizon': [], 'rms_qd_pred_error_horizon': [],
               'tau': [], 'tau_nom': [],
               'diff_tau_nom': [], 'tau_nom_pin': [],
               'diff_tau_nom_pin': [], 'tau_mpc': [], 'diff_tau_mpc': [],
               'avg_ref_time': [], 'avg_param_time': [], 'avg_acados_time_tot': [],
               'avg_acados_time_lin': [], 'avg_acados_time_sim': [], 'avg_acados_time_qp': [],
               'avg_acados_time_reg': [], 'avg_acados_time_sim_ad': [], 'avg_acados_time_sim_la': [],
               'avg_acados_time_solution_sens_lin': [],
               'avg_mpc_cost': [],
               'rms_nom_tau_error': [], 'rms_kf_tau_error': [], 'rms_delan_tau_error': [],
               'rate_rms_kf_nom_tau_error': [], 'rate_rms_delan_nom_tau_error': []
    }
    # Simulate
    print('Start Simulation')

    #Tracking error q
    #Tracking error qd
    #Pred error q
    #Pred error qd
    # Avg - Step time
    # Avg Total time solver
    # Avg Time QP
    # Avg model

    for eval_env in range(n_train_envs, n_rand_envs):
        
        print(f'\n\n### Env {eval_env} ###')
        # Runs in each environment
        for run in range(n_runs):

            panda_mpc.reset(env_id = eval_env)
            
            init_time = time.time()
            for i in range(Nsim):
                panda_mpc.step()
            end_time = time.time()
            
            panda_mpc.logger.convert_log_data_to_np()

            # Track error
            rms_q_track_error = np.sqrt(np.mean(np.square(panda_mpc.logger.logged_data['qp_ref']-panda_mpc.logger.logged_data['qp']),axis=0))
            rms_qd_track_error = np.sqrt(np.mean(np.square(panda_mpc.logger.logged_data['qv_ref']-panda_mpc.logger.logged_data['qv']),axis=0))

            if panda_mpc.ref_type == 'Default':
                print(f'q_sin_ref_amp {panda_mpc.q_sin_ref_amp.T} | q_sin_ref_freq {panda_mpc.q_sin_ref_freq.T}')

            results['rms_q_track_error'].append(rms_q_track_error)
            results['rms_qd_track_error'].append(rms_qd_track_error)

            # Pred error - 1st step
            rms_x_pred_error = np.sqrt(np.mean(np.square(panda_mpc.logger.logged_data['xpred_error']), axis=0))
            rms_qd_pred_error = rms_x_pred_error[:n_dof]
            rms_q_pred_error = rms_x_pred_error[n_dof:]

            results['rms_q_pred_error'].append(rms_q_pred_error)
            results['rms_qd_pred_error'].append(rms_qd_pred_error)

            # Get prediction error at n horizon
            def get_q_pred_errror_n_horizon(n_horizon):
                index = (1+n_horizon)*nx
                q_pred_n = panda_mpc.logger.logged_data['xpred_horizon'][:-(1+n_horizon),index+n_dof:index+nx]
                q_real_n = panda_mpc.logger.logged_data['qp'][(1+n_horizon):]
                return np.sqrt(np.mean(np.square(q_real_n - q_pred_n), axis=0))

            def get_qd_pred_errror_n_horizon(n_horizon):
                index = (1+n_horizon)*nx
                qd_pred_n = panda_mpc.logger.logged_data['xpred_horizon'][:-(1+n_horizon),index:index+n_dof]
                qd_real_n = panda_mpc.logger.logged_data['qv'][(1+n_horizon):]
                return np.sqrt(np.mean(np.square(qd_real_n - qd_pred_n), axis=0))
            
            ## Compute Torque errors
            nom_tau_error = panda_mpc.logger.logged_data['tau'] - panda_mpc.logger.logged_data['tau_nom_pin']
            kf_tau_error = nom_tau_error + panda_mpc.logger.logged_data['tau_kf'] #Positiv sign because of external force convention
            delan_tau_error = nom_tau_error - panda_mpc.logger.logged_data['tau_delan'].squeeze() #Positiv sign because of external force convention

            rms_nom_tau_error = compute_rms(nom_tau_error)
            rms_kf_tau_error = compute_rms(kf_tau_error)
            rms_delan_tau_error = compute_rms(delan_tau_error)

            rate_rms_kf_nom_tau_error = 100 * rms_kf_tau_error / rms_nom_tau_error
            rate_rms_delan_nom_tau_error = 100 * rms_delan_tau_error / rms_nom_tau_error

            results['rms_nom_tau_error'].append(rms_nom_tau_error)
            results['rms_kf_tau_error'].append(rms_kf_tau_error)
            results['rms_delan_tau_error'].append(rms_delan_tau_error)
            results['rate_rms_kf_nom_tau_error'].append(rate_rms_kf_nom_tau_error)
            results['rate_rms_delan_nom_tau_error'].append(rate_rms_delan_nom_tau_error)
    
            rms_q_pred_error_horizon = []

            results['env_id'].append(panda_mpc.env_id)
            results['avg_step_time'].append((end_time - init_time)/Nsim)
            results['avg_solver_time'].append(panda_mpc.time_solver/Nsim)
            results['avg_ref_time'].append(panda_mpc.time_ref/Nsim)
            results['avg_param_time'].append(panda_mpc.time_param/Nsim)

            results['avg_mpc_cost'].append(np.mean(panda_mpc.logger.logged_data['mpc_cost']))

            results['avg_acados_time_tot'].append(np.mean(panda_mpc.logger.logged_data['time_tot']))
            results['avg_acados_time_lin'].append(np.mean(panda_mpc.logger.logged_data['time_lin']))
            results['avg_acados_time_sim'].append(np.mean(panda_mpc.logger.logged_data['time_sim']))
            results['avg_acados_time_qp'].append(np.mean(panda_mpc.logger.logged_data['time_qp']))
            results['avg_acados_time_reg'].append(np.mean(panda_mpc.logger.logged_data['time_reg']))
            results['avg_acados_time_sim_ad'].append(np.mean(panda_mpc.logger.logged_data['time_sim_ad']))
            results['avg_acados_time_sim_la'].append(np.mean(panda_mpc.logger.logged_data['time_sim_la']))
            results['avg_acados_time_solution_sens_lin'].append(np.mean(panda_mpc.logger.logged_data['time_solution_sens_lin']))


    folder_name = 'cadelac/sim_results/'
    results_dict_name = str(n_runs) + '_runs'
    if use_delan:
        results_dict_name += '_cadelac_' + str(hist_length) + name_suffix
    elif kf_filter:
        results_dict_name += '_kf_mpc' + name_suffix
    else:
        results_dict_name += '_nominal_mpc' + name_suffix
    results_dict_name += '_n_envs_' + str(n_eval_envs) + '_n_runs_' + str(n_runs)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    with open(folder_name + results_dict_name + '.pkl', 'wb') as fp:
        pickle.dump(results, fp)