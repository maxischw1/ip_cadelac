import argparse
import torch
import numpy as np
import pickle
import time

from cadelac.control.panda_sim import build_models
from cadelac.control.l4c_context_aware_delan import L4CContextAwareDeLaN
from cadelac.control.main_mpc import PandaMPCSim
from cadelac.control.casadi_model import RealtimeApprox

import os
from pathlib import Path

def compute_rms(error):
    return np.sqrt(np.mean(np.square(error), axis=0))

if __name__ == "__main__":

    ctrl_list = ['Nominal', 'EKF', 'CaDeLaC']

    # Read Command Line Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=int, default=0, help="Controller option.")
    args = parser.parse_args()

    ctrl_index = int(args.c)
    ctrl_name = ctrl_list[ctrl_index]

    # Model Parameteres
    n_dof = 7
    hist_length = 15
    n_lstm_output = 10
    n_enc_input = n_lstm_output
    nx = 2 * n_dof
    RTI_mode = True

    # Evaluation parameters
    n_train_envs = 100
    n_eval_envs = 30
    n_runs = 20
    Tsim_run = 10

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
    
    ## Load Context-Aware DeLaN
    CADELAC_DIR = Path(__file__).resolve().parents[1]
    LEARNING_DIR = str(CADELAC_DIR) + "/learning"
    model_folder = LEARNING_DIR + f"/trained_models/res_model/panda/ContextAware/"
    
    # If you want to try your own model, modify both filename and inference_suffix accordingly
    filename = 'iros2025_epochs_3000_panda_mj_101_rand_envs_20_runs_50Hz_lqr.torch'
    inference_suffix = '_cadelac'

    load_file = model_folder + filename
    print(f'Loading file {load_file}')

    torch_model = torch.load(load_file, map_location=torch.device('cpu'), weights_only=False)
    l4c_delan_inference = L4CContextAwareDeLaN(torch_model, n_dof=n_dof, n_enc_input=n_enc_input, device='cpu')

    if ctrl_name == 'CaDeLaC':
        l4c_delan = l4c_delan_inference
        name_suffix = inference_suffix
    elif ctrl_name == 'EKF':
        l4c_delan = 'KF'
        name_suffix = '_ekf'
    else:
        l4c_delan = None
        name_suffix = '_nominal'

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


    # Print Results
    runs_rms_q_track_error = np.array(results['rms_q_track_error'])
    runs_rms_qd_track_error = np.array(results['rms_qd_track_error'])

    runs_rms_q_pred_error = np.array(results['rms_q_pred_error'])
    runs_rms_qd_pred_error = np.array(results['rms_qd_pred_error'])
    np.set_printoptions(precision=3)
    print(f'\n#### Results Controller {ctrl_name} ####')
    print('## Tracking Error')
    print(f'RMS q \n{np.mean(runs_rms_q_track_error, axis=0)}')
    print(f'RMS qd \n{np.mean(runs_rms_qd_track_error, axis=0)}')

    print('\n## Prediction Error')
    print(f'RMS q \n{np.mean(runs_rms_q_pred_error, axis=0)}')
    print(f'RMS qd\n{np.mean(runs_rms_qd_pred_error, axis=0)}')

    RESULTS_DIR = CADELAC_DIR / "sim_results"
    results_dict_name = str(n_runs) + '_runs'
    if ctrl_name == 'CaDeLaC':
        results_dict_name += '_cadelac_' + str(hist_length) + name_suffix
    elif ctrl_name == 'EKF':
        results_dict_name += '_kf_mpc' + name_suffix
    else:
        results_dict_name += '_nominal_mpc' + name_suffix
    results_dict_name += '_n_envs_' + str(n_eval_envs) + '_n_runs_' + str(n_runs)

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    with open(str(RESULTS_DIR) + '/' + results_dict_name + '.pkl', 'wb') as fp:
        pickle.dump(results, fp)