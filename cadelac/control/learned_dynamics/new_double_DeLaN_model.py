import torch
import torch.nn as nn

import numpy as np

import casadi as cs
import l4casadi as l4c


from l4casadi.naive.nn import activation as activations
from l4casadi.naive.nn.linear import Linear as l4c_Linear

from typing import Callable

import time

class ComponentNNNaive(l4c.naive.NaiveL4CasADiModule):
    def __init__(self, n_input, n_ouput, n_width, n_depth, net_arch = None, n_enc_input = 1,
                 embedding_dim = 1, fwd_embedding = True, activation_name = 'Tanh'):
        super(ComponentNNNaive, self).__init__()

        # Read optional arguments:
        self.n_width = n_width
        self.n_depth = n_depth
        self.net_arch = net_arch
        self.n_input = n_input
        self.n_ouput = n_ouput
        self.n_enc_input = n_enc_input
        self.activation_name = activation_name
        self.apply_tf = True
        self.fwd_embedding = fwd_embedding
        self.embedding_dim = embedding_dim
        self.embedding_mat = np.zeros((n_enc_input, embedding_dim))

        self.n_output = n_ouput

        if self.activation_name is None:
            self.act = lambda x: x
        elif type(self.activation_name) is str:
            self.act = getattr(activations, self.activation_name)()
        else:
            self.act = self.activation_name

        ## Create Networks
        self.layers = []
        # Create Input Layer
        input_layer_size = n_input
        if self.apply_tf:
            input_layer_size = 2 * input_layer_size

        if not fwd_embedding:
            if self.embedding_dim > 1:
                input_layer_size += self.embedding_dim
            elif self.n_enc_input > 1:
                input_layer_size += self.n_enc_input
        else:
            input_layer_size += self.embedding_dim

        self.input_layer_size = input_layer_size

        if self.net_arch is None:
            self.layers.append(l4c_Linear(input_layer_size, self.n_width))
            self.layers.append(self.act)

            # Create Hidden Layer:
            for _ in range(1, self.n_depth):
                self.layers.append(l4c_Linear(self.n_width, self.n_width))
                self.layers.append(self.act)

            # Create Output Layer
            self.layers.append(l4c_Linear(self.n_width, self.n_output))

        else:
            prev_size = input_layer_size
            for hidden_size in self.net_arch:
                self.layers.append(l4c_Linear(prev_size, hidden_size))  # Linear layer
                self.layers.append(self.act)  # Activation function
                prev_size = hidden_size  # Update previous layer size
            # Create Output Layer
            self.layers.append(l4c_Linear(prev_size, self.n_output))

        self.net = nn.Sequential(*self.layers)

    def set_embedding_mat(self, new_mat):
        self.embedding_mat = new_mat

    def input_tf(self, input):
        return cs.vertcat(cs.cos(input), cs.sin(input))

    def forward(self, input):
        # input = input
        input_q = input[:self.n_input]
        input_enc = input[self.n_input:]

        if self.apply_tf:
            input_q = self.input_tf(input_q)
        input = input_q

        if self.fwd_embedding:
            input = cs.vertcat(input, input_enc)
        # If not fwd embedding, use one hot encode or embedding matrix
        else:
            # Apply embedding
            if self.embedding_dim > 1:
                # Multiply matrix by one hot encoder
                embedding = self.embedding_mat.T @ input_enc
                input = cs.vertcat(input, embedding)

            # Use one hot encode input
            elif self.n_enc_input > 1:
                input = cs.vertcat(input, input_enc)

        return self.net(input).reshape((1,-1))

class ComponentNN(nn.Module):
    def __init__(self, n_input, n_ouput, n_width, n_depth, net_arch, n_enc_input = 1,
                 embedding_dim = 1, fwd_embedding = True, activation_name = 'Tanh'):
        super(ComponentNN, self).__init__()

        # Read optional arguments:
        self.n_width = n_width
        self.n_depth = n_depth
        self.net_arch = net_arch
        self.n_input = n_input
        self.n_ouput = n_ouput
        self.n_enc_input = n_enc_input
        self.activation_name = activation_name
        self.apply_tf = True
        self.fwd_embedding = fwd_embedding
        self.embedding_dim = embedding_dim
        self.embedding_mat = np.zeros((n_enc_input, embedding_dim))

        self.n_output = n_ouput

        if self.activation_name == 'Tanh':
            self.act = nn.Tanh()
        elif self.activation_name == 'ReLu':
            self.act = nn.ReLU()
        else:
            raise AssertionError

        ## Create Networks
        self.layers = []
        # Create Input Layer
        input_layer_size = n_input
        if self.apply_tf:
            input_layer_size = 2 * input_layer_size

        if not fwd_embedding:
            if self.embedding_dim > 1:
                input_layer_size += self.embedding_dim
            elif self.n_enc_input > 1:
                input_layer_size += self.n_enc_input
        else:
            input_layer_size += self.embedding_dim

        self.input_layer_size = input_layer_size

        if self.net_arch is None:
            self.layers.append(torch.nn.Linear(input_layer_size, self.n_width))
            self.layers.append(self.act)

            # Create Hidden Layer:
            for _ in range(1, self.n_depth):
                self.layers.append(torch.nn.Linear(self.n_width, self.n_width))
                self.layers.append(self.act)

            # Create Output Layer
            self.layers.append(torch.nn.Linear(self.n_width, self.n_output))

        else:
            prev_size = input_layer_size
            for hidden_size in self.net_arch:
                self.layers.append(torch.nn.Linear(prev_size, hidden_size))  # Linear layer
                self.layers.append(self.act)  # Activation function
                prev_size = hidden_size  # Update previous layer size
            # Create Output Layer
            self.layers.append(torch.nn.Linear(prev_size, self.n_output))
        
        self.net = nn.Sequential(*self.layers)
        # self.net.apply(self.init_weights)
    
    def set_embedding_mat(self, new_mat):
        self.embedding_mat = new_mat

    def input_tf(self, input):
        return torch.cat([torch.cos(input), torch.sin(input)], axis=-1)

    def forward(self, input):
        input = input.flatten()
        input_q = input[:self.n_input]
        input_enc = input[self.n_input:]

        if self.apply_tf:
            input_q = self.input_tf(input_q)
        input = input_q

        if self.fwd_embedding:
            input = cs.vertcat(input, input_enc)
        # If not fwd embedding, use one hot encode or embedding matrix
        else:
            # Apply embedding
            if self.embedding_dim > 1:
                embedding = self.embedding_mat[input_enc]
                input = torch.cat((input, embedding), dim=-1)

            # Use one hot encode input
            elif self.n_enc_input > 1:
                input = torch.cat((input, input_enc), dim=-1)

        # Hack to proper compute the jacobians from the aprox solution
        return self.net(input).reshape((1,-1))

    # def forward(self, input_q, input_enc):
    #     # input = input.flatten()
    #     # input_q = input[:self.n_input]
    #     # input_enc = input[self.n_input:]

    #     if self.apply_tf:
    #         input_q = self.input_tf(input_q)
    #     # input = input_q

    #     if self.fwd_embedding:
    #         input = cs.vertcat(input_q, input_enc)
    #     # If not fwd embedding, use one hot encode or embedding matrix
    #     else:
    #         # Apply embedding
    #         if self.embedding_dim > 1:
    #             embedding = self.embedding_mat[input_enc]
    #             input = torch.cat((input_q, embedding), dim=-1)

    #         # Use one hot encode input
    #         elif self.n_enc_input > 1:
    #             input = torch.cat((input_q, input_enc), dim=-1)

    #     # Hack to proper compute the jacobians from the aprox solution
    #     return self.net(input).reshape((1,-1))

    # def forward(self, input_q, input_e):
    #     input = torch.cat((self.input_tf(input_q), input_e), dim=-1)
    #     # Hack to proper compute the jacobians from the aprox solution
    #     return self.net(input).reshape((1,-1))
    
    # def forward(self, input_all):
    #     # Hack to proper compute the jacobians from the aprox solution
    #     return self.net(input_all.squeeze()).reshape((1,-1))

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()

        # Define a stacked LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer to produce the output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Forward propagate through LSTM
        out, (hn, cn) = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Pass through the fully connected layer (output at the last time step)
        out = self.fc(out[:, -1, :])  # Use the last hidden state for output prediction
        return out


class L4CDoubleDeLaN():
    def __init__(self, torch_model, n_dof = 2, n_enc_input = 1,
                 embedding_dim = 1, fwd_embedding = True, device = 'cpu'):
        
        self.n_dof = n_dof
        self.n_enc_input = n_enc_input
        self.embedding_dim = embedding_dim
        self.fwd_embedding = fwd_embedding

        hyper = torch_model['hyper']
        self.n_width = hyper['n_width']
        self.n_depth = hyper['n_depth']
        self.net_arch_inertia = hyper.get('net_arch_inertia', None)
        self.net_arch_pot = hyper.get('net_arch_pot', None)
        self.non_linearity = hyper['activation']
        self.gain_hidden = hyper['gain_hidden']
        self.gain_output = hyper['gain_output']
        self._epsilon = hyper['diagonal_epsilon']
        self.softplus_beta = 1.0
        self.device = device
        self.hist_length = hyper.get('hist_length', 0)

        self.state_dict = torch_model['state_dict']

        # Compute non-zero elements of L:
        self.l_output_size = int((self.n_dof ** 2 + self.n_dof) / 2)
        self.l_diag_size = self.n_dof
        self.l_lower_size = self.l_output_size - self.n_dof

        # Indices for matrix version of l
        self.idx_l_diag = np.diag_indices(self.n_dof)
        self.idx_l_off_diag = np.tril_indices(self.n_dof, -1)

        # Indices for vector version l
        # Calculate the indices of the diagonal elements of L:
        idx_diag = np.arange(self.n_dof) + 1
        idx_diag = idx_diag * (idx_diag + 1) / 2 - 1

        # Calculate the indices of the off-diagonal elements of L:
        idx_tril = np.extract([x not in idx_diag for x in np.arange(self.l_output_size)], np.arange(self.l_output_size))

        # Indexing for concatenation of l_o  and l_d
        cat_idx = np.hstack((idx_diag, idx_tril))
        order = np.argsort(cat_idx)
        self._idx = np.arange(cat_idx.size)[order]

        self.inertia_net = ComponentNN(self.n_dof, self.l_output_size, self.n_width, self.n_depth, self.net_arch_inertia, self.n_enc_input, self.embedding_dim, self.fwd_embedding)
        self.potential_net = ComponentNN(self.n_dof, 1, self.n_width, self.n_depth, self.net_arch_pot, self.n_enc_input, self.embedding_dim, self.fwd_embedding)

        inertia_dict = {}
        potential_dict = {}
        for key, values in self.state_dict.items():
            if 'inertia' in key and 'embedding' not in key:
                inertia_dict.update({key.removeprefix('inertia_net.'): values})
            if 'inertia' in key and 'embedding' in key:
                inertia_embedding_mat = values.detach().cpu().numpy()
            if 'potential' in key and 'embedding' not in key:
                potential_dict.update({key.removeprefix('potential_net.'): values})
            if 'potential' in key and 'embedding' in key:
                potential_embedding_mat = values.detach().cpu().numpy()

        cmp_inertia_dict = self.inertia_net.state_dict()
        for key1, value1 in inertia_dict.items():
            if key1 in cmp_inertia_dict:  # Ensure key1 exists in cmp_inertia_dict
                cmp_inertia_dict[key1] = value1

        cmp_potential_dict = self.potential_net.state_dict()
        for key1, value1 in potential_dict.items():
            if key1 in cmp_potential_dict:  # Ensure key1 exists in cmp_inertia_dict
                cmp_potential_dict[key1] = value1

        self.inertia_net.load_state_dict(cmp_inertia_dict)
        self.potential_net.load_state_dict(cmp_potential_dict)

        # if not self.fwd_embedding:
        if self.embedding_dim > 1:
            self.inertia_net.set_embedding_mat(inertia_embedding_mat)
            self.potential_net.set_embedding_mat(potential_embedding_mat)

        self.l4c_inertia_net = l4c.L4CasADi(self.inertia_net, name='inertia_net', model_expects_batch_dim=False)
        self.l4c_potential_net = l4c.L4CasADi(self.potential_net, name='potential_net', model_expects_batch_dim=False)

        self.l4c_inertia_net_realtime = l4c.realtime.RealTimeL4CasADi(self.inertia_net, name='inertia_net_realtime', approximation_order=1)
        self.l4c_potential_net_realtime = l4c.realtime.RealTimeL4CasADi(self.potential_net, name='potential_net_realtime', approximation_order=1)


        self.l4c_inertia_net_naive_nn = l4c.naive.MultiLayerPerceptron(self.inertia_net.input_layer_size, self.n_width, self.inertia_net.n_ouput, self.n_depth, 'Tanh')
        naive_inertia_dict = self.l4c_inertia_net_naive_nn.state_dict()
        for key1, value1 in inertia_dict.items():
            if key1 in naive_inertia_dict:  # Ensure key1 exists in naive_inertia_dict
                naive_inertia_dict[key1] = value1
        self.l4c_inertia_net_naive_nn.load_state_dict(naive_inertia_dict)


        self.l4c_potential_net_naive_nn = l4c.naive.MultiLayerPerceptron(self.potential_net.input_layer_size, self.n_width, self.potential_net.n_ouput, self.n_depth, 'Tanh')
        naive_potential_dict = self.l4c_potential_net_naive_nn.state_dict()
        for key1, value1 in potential_dict.items():
            if key1 in naive_potential_dict:  # Ensure key1 exists in naive_potential_dict
                naive_potential_dict[key1] = value1
        self.l4c_potential_net_naive_nn.load_state_dict(naive_potential_dict)

        self.l4c_inertia_net_naive = l4c.L4CasADi(self.l4c_inertia_net_naive_nn, name='naive_inertia_net', model_expects_batch_dim=False)
        self.l4c_potential_net_naive = l4c.L4CasADi(self.l4c_potential_net_naive_nn, name='naive_potential_net', model_expects_batch_dim=False)

        # L4C Naive with Componet NN
        self.inertia_net_nn_l4c_naive_comp = ComponentNNNaive(self.n_dof, self.l_output_size, self.n_width, self.n_depth, self.net_arch_inertia, self.n_enc_input, self.embedding_dim, self.fwd_embedding)
        self.inertia_net_nn_l4c_naive_comp.load_state_dict(cmp_inertia_dict)
        # if not self.fwd_embedding and self.embedding_dim > 1:
        if self.embedding_dim > 1:
            self.inertia_net_nn_l4c_naive_comp.set_embedding_mat(inertia_embedding_mat)
        self.inertia_net_naive_comp = l4c.L4CasADi(self.inertia_net_nn_l4c_naive_comp, name='inertia_net_nn_l4c_naive_comp', model_expects_batch_dim=False)

        self.potential_net_nn_l4c_naive_comp = ComponentNNNaive(self.n_dof, 1, self.n_width, self.n_depth, self.net_arch_pot, self.n_enc_input, self.embedding_dim, self.fwd_embedding)
        self.potential_net_nn_l4c_naive_comp.load_state_dict(cmp_potential_dict)
        # if not self.fwd_embedding and self.embedding_dim > 1:
        if self.embedding_dim > 1:
            self.potential_net_nn_l4c_naive_comp.set_embedding_mat(potential_embedding_mat)
        self.potential_net_naive_comp = l4c.L4CasADi(self.potential_net_nn_l4c_naive_comp, name='potential_net_nn_l4c_naive_comp', model_expects_batch_dim=False)

        ## Casadi
        q_eval_delan = cs.MX.sym("q_eval_delan", self.n_dof, 1)
        x_eval_delan = q_eval_delan

        if self.n_enc_input > 1:
            if not self.fwd_embedding:
                if self.embedding_dim > 1:
                    enc_eval_delan = cs.MX.sym("enc", self.n_enc_input, 1)
                elif self.n_enc_input > 1:
                    enc_eval_delan = cs.MX.sym("enc", self.n_enc_input, 1)
            else:
                enc_eval_delan = cs.MX.sym("enc", self.embedding_dim, 1)
            x_eval_delan = cs.vertcat(x_eval_delan, enc_eval_delan)

        self.lower_tri_inertia_eval_fn = cs.Function('lower_mat_naive_fn',
                        [x_eval_delan],
                        [self.lower_tri_inertia_fn_cs(self.inertia_net_naive_comp(x_eval_delan))])
        self.lower_tri_inertia_jac_eval_fn = cs.Function('lower_mat_naive_jac_fn',
                        [x_eval_delan],
                        [cs.jacobian(self.lower_tri_inertia_fn_cs(self.inertia_net_naive_comp(x_eval_delan)), q_eval_delan)])

        self.potential_eval_fn = cs.Function('potential_naive_fn',
                        [x_eval_delan],
                        [self.potential_net_naive_comp(x_eval_delan)])
        self.potential_jac_eval_fn = cs.Function('potential_naive_jac_fn',
                        [x_eval_delan],
                        [cs.jacobian(self.potential_net_naive_comp(x_eval_delan), q_eval_delan)])

        # Define lstm
        if self.hist_length > 0:
            self.n_lstm_input = hyper['n_lstm_input']
            self.n_lstm_hidden = hyper['n_lstm_hidden']
            self.n_lstm_depth = hyper['n_lstm_depth']
            self.lstm = LSTMModel(self.n_lstm_input, self.n_lstm_hidden, self.n_enc_input, self.n_lstm_depth)
            lstm_dict = {}
            for key, values in self.state_dict.items():
                if 'lstm' in key:
                    lstm_dict.update({key.removeprefix('lstm.'): values})
            self.lstm.load_state_dict(lstm_dict)

        if self.device == 'cuda':
            self.inertia_net = self.inertia_net.to('cuda')
            self.potential_net = self.potential_net.to('cuda')

    def lower_tri_inertia_fn(self, q, enc_input):
        output = self.inertia_net(q, enc_input).view(-1)
        l_diagonal, l_off_diagonal = torch.split(output, [self.l_diag_size, self.l_lower_size], dim=-1)

        # Ensure positive diagonal
        l_diagonal = torch.nn.Softplus(self.softplus_beta)(l_diagonal) + self._epsilon

        # Assemble l
        l_vec = torch.cat((l_diagonal, l_off_diagonal), -1)[..., self._idx]

        l = torch.zeros((self.n_dof, self.n_dof)).to(l_vec.device)
        tril_indices = torch.tril_indices(self.n_dof, self.n_dof)
        l = l.index_put((tril_indices[0], tril_indices[1]), l_vec)

        # Returning twice to use as aux varible over the jacobian
        return l, l

    def lower_tri_inertia_fn_no_aux(self, q, enc_input):
        return self.lower_tri_inertia_fn(q, enc_input)[0]

    def SoftPlusCS(self, x, beta = 1.0):
        return 1.0 / beta * cs.log(1 + cs.exp(beta * x))

    def lower_tri_inertia_fn_cs(self, nn_output):

        l_diagonal, l_off_diagonal = cs.vertsplit(nn_output.T, [0, self.l_diag_size, self.l_output_size])

        # Ensure positive diagonal
        l_diagonal = self.SoftPlusCS(l_diagonal) + self._epsilon

        # # Diagonal assignment
        l = cs.diag(l_diagonal)
        for index in range(self.l_lower_size):
            l[self.idx_l_off_diag[0][index], self.idx_l_off_diag[1][index]] = l_off_diagonal[index]

        return l.reshape((-1,1))

    def inertia_nn_out(self, q, enc_input):
        l_raw = self.inertia_net(q, enc_input)
        return l_raw, l_raw

    def potential_nn_out(self, q, enc_input):
        V = self.potential_net(q, enc_input)
        return V, V

    def get_jac_value_inertia_torch(self, q_np, enc_input_np):

        q = torch.from_numpy(q_np).float()
        if self.device == 'cuda':
            q = q.to('cuda')

        if enc_input_np is not None:
            enc_input = torch.from_numpy(enc_input_np).float()
            if self.device == 'cuda':
                enc_input = enc_input.to('cuda')
        else:
            enc_input = None

        if self.n_enc_input == 1:
            # (dldq, l) = torch.func.vmap(torch.func.jacfwd(self.inertia_nn_out, argnums=0, has_aux=True), in_dims=(0, None))(q, enc_input)
            (dldq, l) = torch.func.vmap(torch.func.jacfwd(self.lower_tri_inertia_fn, argnums=0, has_aux=True), in_dims=(0, None))(q, enc_input)
            (dVdq, V) = torch.func.vmap(torch.func.jacfwd(self.potential_nn_out, argnums=0, has_aux=True), in_dims=(0, None))(q, enc_input)

        else:
            # (dldq, l) = torch.func.vmap(torch.func.jacfwd(self.inertia_nn_out, argnums=0, has_aux=True))(q, enc_input)
            (dldq, l) = torch.func.vmap(torch.func.jacfwd(self.lower_tri_inertia_fn, argnums=0, has_aux=True))(q, enc_input)
            (dVdq, V) = torch.func.vmap(torch.func.jacfwd(self.potential_nn_out, argnums=0, has_aux=True))(q, enc_input)

        return l, dldq, V, dVdq

    def get_hessian_jac_value_inertia_torch(self, q_np, enc_input_np):

        q = torch.from_numpy(q_np).float()
        if self.device == 'cuda':
            q = q.to('cuda')

        if enc_input_np is not None:
            enc_input = torch.from_numpy(enc_input_np).float()
            if self.device == 'cuda':
                enc_input = enc_input.to('cuda')
        else:
            enc_input = None

        if self.n_enc_input == 1:
            # (dldq, l) = torch.func.vmap(torch.func.jacfwd(self.inertia_nn_out, argnums=0, has_aux=True), in_dims=(0, None))(q, enc_input)
            (dldq, l) = torch.func.vmap(torch.func.jacfwd(self.lower_tri_inertia_fn, argnums=0, has_aux=True), in_dims=(0, None))(q, enc_input)
            (dVdq, V) = torch.func.vmap(torch.func.jacfwd(self.potential_nn_out, argnums=0, has_aux=True), in_dims=(0, None))(q, enc_input)

        else:
            # (dldq, l) = torch.func.vmap(torch.func.jacfwd(self.inertia_nn_out, argnums=0, has_aux=True))(q, enc_input)
            jac_init = time.time()
            (dldq, l) = torch.func.vmap(torch.func.jacfwd(self.lower_tri_inertia_fn, argnums=0, has_aux=True))(q, enc_input)
            (dVdq, V) = torch.func.vmap(torch.func.jacfwd(self.potential_nn_out, argnums=0, has_aux=True))(q, enc_input)
            jac_time = time.time() - jac_init

            time_init = time.time()
            dldqdq = torch.func.vmap(torch.func.hessian(self.lower_tri_inertia_fn_no_aux, argnums=0))(q, enc_input)
            hess_eval = time.time() - time_init

            time_init = time.time()
            dldqdq_fwd = torch.func.vmap(torch.func.jacfwd(torch.func.jacfwd(self.lower_tri_inertia_fn_no_aux, argnums=0)))(q, enc_input)
            hess_fwd = time.time() - time_init

            time_init = time.time()
            dldqdq_rev = torch.func.vmap(torch.func.jacrev(torch.func.jacrev(self.lower_tri_inertia_fn_no_aux, argnums=0)))(q, enc_input)
            hess_rev = time.time() - time_init

            time_init = time.time()
            dldqdq_rev_fwd = torch.func.vmap(torch.func.jacrev(torch.func.jacfwd(self.lower_tri_inertia_fn_no_aux, argnums=0)))(q, enc_input)
            hess_rev_fwd = time.time() - time_init

            time_init = time.time()
            dldqdq_fwd_rev = torch.func.vmap(torch.func.jacfwd(torch.func.jacrev(self.lower_tri_inertia_fn_no_aux, argnums=0)))(q, enc_input)
            hess_fwd_rev = time.time() - time_init

            print(f'jac_time {jac_time} | hess_eval {hess_eval} | hess_fwd {hess_fwd} | hess_rev {hess_rev} | hess_rev_fwd {hess_rev_fwd} | hess_fwd_rev {hess_fwd_rev}')

        return l, dldq, V, dVdq

    def eval_lstm_np(self, input_np):
        input = torch.from_numpy(input_np).float().view(1,self.hist_length,-1)
        output = self.lstm(input).view(-1)
        return output.cpu().detach().numpy()