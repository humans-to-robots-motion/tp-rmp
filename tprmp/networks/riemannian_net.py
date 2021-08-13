import numpy as np
import torch
from torch import nn

from tprmp.networks.rmp_net import LowTri, LagrangianLayer


class RiemannianNetwork(nn.Module):
    def __init__(self, dim_M, **kwargs):
        super(RiemannianNetwork, self).__init__()
        self.n_width = kwargs.get("n_width", 128)
        self.n_hidden = kwargs.get("n_depth", 2)
        self._b0 = kwargs.get("b_init", 0.1)
        self._b0_diag = kwargs.get("b_diag_init", 0.1)
        self._w_init = kwargs.get("w_init", "xavier_normal")
        self._g_hidden = kwargs.get("g_hidden", np.sqrt(2.))
        self._g_output = kwargs.get("g_hidden", 0.125)
        self._p_sparse = kwargs.get("p_sparse", 0.2)
        self._epsilon = kwargs.get("diagonal_epsilon", 1.e-5)
        # Construct Weight Initialization:
        if self._w_init == "xavier_normal":
            # Construct initialization function:
            def init_hidden(layer):
                # Set the Hidden Gain:
                if self._g_hidden <= 0.0:
                    hidden_gain = torch.nn.init.calculate_gain('relu')
                else:
                    hidden_gain = self._g_hidden
                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.xavier_normal_(layer.weight, hidden_gain)
        elif self._w_init == "orthogonal":
            # Construct initialization function:
            def init_hidden(layer):
                # Set the Hidden Gain:
                if self._g_hidden <= 0.0:
                    hidden_gain = torch.nn.init.calculate_gain('relu')
                else:
                    hidden_gain = self._g_hidden
                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.orthogonal_(layer.weight, hidden_gain)
        elif self._w_init == "sparse":
            assert self._p_sparse < 1. and self._p_sparse >= 0.0

            # Construct initialization function:
            def init_hidden(layer):
                p_non_zero = self._p_sparse
                hidden_std = self._g_hidden
                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.sparse_(layer.weight, p_non_zero, hidden_std)
        else:
            raise ValueError(f"Weight Initialization Type must be in ['xavier_normal', 'orthogonal', 'sparse'] but is {self._w_init}")
        # Compute In- / Output Sizes:
        self.dim_M = dim_M
        self.m = int((self.dim_M**2 + self.dim_M) / 2)
        # Compute non-zero elements of L:
        l_output_size = int((self.dim_M**2 + self.dim_M) / 2)
        l_lower_size = l_output_size - self.dim_M
        # Calculate the indices of the diagonal elements of L:
        idx_diag = np.arange(self.dim_M) + 1
        idx_diag = idx_diag * (idx_diag + 1) / 2 - 1
        # Calculate the indices of the off-diagonal elements of L:
        idx_tril = np.extract([x not in idx_diag for x in np.arange(l_output_size)], np.arange(l_output_size))
        # Indexing for concatenation of l_o  and l_d
        cat_idx = np.hstack((idx_diag, idx_tril))
        order = np.argsort(cat_idx)
        self._idx = np.arange(cat_idx.size)[order]
        # create it once and only apply repeat, this may decrease memory allocation
        self._eye = torch.eye(self.dim_M).view(1, self.dim_M, self.dim_M)
        self.low_tri = LowTri(self.dim_M)
        # Create Network:
        self.layers = nn.ModuleList()
        non_linearity = kwargs.get("activation", "ReLu")
        # Create Input Layer:
        self.layers.append(LagrangianLayer(self.dim_M, self.n_width, activation=non_linearity))
        init_hidden(self.layers[-1])
        # Create Hidden Layer:
        for _ in range(1, self.n_hidden):
            self.layers.append(LagrangianLayer(self.n_width, self.n_width, activation=non_linearity))
            init_hidden(self.layers[-1])
        # Create output Layer:
        self.net_lo = LagrangianLayer(self.n_width, l_lower_size, activation="Linear")
        init_hidden(self.net_lo)
        # The diagonal must be non-negative. Therefore, the non-linearity is set to ReLu.
        self.net_ld = LagrangianLayer(self.n_width, self.dim_M, activation="ReLu")
        init_hidden(self.net_ld)
        torch.nn.init.constant_(self.net_ld.bias, self._b0_diag)

    def forward(self, q, qd):
        qd_3d = qd.view(-1, self.dim_M, 1)
        qd_4d = qd.view(-1, 1, self.dim_M, 1)
        # Create initial derivative of dq/dq.
        der = self._eye.repeat(q.shape[0], 1, 1).type_as(q)
        # Compute hiddens layers
        y, der = self.layers[0](q, der)
        for i in range(1, len(self.layers)):
            y, der = self.layers[i](y, der)
        # Compute the network heads including the corresponding derivative:
        l_lower, der_l_lower = self.net_lo(y, der)
        l_diag, der_l_diag = self.net_ld(y, der)
        # Assemble l and der_l
        l_diag = l_diag + self._epsilon
        l = torch.cat((l_diag, l_lower), 1)[:, self._idx]  # noqa
        der_l = torch.cat((der_l_diag, der_l_lower), 1)[:, self._idx, :]
        # Compute M:
        L = self.low_tri(l)
        LT = L.transpose(dim0=1, dim1=2)
        M = torch.matmul(L, LT)
        M_inv = torch.cholesky_inverse(L)
        # Calculate dM/dt
        Ldt = self.low_tri(torch.matmul(der_l, qd_3d).view(-1, self.m))
        Mdt = torch.matmul(L, Ldt.transpose(dim0=1, dim1=2)) + torch.matmul(Ldt, LT)
        # Calculate dM/dq:
        Ldq = self.low_tri(der_l.transpose(2, 1).reshape(-1, self.m)).reshape(-1, self.dim_M, self.dim_M, self.dim_M)
        Mdq = torch.matmul(Ldq, LT.view(-1, 1, self.dim_M, self.dim_M)) + torch.matmul(L.view(-1, 1, self.dim_M, self.dim_M), Ldq.transpose(2, 3))
        # Compute the Coriolis & Centrifugal forces:
        Mdt_qd = torch.matmul(Mdt, qd_3d).view(-1, self.dim_M)
        quad_dq = torch.matmul(qd_4d.transpose(dim0=2, dim1=3), torch.matmul(Mdq, qd_4d)).view(-1, self.dim_M)
        c = Mdt_qd - 1. / 2. * quad_dq
        # Compute kinetic energy T
        M_qd = torch.matmul(M, qd_3d).view(-1, self.dim_M)
        T = 1. / 2. * torch.matmul(qd_4d.transpose(dim0=2, dim1=3), M_qd.view(-1, 1, self.dim_M, 1)).view(-1)
        return M, M_inv, c, T

    def cuda(self, device=None):
        # Move the Network to the GPU:
        super(RiemannianNetwork, self).cuda(device=device)
        # Move the eye matrix to the GPU:
        self._eye = self._eye.cuda()
        self.device = self._eye.device
        return self

    def cpu(self):
        # Move the Network to the CPU:
        super(RiemannianNetwork, self).cpu()
        # Move the eye matrix to the CPU:
        self._eye = self._eye.cpu()
        self.device = self._eye.device
        return self
