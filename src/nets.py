'''Contain all related to networks'''
import torch.nn as nn
import torch
import utils
import torchdiffeq as tdeq  # Only need to odeint part, not the entire adjoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


''' Define classifier net '''

def def_classifier_net(args):
    # Simpler classifier net

    class LinearClassifier(torch.nn.Module):
        def __init__(self, args):
            super(LinearClassifier, self).__init__()
            # Define hyperparameters for the classifier
            input_dim = args.Xdim
            classifier_hidden_dim_str = args.classifier_hidden_dim_str # Defines # layers
            classifier_activation = args.classifier_activation # Type of classifier activation
            activation_dict = {'elu': nn.ELU(), 'tanh': nn.Tanh(),
                               'softplus': nn.Softplus(beta=20),
                               'relu': nn.ReLU(), 'prelu': nn.PReLU()}
            # Construct the classifier
            hidden_dims = tuple(map(int, classifier_hidden_dim_str.split("-")))
            dims = (input_dim,) + tuple(hidden_dims) + (1,)
            layers_in_block = []
            for (in_dim, out_dim) in zip(dims[:-1], dims[1:]):
                layers_in_block.append(nn.Linear(in_dim, out_dim))
                if out_dim != 1:
                    layers_in_block.append(activation_dict[classifier_activation])
            self.classifier = nn.Sequential(*layers_in_block)
        def forward(self, x):
            x = self.classifier(x)
            return x

    classifier = LinearClassifier(args).to(device)
    return classifier

'''1. Specific to problems'''

class ODEnet(nn.Module):
    def __init__(self, args, logpx=True, layer_type = 'ODE'):
        super(ODEnet, self).__init__()
        self.net = build_net(
            args.Xdim, args.hidden_dim_str, activation=args.activation, layer=layer_type)
        self.logpx = logpx
        self.layer_type = layer_type

    def forward(self, t, x):
        return net_forward(self, t, x, layer=self.layer_type)


'''2. Generic to all nets'''


def net_forward(self, t, x_and_dlogpx, layer='FC', edge_index=None):
    x, _ = x_and_dlogpx
    if self.stackX:
        N, V, C = x.shape
        x = x.view(x.shape[0], -1)  # Transform to a matrix (N, VC), for graph data
    def get_output(layer):
        out_grad = None
        if layer == 'FC':
            out = self.net(x)
            if self.stackX:
                out_grad = out
                out = out.reshape(N, V, C)
        if 'ODE' in layer:
            out = self.net(t, x)[1]
        return out_grad, out
    if self.logpx:
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            t.requires_grad_(True)
            out_grad, out = get_output(layer)
            if self.div_bf:
                divf = utils.divergence_bf(out, x).flatten()
            else:
                # Sample from Gaussian distribution to get Hutchinson trace estimator
                # Used in FFJORD
                if self.stackX: 
                    # This is because we instead want to compute trace over a matrix of size VC by VC
                    if self.fix_e_ls:
                        if self.e_ls is None:
                            self.e_ls = get_e_ls(out_grad, self.num_e)
                    else:
                        self.e_ls = get_e_ls(out_grad, self.num_e)
                    divf = utils.divergence_approx(out_grad, x, self.e_ls)
                else:
                    if self.fix_e_ls:
                        if self.e_ls is None:
                            self.e_ls = get_e_ls(out, self.num_e)
                    else:
                        self.e_ls = get_e_ls(out, self.num_e)
                    if self.e_ls is None:
                        self.e_ls = get_e_ls(out, self.num_e)
                    divf = utils.divergence_approx(out, x, self.e_ls)
    else:
        # Do not compute divf, just act as placeholder
        divf = torch.zeros(x.shape[0]).to(device)
        _, out = get_output(layer)
    self.counter += 1 # Count how many times the function is called in one odeint along [t_k, t_{k+1}]
    return out, -divf

def get_e_ls(out, num_e):
    e_ls = []
    for i in range(num_e):
        # torch.manual_seed(1103+i)
        e_ls.append(torch.randn_like(out).to(device))
    return e_ls

class CNF(nn.Module):
    '''
        odefunc can be any function, as long as its forward mapping takes t,x and outputs 'out, -divf'
        where out is the output of the function and divf is the divergence of the function
        and the shape of out is the same as the shape of x.
    '''

    def __init__(self, odefunc):
        super(CNF, self).__init__()
        self.odefunc = odefunc

    def forward(self, x, args, reverse=False, test=False, mult_gpu=False):
        self.odefunc.logpx = True
        # NOTE, if we actually build time into self.odefunc, we would have
        # time be accumulated, rather than starting at 0.0
        integration_times = torch.linspace(
            args.Tk_1, args.Tk, args.num_int_pts+1).to(device)
        if test:
            self.odefunc.logpx = False  # Need not track dlogpx
        if reverse:
            integration_times = torch.flip(integration_times, [0])
        # True only for using FCnet on graph node feature
        self.odefunc.stackX = False
        # Default is True for using brute force divergence for 2D examples
        self.odefunc.div_bf = args.div_bf
        # Number of random projection to get trace estimator
        self.odefunc.num_e = args.num_e
        dlogpx = torch.zeros(x.shape[0]).to(device)
        ######################
        self.odefunc.e_ls = None 
        # Same eps. for the entire [0,T] (when breaking to sub-intervals)
        self.odefunc.fix_e_ls = args.fix_e_ls
        self.odefunc.counter = 0
        ########################
        if args.use_NeuralODE is False:
            # Naive backprop, No backward adjoint step
            predz, dlogpx = odeint(
                self.odefunc, (x, dlogpx), integration_times, mtd=args.int_mtd,
                rtol = args.rtol, atol = args.atol)
        else:
            # Extract only the essential part from it
            predz, dlogpx = odeint_adjoint(
                self.odefunc, (x, dlogpx), integration_times, mtd=args.int_mtd,
                rtol = args.rtol, atol = args.atol)
        if mult_gpu:
            # This is because predz stores intermediate trajectories along dim 0, 
            # and in multi-gpu context,
            # this would not be correct
            return predz[-1], dlogpx[-1]
        else:
            return predz, dlogpx
    
mtd_dict = {'RK4': 'rk4', 'Euler': 'euler', 'DOPRI5': 'dopri5'}


def odeint(func, x_now, t_ls, mtd='RK4', rtol=1e-4, atol=1e-4):
    '''
        # New implementation based on torchdiffeq ODE
        # Old implementation based on no tolerance is below
    '''
    shapes, func, x_now = utils._check_inputs(func, x_now)
    mtd = mtd_dict[mtd]
    # atol and rtol only used for adaptive solver.
    # but this on image is very very slow, esp if rtol and atol to 1e-5
    # It may help with performance.
    solution = tdeq.odeint(func, x_now, t_ls, rtol=rtol,
                           atol=atol, method=mtd)
    solution = utils._flat_to_shape(solution, (len(t_ls),), shapes)
    return solution

def odeint_adjoint(func, x_now, t_ls, mtd='RK4', rtol=1e-4, atol=1e-4):
    '''
        # New implementation based on torchdiffeq ODE
        # Old implementation based on no tolerance is below
    '''
    shapes, func, x_now = utils._check_inputs(func, x_now)
    mtd = mtd_dict[mtd]
    # atol and rtol only used for adaptive solver.
    # but this on image is very very slow, esp if rtol and atol to 1e-5
    # It may help with performance.
    solution = tdeq.odeint_adjoint(func, x_now, t_ls, rtol=rtol,
                                   atol=atol, method=mtd)
    solution = utils._flat_to_shape(solution, (len(t_ls),), shapes)
    return solution

def build_net(Xdim, hidden_dim_str, activation='elu', layer='FC'):
    hidden_dims = tuple(map(int, hidden_dim_str.split("-")))
    dims = (Xdim,) + tuple(hidden_dims) + (Xdim,)
    activation_dict = {'elu': nn.ELU(), 'tanh': nn.Tanh(),
                       'softplus': nn.Softplus(beta=20)}
    layers_in_block = []
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        if layer == 'FC':
            layers_in_block.append(nn.Linear(in_dim, out_dim))
            if i < len(hidden_dims):
                layers_in_block.append(activation_dict[activation])
        if layer == 'ODE':
            if i < len(hidden_dims):
                act = activation_dict[activation]
            else:
                act = None
            layers_in_block.append(
                ConcatSquashLinear(in_dim, out_dim, act))
        if layer == 'ODE_concat':
            if i < len(hidden_dims):
                layers_in_block.append(ConcatLinear(in_dim, out_dim, activation_dict[activation]))
            else:
                layers_in_block.append(ConcatLinear(in_dim, out_dim, None))
    seq_class = utils.mySequential
    return seq_class(*layers_in_block)

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

### Customized nets ###

class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, act):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)
        self.act = act

    def forward(self, t, x):
        # Note, return t below is like a placeholder I guess. Keep in
        if self.act is not None:
            return t, self.act(self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(1, 1)))
                               + self._hyper_bias(t.view(1, 1)))
        else:
            return t, self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(1, 1))) \
                + self._hyper_bias(t.view(1, 1))

class ConcatLinear(nn.Module):
    def __init__(self, dim_in, dim_out, act):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in + 1, dim_out)
        self.act = act

    def forward(self, t, x):
        tt = torch.ones_like(x[:,:1]) * t
        ttx = torch.cat([tt, x], 1)
        if self.act is not None:
            return t, self.act(self._layer(ttx))
        else:
            return t, self._layer(ttx)


