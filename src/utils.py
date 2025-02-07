'''Contain other utilities, including class for training and evaluation'''
import torch.nn as nn
import torch
import numpy as np
import os
import numpy as np
import itertools as it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' Quick '''
def get_batches(xfull, bsize):
    train_tot = xfull.shape[0]
    num_b = int(train_tot / bsize)
    seed = np.random.randint(low = 0, high = 1e7)
    np.random.seed(seed)
    idxes = np.random.choice(range(train_tot), train_tot, replace=False)
    cum, batches = 0, []
    for i in range(num_b):
        batches.append(idxes[cum:cum + bsize])
        cum += bsize
    if train_tot - cum > 0:
        batches.append(idxes[cum:])
    return batches

def SimpleDloader(x, y, batches):
    dloader = []
    for batch_idx in batches:
        dloader_tmp = [x[batch_idx], y[batch_idx]]
        dloader.append(dloader_tmp)
    return dloader


'''0. For NeuralODE
From torchdiffeq
'''




def _flat_to_shape(tensor, length, shapes):
    tensor_list = []
    total = 0
    for shape in shapes:
        next_total = total + shape.numel()
        # It's important that this be view((...)), not view(...). Else when length=(), shape=() it fails.
        tensor_list.append(
            tensor[..., total:next_total].view((*length, *shape)))
        total = next_total
    return tuple(tensor_list)


class _TupleFunc(torch.nn.Module):
    def __init__(self, base_func, shapes):
        super(_TupleFunc, self).__init__()
        self.base_func = base_func
        self.shapes = shapes

    def forward(self, t, y):
        f = self.base_func(t, _flat_to_shape(y, (), self.shapes))
        return flatten_cat(f)


def _check_inputs(func, y0):
    is_tuple = not isinstance(y0, torch.Tensor)
    shapes = None
    if is_tuple:
        assert isinstance(
            y0, tuple), 'y0 must be either a torch.Tensor or a tuple'
        shapes = [y0_.shape for y0_ in y0]
        y0 = flatten_cat(y0)
        func = _TupleFunc(func, shapes)
    return shapes, func, y0


def flatten_cat(y):
    return torch.cat([y_.reshape(-1) for y_ in y])

def divergence_bf(dx, x):
    import itertools as it
    sum_diag = 0.
    if len(x.shape) == 2:
        # Tabular data with shape (N,C)
        for i in range(x.shape[1]):
            sum_diag += torch.autograd.grad(dx[:, i].sum(),
                                            x, create_graph=True)[0][:, i]
    elif len(x.shape) == 3:
        # Graph data with shape (N,V,C)
        V, C = x.shape[1], x.shape[2]
        for v, c in it.product(*[range(V), range(C)]):
            sum_diag += torch.autograd.grad(dx[:, v,c].sum(),
                                            x, create_graph=True)[0][:, v,c]
    else:
        # Image data with shape (N,in_channel,h,w),
        # where in_channel dictates color channels and h,w are height and width
        in_c, h, w, = x.shape[-3], x.shape[-2], x.shape[-1]
        for c, i, j in it.product(*[range(in_c), range(h), range(w)]):
            sum_diag += torch.autograd.grad(dx[:, c, i, j].sum(),
                                            x, create_graph=True)[0][:, c, i, j]
    return sum_diag.view(x.shape[0], 1)

def divergence_approx(out, x, e_ls=[]):
    '''
        e_ls = [e_1,e_2,...] where
            e_i has the same dimension as x.
            We have multiple e_i to have more accurate estimation of the trace
        It is taken from FFJORD
        Here, e would be generated with i.i.d. Radamacher RVs. as
            e = torch.randint(low=0, high=2, size=x.shape).to(x) * 2 - 1
        It is pre-stored as explained in FFJORD
            "To keep the dynamics deterministic within each call to the ODE solver, we can use a fixed noise vector ε for the duration of each solve without introducing bias:"
    '''
    approx_tr_dzdx_ls = []
    for e in e_ls:
        # See https://blog.csdn.net/waitingwinter/article/details/105774720 for how
        # torch.autograd.grad works
        # out = f(x(t),t) (velocity field): bsize-by-d
        # x = x(t) (input): bsize-by-d
        # e = epsilon (random direction): bsize-by-d
        e_dzdx = torch.autograd.grad(out, x, e, create_graph=True)[0]
        e_dzdx_e = e_dzdx * e
        approx_tr_dzdx = e_dzdx_e.view(
            x.shape[0], -1).sum(dim=1)  # = batch size
        approx_tr_dzdx_ls.append(approx_tr_dzdx)
    return torch.vstack(approx_tr_dzdx_ls).mean(dim=0)


'''1. Network related'''


def reparam_t(args, iter):
    # args.T_ls has length B and stores T_1,...,T_B for B ends of the integrals
    # args.T = args.T_ls[iter]
    args.Tk_1 = np.sum(args.T_ls[:iter])
    args.Tk = np.sum(args.T_ls[:iter+1])


class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input


def map_for_or_back(input, num_blocks, FlowNet, args, reverse=True, return_dlogpx=False, T=None, test = False):
    # Map forward or backward
    with torch.no_grad():
        est_ls = []
        dlogpx_ls = []
        for j in range(num_blocks):
            which_b = num_blocks-1-j if reverse else j
            model = FlowNet[which_b]
            reparam_t(args, which_b)  # get args.T based on args.T0
            if T is not None:
                # When we check invertibility for ONE block, we need this.
                # O/W, the T is incorrect
                args.T = T
            input_tmp, dlogpx_block_ls = model(input, args, reverse=reverse, test = test)
            input = input_tmp[-1]
            # Because the end of previous block = start of next block, to avoid repetition
            # est_ls thus store the trajectory over each block on sub-intervals
            start_idx = 0 if j == 0 else 1
            est_ls.append(input_tmp[start_idx:])
            # How much THIS block changes log-lik
            # dlogpx_ls.append(dlogpx_block_ls[-1].mean()) # Average over sample at EACH block
            dlogpx_ls.append(dlogpx_block_ls[-1])
        est_ls = torch.vstack(est_ls)
        dlogpx_ls = torch.vstack(dlogpx_ls)
        if return_dlogpx:
            return est_ls, dlogpx_ls
        else:
            return est_ls
        
def map_for_or_back_cont_t(input, num_blocks, model, args, reverse=True, return_dlogpx=False, 
                           test = False, mult_gpu = False):
    # Map forward or backward
    with torch.no_grad():
        est_ls = []
        dlogpx_ls = []
        for j in range(num_blocks):
            which_b = num_blocks-1-j if reverse else j
            reparam_t(args, which_b)  # get args.T based on args.T0
            if mult_gpu:
                output, dlogpx_block = model(input, args, reverse=reverse, test = test, 
                                               mult_gpu = True)
                est_ = [input, output] if j == 0 else [output]
                est_ls.append(torch.stack(est_))
                dlogpx_ls.append(dlogpx_block)
                input = output
            else:
                input_tmp, dlogpx_block_ls = model(input, args, reverse=reverse, test = test)
                input = input_tmp[-1]
                # Because the end of previous block = start of next block, to avoid repetition
                # est_ls thus store the trajectory over each block on sub-intervals
                start_idx = 0 if j == 0 else 1
                est_ls.append(input_tmp[start_idx:])
                dlogpx_ls.append(dlogpx_block_ls[-1])
        est_ls = torch.vstack(est_ls)
        dlogpx_ls = torch.vstack(dlogpx_ls)
        if return_dlogpx:
            return est_ls, dlogpx_ls
        else:
            return est_ls

def mode_2_joint_training(xinput, model, args, return_raw_movement = False):
    # Flow through all intervals to get final loss and KL
    num_blocks = len(args.T_ls)
    loss_V, loss_div, loss_W2 = 0, 0, 0
    raw_movement = 0
    for k in range(num_blocks):
        reparam_t(args, k)
        predz, dlogpx = model(xinput, args, reverse = False, test = False) 
        xpk = predz[-1]
        loss_div += dlogpx[-1].mean()
        loss_W2 += l2_norm_sqr(xpk - predz[0])/(args.Tk - args.Tk_1)
        raw_movement += l2_norm_sqr(xpk - predz[0])
        xinput = xpk
    loss_V = l2_norm_sqr(xpk)
    if return_raw_movement:
        return loss_V, loss_div, loss_W2, raw_movement
    else:
        return loss_V, loss_div, loss_W2


def get_l2_squared(input):
    if len(input.size()) > 2:
        return input.view(input.shape[0], -1).pow(2).sum(axis=1)
    else:
        return input.pow(2).sum(axis=1)


def l2_norm_sqr(input, return_full = False):
    '''
        For tensor with shape (N,M1,M2,...),
        We flatten it to be (N,M1*M2*...)
        Then treate it as N vectors to compute l2^2 norm
    '''
    if len(input.size()) > 2:
        norms = 0.5*input.view(input.shape[0], -1).pow(2).sum(axis=1)
    else:
        norms = 0.5*input.pow(2).sum(axis=1)
    if return_full:
            return norms
    else:
        return norms.mean()


'''4. Others minor ones'''


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def load_saved_checkpoint(loader_args):
    int_mtd, Xdim, reparam_type, word, netname, continuous_param, p = loader_args.int_mtd, loader_args.Xdim, loader_args.reparam_type, loader_args.word, loader_args.netname, loader_args.continuous_param, loader_args.p
    netpre = netname[1:-3]
    if word == '':
        netname = ''
    param_type = '_cont_param' if continuous_param else ''
    filepath = f'JKO_{int_mtd}_{netpre}_Xdim={Xdim}_reparam={reparam_type}{word}{netname}{param_type}_phase{p}'
    suffix = '_refinement' if loader_args.refinement else ''
    filepath += suffix
    filepath += loader_args.arbitary_save_suffix
    return filepath

# MMD stuff
def pdist(sample_1, sample_2, norm=2):
    r"""Compute the matrix of all squared pairwise distances.
    """
    return torch.cdist(sample_1, sample_2, p=norm)


def get_MMD(kernels,sample1_idx,sample2_idx, a00, a11, a01):
    k_1 = torch.index_select(torch.index_select(kernels,0,sample1_idx),1,sample1_idx)
    k_2 = torch.index_select(torch.index_select(kernels,0,sample2_idx),1,sample2_idx)
    k_12 = torch.index_select(torch.index_select(kernels,0,sample1_idx),1,sample2_idx)
    mmd =  a00 * k_1.sum() + a11 * k_2.sum() + 2 * a01 * k_12.sum()
    return mmd

def get_bootstrap_MMD_imbalance(num_b, test_full, X_test_hat, two_dim=False):
    import numpy as np
    with torch.no_grad():
        distancesX = pdist(test_full, test_full) # Size N-by-N
        gamma = torch.median(distancesX)
        if two_dim:
            gamma = gamma/10
        alpha_local = 0.5/gamma**2
        alphas = [alpha_local]
        alphas.append(torch.tensor(0.5).to(device))
        full_sample = torch.vstack([test_full, X_test_hat])
        distances = pdist(full_sample,full_sample) # Size (N+M)-by-(N+M)
        N = test_full.shape[0]
        M = X_test_hat.shape[0]
        a00, a11, a01 = 1./N**2, 1./M**2,-1./(N*M)
        kernels_ls = {}        
        for alpha in alphas:
            kernels_ls[alpha] = torch.exp(-alpha*distances**2)
        boot_all_alpha_thres = {}
        boot_all_alpha_dict = {alpha.item():[] for alpha in alphas}
        for _ in range(num_b):
            perm_idx = torch.randperm(N+M).to(device)
            sample1_idx = perm_idx[:N]
            sample2_idx = perm_idx[N:]
            for alpha in alphas:
                boot_all_alpha_dict[alpha.item()].append(get_MMD(kernels_ls[alpha],sample1_idx,sample2_idx, a00, a11, a01).item())
        for j, alpha in enumerate(alphas):
            prefix = 'Under median trick' if j == 0 else 'Under constant'
            print(f'{prefix} {alpha.item():.2e}, upper 5% MMD is {np.percentile(boot_all_alpha_dict[alpha.item()],95):.2e}')
            boot_all_alpha_thres[alpha.item()] = np.percentile(boot_all_alpha_dict[alpha.item()],95)
    return boot_all_alpha_dict, boot_all_alpha_thres


def quick_MMD(alphas, test_full, X_test_hat):
    full_sample = torch.vstack([test_full, X_test_hat])
    distances = pdist(full_sample,full_sample) # Size (N+M)-by-(N+M)
    kernels_ls = {}        
    for alpha in alphas:
        kernels_ls[alpha] = torch.exp(-alpha*distances**2)
    N = test_full.shape[0]
    M = X_test_hat.shape[0]
    a00, a11, a01 = 1./N**2, 1./M**2,-1./(N*M)
    perm_idx = torch.arange(N+M).to(device)
    sample1_idx = perm_idx[:N]
    sample2_idx = perm_idx[N:]
    MMD_dict = {}
    for alpha in alphas:
        MMD_dict[alpha]=get_MMD(kernels_ls[alpha],sample1_idx,sample2_idx, a00, a11, a01).item()
    for alpha, MMD in MMD_dict.items():
        print(f'Under alpha={alpha:.2e}, MMD is {MMD:.2e}')
    return list(MMD_dict.values())
############
############
