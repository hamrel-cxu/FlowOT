import utils
import numpy as np
import matplotlib.pyplot as plt
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import copy
import nets
from torch.nn import ModuleList as ModList
import os

''' Part I: Loss computation '''

def logit_loss(rnet, x, y):
    '''
        rnet:: R^d -> R
        At optimum, rnet^*(x) = \log q_y(x) - \log p_x(x)
    '''
    softplus = torch.nn.Softplus(beta = 1)
    loss_X = softplus(rnet(x)).mean()
    loss_Y = softplus(-rnet(y)).mean()
    return loss_X + loss_Y


def alternate_training_ode_loss(rnetPQ, rnetQP, FlowNet_P, FlowNet_Q, 
                             args_P, args_Q, Pdata, Qdata, 
                             return_full = True, gamma = 1,
                             PtoQ = True):
    if PtoQ:
        W2_PtoQ, Qest = flow_P_Q(FlowNet_P, FlowNet_Q, args_P, args_Q, Pdata, Qdata, PtoQ = PtoQ, return_W2 = True)
        Logit_PtoQ = - rnetPQ(Qest).mean()       
        if return_full:
            return Logit_PtoQ + gamma*W2_PtoQ
        else:
            return Logit_PtoQ, gamma*W2_PtoQ
    else:
        W2_QtoP, Pest = flow_P_Q(FlowNet_P, FlowNet_Q, args_P, args_Q, Pdata, Qdata, PtoQ = PtoQ, return_W2 = True)
        Logit_QtoP = - rnetQP(Pest).mean()        
        if return_full:
            return Logit_QtoP + gamma*W2_QtoP
        else:
            return Logit_QtoP, gamma*W2_QtoP

    
def flow_P_Q(FlowNetP, FlowNetQ, args_P, args_Q, 
             Pdata, Qdata, PtoQ = True, return_W2 = False):
    '''
        This computes either P -> Z -> Q or Q -> Z -> P
        With either the complete list of intermediate results returned
        Or just the final results and W2 (for training)   
    '''
    #### Get right model and data
    if PtoQ:
        xraw = Pdata.clone()
        yraw = Qdata.clone()
        FlowNet_P = FlowNetP
        FlowNet_Q = FlowNetQ
        args_P_ = args_P
        args_Q_ = args_Q
    else:
        xraw = Qdata.clone()
        yraw = Pdata.clone()
        FlowNet_P = FlowNetQ
        FlowNet_Q = FlowNetP
        args_P_ = args_Q
        args_Q_ = args_P
    ##### Flow
    zhat_ls = []
    xinput = xraw.clone()
    W2_over_blocks = 0 # W2 penalty over blocks
    numP_blocks = args_P_.num_blocks
    numQ_blocks = args_Q_.num_blocks
    for b in range(numP_blocks):
        utils.reparam_t(args_P_, b)
        xfor_ls, _ = FlowNet_P(xinput, args_P_, 
                        reverse = False, test = True)
        zhat_ls.append(xfor_ls[-1])
        xinput = xfor_ls[-1]
        W2_block = utils.l2_norm_sqr(xfor_ls[-1] - xfor_ls[0])/(args_P_.Tk - args_P_.Tk_1)
        W2_over_blocks += W2_block
    yinput = zhat_ls[-1]
    yhat_ls = []
    for b in range(numQ_blocks):
        utils.reparam_t(args_Q_, numQ_blocks - b - 1)
        yfor_ls, _ = FlowNet_Q(yinput, args_Q_, 
                        reverse = True, test = True)
        yhat_ls.append(yfor_ls[-1])
        yinput = yfor_ls[-1]
        W2_block = utils.l2_norm_sqr(yfor_ls[-1] - yfor_ls[0])/(args_Q_.Tk - args_Q_.Tk_1)
        W2_over_blocks += W2_block
    if return_W2:
        # This is for training, so only need W2 and last pushed samples
        return W2_over_blocks, yhat_ls[-1]
    else:
        # This is for visualization alone over trajectory, so we need all samples
        full_PQ_traj = [xraw] + zhat_ls + yhat_ls + [yraw]
        return full_PQ_traj
    



''' Part II: Training '''

def train_rnet(rnetPQ, rnetQP, optimizer_logitP, optimizer_logitQ,
               dloader, Dataest_full, target_data,
               losses_per_flip, flip,
               PtoQ = True):

    ###### Training:
    with torch.no_grad():
        if PtoQ:
            rnet = rnetPQ
            optimizer = optimizer_logitP
            dict_name = 'r_lp1'
        else:
            rnet = rnetQP
            optimizer = optimizer_logitQ
            dict_name = 'r_tilde0'
    for batch_now in dloader:
        Dataest_sub, Datadata_sub = batch_now
        Logit_loss = logit_loss(rnet, Dataest_sub, Datadata_sub)
        optimizer.zero_grad()
        Logit_loss.backward()
        optimizer.step()

    ###### Append latest loss on all training samples
    logit_loss_ = logit_loss(rnet, Dataest_full, target_data).item()
    if dict_name in losses_per_flip[f'Flip{flip}'].keys():
        losses_per_flip[f'Flip{flip}'][dict_name].append(logit_loss_)
    else:
        losses_per_flip[f'Flip{flip}'][dict_name] = [logit_loss_]
    return losses_per_flip



def train_PQ_flow(FlowNet_P, FlowNet_Q, args_P, args_Q, Pdata, Qdata,
                  Pdata_test, Qdata_test,
                  optimizer_flow, rnetPQ, rnetQP,
                  losses_per_flip, overall_progress_loss, flip, meta_args,
                  PtoQ = True):
    ###### Training
    batch_idx_flow = utils.get_batches(Pdata, bsize=meta_args.batch_size_flow)
    data_loader_flow = utils.SimpleDloader(Pdata, Qdata, batch_idx_flow)
    for batch_now in data_loader_flow:
        Pdata_sub, Qdata_sub = batch_now
        optimizer_flow.zero_grad()
        loss = alternate_training_ode_loss(rnetPQ, rnetQP, FlowNet_P, FlowNet_Q, 
                                              args_P, args_Q, Pdata_sub, Qdata_sub,
                                              gamma = meta_args.gamma,
                                              PtoQ = PtoQ)
        loss.backward()
        optimizer_flow.step()
    
    ###### Append latest loss on all training samples
    loss_full = alternate_training_ode_loss(rnetPQ, rnetQP, FlowNet_P, FlowNet_Q, 
                                                args_P, args_Q, Pdata, Qdata,
                                                gamma = meta_args.gamma,
                                                return_full = False,
                                                PtoQ = PtoQ)
    loss_full = [l.item() for l in loss_full]
    KL_PQ, W2_PQ = loss_full
    if PtoQ:
        dict_name = 'Tfor'
    else:
        dict_name = 'Tback'
    if dict_name in losses_per_flip[f'Flip{flip}'].keys():
        losses_per_flip[f'Flip{flip}'][dict_name].append([KL_PQ, W2_PQ, KL_PQ+W2_PQ])
    else:
        losses_per_flip[f'Flip{flip}'][dict_name] = [[KL_PQ, W2_PQ, KL_PQ+W2_PQ ]]

    ###### Append the est_KL, trueKL, and W2 on TEST data
    loss_full = alternate_training_ode_loss(rnetPQ, rnetQP, FlowNet_P, FlowNet_Q, 
                                                args_P, args_Q, Pdata_test, Qdata_test,
                                                gamma = 1,
                                                return_full = False,
                                                PtoQ = PtoQ)
    loss_full = [l.item() for l in loss_full]
    KL_PQ, W2_PQ = loss_full
    KL_true = 0
    if meta_args.get_true_KL:
        assert Pdata_test.shape[1] == 2
        # Get KL for Gaussian mixture of P_L vs Q
        # Need to store th dlogpx during backward integration as well
        import data
        mvns = data.get_mvn()
        KL_true = get_true_KL_func(FlowNet_P, FlowNet_Q, args_P, args_Q,
                                   Pdata_test, Qdata_test,
                                   mvns, PtoQ = PtoQ)
    overall_progress_loss.append([KL_PQ, KL_true, W2_PQ])
    return losses_per_flip, overall_progress_loss


''' Part III: Initialization and saving '''

def initialization_nets(rnet_args, lrs, init_P, init_Q):
    ### Collect parameters and define logit nets
    FlowNet_P, FlowNet_Q = copy.deepcopy(init_P.model), copy.deepcopy(init_Q.model)
    args_P, args_Q = init_P.args, init_Q.args
    rnet_lr, flow_lr = lrs
    optimizer_flow = torch.optim.Adam(ModList([FlowNet_P]+[FlowNet_Q]).parameters(), lr = flow_lr)
    # Use untrained logits with initial flow
    rnetPQ = nets.def_classifier_net(rnet_args)
    rnetQP = nets.def_classifier_net(rnet_args)
    optimizer_logitP = torch.optim.Adam(rnetPQ.parameters(), lr = rnet_lr)
    optimizer_logitQ = torch.optim.Adam(rnetQP.parameters(), lr = rnet_lr)
    output = [FlowNet_P, FlowNet_Q, args_P, args_Q, optimizer_flow, 
              rnetPQ, rnetQP, optimizer_logitP, optimizer_logitQ]
    return output

def load_or_save_checkpt(savename, FlowNet_P, FlowNet_Q, rnetPQ, rnetQP, 
                         optimizer_flow = None, optimizer_logitP = None, optimizer_logitQ = None,
                         losses_per_flip = None,
                         overall_progress_loss_PQ = None,
                         overall_progress_loss_QP = None,
                         meta_args = None,
                         flip = None, load = True):
    if load and os.path.exists(savename):
        state_dicts = torch.load(savename)
        FlowNet_P.load_state_dict(state_dicts['params_ls_P'])
        FlowNet_Q.load_state_dict(state_dicts['params_ls_Q'])
        rnetPQ.load_state_dict(state_dicts['rnetPQ'])
        rnetQP.load_state_dict(state_dicts['rnetQP'])
        if 'optimizer_flow' in state_dicts.keys():
            optimizer_flow.load_state_dict(state_dicts['optimizer_flow'])
        if 'optimizer_logitP' in state_dicts.keys():
            optimizer_logitP.load_state_dict(state_dicts['optimizer_logitP'])
        if 'optimizer_logitQ' in state_dicts.keys():
            optimizer_logitQ.load_state_dict(state_dicts['optimizer_logitQ'])
        meta_args = state_dicts['meta_args']
        losses_per_flip = state_dicts['losses_per_flip']
        overall_progress_loss_PQ = state_dicts['overall_progress_loss_PQ']
        overall_progress_loss_QP = state_dicts['overall_progress_loss_QP']
        return meta_args, losses_per_flip, overall_progress_loss_PQ, overall_progress_loss_QP

    else:
        meta_args.flip_start = flip # This indicates we will start at the NEXT flip
        params_ls_P = FlowNet_P.state_dict()
        params_ls_Q = FlowNet_Q.state_dict()
        state_dicts = {'params_ls_P': params_ls_P, 'params_ls_Q': params_ls_Q,
                       'rnetPQ': rnetPQ.state_dict(), 'rnetQP': rnetQP.state_dict(),
                       'optimizer_flow': optimizer_flow.state_dict(),
                       'optimizer_logitP': optimizer_logitP.state_dict(),
                       'optimizer_logitQ': optimizer_logitQ.state_dict(),
                       'meta_args': meta_args,
                       'losses_per_flip': losses_per_flip,
                       'overall_progress_loss_PQ': overall_progress_loss_PQ,
                       'overall_progress_loss_QP': overall_progress_loss_QP}
        torch.save(state_dicts, savename)


''' Part IV: Visualization '''

def visualize_PQ_flow(FlowNet_P, FlowNet_Q, args_P, args_Q, 
                      Pdata_test, Qdata_test, flip,
                      losses_per_flip, overall_progress_loss,
                      PtoQ = True,
                      return_full_traj = False,
                      plot_loss = True, s=1):
    ###### Check inversion error from P->Q->P
    with torch.no_grad():
        if PtoQ:
            _, Qest_full = flow_P_Q(FlowNet_P, FlowNet_Q, args_P, args_Q, 
                            Pdata_test, Qdata_test, PtoQ = True, return_W2 = True)
            _, Pest_full = flow_P_Q(FlowNet_P, FlowNet_Q, args_P, args_Q, 
                            Pdata_test, Qest_full, PtoQ = False, return_W2 = True)
            inv_err_P = utils.l2_norm_sqr(Pest_full - Pdata_test).item()
            print(f'Inversion error P -> Q -> P: {inv_err_P:.2e}')
        else:
            _, Pest_full = flow_P_Q(FlowNet_P, FlowNet_Q, args_P, args_Q,
                            Pdata_test, Qdata_test, PtoQ = False, return_W2 = True)
            _, Qest_full = flow_P_Q(FlowNet_P, FlowNet_Q, args_P, args_Q,
                            Pest_full, Qdata_test, PtoQ = True, return_W2 = True)
            inv_err_Q = utils.l2_norm_sqr(Qest_full - Qdata_test).item()
            print(f'Inversion error Q -> P -> Q: {inv_err_Q:.2e}')
    
    if plot_loss:
        ###### Visualize this J2 many loss 
        print(f'Visualize J2 training losses')
        fig0, ax = plt.subplots(1, 3, figsize = (12, 4))
        if PtoQ:
            dict_name = 'Tfor'
        else:
            dict_name = 'Tback'
        losses = np.array(losses_per_flip[f'Flip{flip}'][dict_name])
        ax[0].plot(losses[:,0], '-o')
        ax[1].plot(losses[:,1], '-o')
        if PtoQ:
            ax[0].set_title(r'$\mathbb{E}_{X\sim P }[-r_{L+1}(T_{1:L}(X))]$')
            ax[1].set_title(r'$\gamma*W_2^2$ from $P$ to $Q$')
        else:
            ax[0].set_title(r'$\mathbb{E}_{Y\sim Q }[-\tilde{r}_0(T_{1:L}^{-1}(Y))]$')
            ax[1].set_title(r'$\gamma*W_2^2$ from $Q$ to $P$')
        ax[2].plot(losses[:,2], '-o')
        ax[2].set_title('Sum of two')
    
    ###### Visualize overall_progress_loss
    print(f'Visualize overall progress (on test data)')
    fig1, ax = plt.subplots(1, 2, figsize = (8, 4))
    losses = np.array(overall_progress_loss)
    ax[0].plot(losses[:,0], '-o', label = 'Estimated KL')
    if np.max(losses[:,1]) > 0:
        ax[0].plot(losses[:,1], '-o', label = 'True KL')
    ax[0].legend(fontsize = 14, loc = 'upper right')
    ax[1].plot(losses[:,2], '-o')
    if PtoQ:
        ax[0].set_title(r'$KL(P_{L}||Q)$')
        ax[1].set_title(r'$W_2^2$ from $P$ to $Q$')
    else:
        ax[0].set_title(r'$KL(\tilde{P}_0||P)$')
        ax[1].set_title(r'$W_2^2$ from $Q$ to $P$')


    ###### Check P->Q trajectory
    if PtoQ:
        print(f'Visualization P -> Z -> Q')
    else:
        print(f'Visualization Q -> Z -> P')
    with torch.no_grad():
        full_traj = flow_P_Q(FlowNet_P, FlowNet_Q, args_P, args_Q, Pdata_test, Qdata_test, 
                                    PtoQ = PtoQ, return_W2 = False)
        long_row = True if len(full_traj) <= 10 else False
        fig2 = visualize_trajectory(full_traj, args_P.num_blocks, args_Q.num_blocks, PtoQ = PtoQ, long_row = long_row,s=s)
    return full_traj, fig0, fig1, fig2


def visualize_trajectory(full_PQ_traj, num_blocks_P, num_blocks_Q, PtoQ = True, long_row = False,s=1):
    tot_len = len(full_PQ_traj)
    if long_row:
        max_per_row = tot_len
    else:
        max_per_row = int(np.ceil(tot_len/2))
    nrows = int(np.ceil(tot_len / max_per_row))
    fig, axs = plt.subplots(nrows, max_per_row, figsize=(max_per_row*3, nrows*3), sharex = True, sharey = True)
    if PtoQ:
        num_b_P = num_blocks_P
        num_b_Q = num_blocks_Q
    else:
        num_b_P = num_blocks_Q
        num_b_Q = num_blocks_P
    for i in range(tot_len):
        ax = axs.flatten()[i]
        samples = full_PQ_traj[i].cpu().numpy()
        ax.scatter(samples[:,0], samples[:,1], s=s)
        if i == 0:
            if PtoQ:
                ax.set_title(r'$X\sim P$')
            else:
                ax.set_title(r'$Y\sim Q$')
        elif i == tot_len - 1:
            if PtoQ:
                ax.set_title(r'$Y\sim Q$')
            else:
                ax.set_title(r'$X\sim P$')
        elif i <= num_b_P:
            if PtoQ:
                ax.set_title(f'(F) Block {i} from P to Z')
            else:
                ax.set_title(f'(F) Block {i} from Q to Z')
        else:
            if PtoQ:
                ax.set_title(f'(B) Block {num_b_Q - (i-num_b_P)+1} from Z to Q')
            else:
                ax.set_title(f'(B) Block {num_b_Q - (i-num_b_P)+1} from Z to P')
    # Remove unused axes
    for i in range(tot_len, max_per_row*nrows):
        fig.delaxes(axs.flatten()[i])
    fig.tight_layout()
    return fig

def rnet_visualize(rnet, flip, losses_per_flip, logit_arg, 
                   est_full, true_full, PtoQ = True,
                   plot_loss = True,s=1):
    if plot_loss:
        ###### Visualize this J1 many loss
        dict_name = 'r_lp1' if PtoQ else 'r_tilde0'
        fig, ax = plt.subplots(1, 1, figsize = (8, 4))
        ax.plot(losses_per_flip[f'Flip{flip}'][dict_name])
        if PtoQ:
            plt.title(f'Training loss of r_L+1 at flip {flip}')
        else:
            plt.title(f'Training loss of r_tilde0 at flip {flip}')
    ###### Visualize logit and P_L on top of Q
    return fig, logit_visualize_single(rnet, 
                                Xdim = 2, 
                                logit_arg = logit_arg,
                                pushed_data = est_full,
                                target_data = true_full,
                                PtoQ = PtoQ,s=s)

def logit_visualize_single(rnetPQ, Xdim = 2, logit_arg = None, 
                           pushed_data = None, target_data = None,
                           PtoQ = True,s=1):
    
    def get_xygrid(logit_arg):
        xmin, xmax, ymin, ymax = logit_arg
        xgrid = torch.linspace(xmin, xmax, 50)
        ygrid = torch.linspace(ymin, ymax, 50)
        xgrid, ygrid = torch.meshgrid(xgrid, ygrid)
        xinput = torch.stack([xgrid, ygrid], dim = 2).reshape(-1, 2).to(device)
        return xinput
    if Xdim == 2:
        xinput_PQ = get_xygrid(logit_arg)
        xinput_PQ_ = xinput_PQ.cpu().detach().numpy()
        fig, ax = plt.subplots(1, 2, figsize=(8,4), sharex = True, sharey = True)
        logit_PQ = rnetPQ(xinput_PQ)
        logit_PQ = logit_PQ.cpu().detach().numpy()
        # Use the smaller of the two absolute values for the colorbar
        v_val = min(np.abs(logit_PQ.min()), np.abs(logit_PQ.max()))
        sc = ax[0].scatter(xinput_PQ_[:,0], xinput_PQ_[:,1], c = logit_PQ, cmap = 'bwr', vmin = -v_val, vmax = v_val)
        ax[0].set_title(r'Logit by $r_{L+1}$')
        if PtoQ == False:
            ax[0].set_title(r'Logit by $\tilde{r}_0$')
        ax[0].grid()
        fig.colorbar(sc, ax=ax[0])
        plt_on_ax(ax[1], pushed_data, target_data, PtoQ, s=s)
    else:
        raise NotImplementedError
    return fig

def plt_on_ax(ax, est, true, PtoQ = True,s=1):
    est, true = est.cpu().detach().numpy(), true.cpu().detach().numpy()
    if PtoQ:
        ax.scatter(true[:,0], true[:,1], s=s, c='gray', label = r'$Y\sim Q$')
        ax.scatter(est[:,0], est[:,1], s=s, c='r', alpha = 0.5, label = r'$\hat{Y}\sim P_L$')
    else:
        ax.scatter(true[:,0], true[:,1], s=s, c='gray', label = r'$X\sim P$')
        ax.scatter(est[:,0], est[:,1], s=s, c='r', alpha = 0.5, label = r'$\hat{X} \sim \tilde{P}_0$')
    ax.legend(fontsize = 16, markerscale = 3, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                fancybox=True, shadow=True, ncol=2)
    ax.grid()

def overlay_on_true(full_PQ_traj, full_QP_traj, s=1):
    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    plt_on_ax(ax[0], full_PQ_traj[-2], full_PQ_traj[-1], PtoQ = True,s=s)
    plt_on_ax(ax[1], full_QP_traj[-2], full_QP_traj[-1], PtoQ = False,s=s)
    fig.tight_layout()
    return fig


''' Part V: Miscellaneous '''

def get_rnet_loader(FlowNet_P, FlowNet_Q, args_P, args_Q, 
                    Pdata, Qdata, batch_size_rnet, PtoQ = True):
    with torch.no_grad():
        if PtoQ:
            _, Dataest_full = flow_P_Q(FlowNet_P, FlowNet_Q, args_P, args_Q, 
                                    Pdata, Qdata, PtoQ = True, return_W2 = True)
            target_data = Qdata
        else:
            _, Dataest_full = flow_P_Q(FlowNet_P, FlowNet_Q, args_P, args_Q, 
                                    Pdata, Qdata, PtoQ = False, return_W2 = True)
            target_data = Pdata
    batch_idx_rnet = utils.get_batches(Dataest_full, bsize=batch_size_rnet)
    dloader = utils.SimpleDloader(Dataest_full, target_data, batch_idx_rnet)
    return dloader, Dataest_full

def get_true_KL_func(FlowNet_P, FlowNet_Q, args_P, args_Q,
                    Pdata_test, Qdata_test,
                    mvns, PtoQ = True):
    def analytic_logp(xin, xout, mvns, PtoQ = True):
        # By default, log_q always is the target
        mvn00, mvn01, mvn10, mvn11, mvn12 = mvns
        if PtoQ:
            log_p = torch.log(0.5 * torch.exp(mvn00.log_prob(xin)) 
                            + 0.5 * torch.exp(mvn01.log_prob(xin)))
            log_q = torch.log((torch.exp(mvn10.log_prob(xout))
                                + torch.exp(mvn11.log_prob(xout))
                                + torch.exp(mvn12.log_prob(xout)))/3)
        else:
            log_p = torch.log((torch.exp(mvn10.log_prob(xin))
                                + torch.exp(mvn11.log_prob(xin))
                                + torch.exp(mvn12.log_prob(xin)))/3)
            log_q = torch.log(0.5 * torch.exp(mvn00.log_prob(xout)) 
                            + 0.5 * torch.exp(mvn01.log_prob(xout)))
        return log_p, log_q
    num_blocks_P, num_blocks_Q = args_P.num_blocks, args_Q.num_blocks
    if PtoQ:
        input = Pdata_test
        x = Pdata_test
        est_ls_P, dlogpx_ls_P = utils.map_for_or_back(input, num_blocks_P, 
                                                        FlowNet_P, args_P, reverse = False, 
                                                        return_dlogpx = True,
                                                        test = False)
        est_ls, dlogpx_ls_Q = utils.map_for_or_back(est_ls_P[-1], num_blocks_Q, 
                                                        FlowNet_Q, args_Q, reverse = True, 
                                                        return_dlogpx = True,
                                                        test = False)
    else:
        input = Qdata_test
        x = Qdata_test
        est_ls_Q, dlogpx_ls_Q = utils.map_for_or_back(input, num_blocks_Q, 
                                                        FlowNet_Q, args_Q, reverse = False, 
                                                        return_dlogpx = True,
                                                        test = False)
        est_ls, dlogpx_ls_P = utils.map_for_or_back(est_ls_Q[-1], num_blocks_P, 
                                                        FlowNet_P, args_P, reverse = True, 
                                                        return_dlogpx = True,
                                                        test = False)
    dlogpx = dlogpx_ls_P.sum(axis = 0) + dlogpx_ls_Q.sum(axis = 0)
    logpk, logqk = analytic_logp(x, est_ls[-1], mvns, PtoQ = PtoQ)
    print(f'logp(x) = {logpk.mean().item():.2f}, logq(x) = {logqk.mean().item():.2f}, dlogpx = {dlogpx.mean().item():.2f}')
    logpk += dlogpx
    return (logpk.mean() - logqk.mean()).item()