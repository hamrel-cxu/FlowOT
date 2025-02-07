import src.refinement as jt
import src.nets as nets
import src.data as data
import os
import time
import importlib as ipb
from argparse import Namespace
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_initial_flow(PtoZ = True):
    initializer = Namespace()
    args = Namespace(
        word='moon_to_checkerboard',
        Xdim = 2,
        train_data_size = 100000,
        netname='ODEnet',
        hidden_dim_str = '256-256', # Hidden dims
        activation = 'softplus', # Activation function
        use_NeuralODE = True, # During back prop
        num_blocks = 4,
        T0 = 0.25,
        int_mtd = 'RK4',
        num_int_pts = 3, # Esp. at small h
    )   
    args.T_ls = [args.T0]*args.num_blocks
    args.div_bf = True
    args.num_e = 3; args.fix_e_ls = True; args.rtol = 1e-5; args.atol = 1e-5
    args.test_data_size = args.train_data_size
    initializer.args = args
    odefunc = nets.ODEnet(args, layer_type = 'ODE')
    initializer.model = nets.CNF(odefunc).to(device)
    suffix = 'PtoZ' if PtoZ else 'QtoZ'
    filepath = os.path.join(checkpt_dir, f'ckpt_{suffix}.pth')
    checkpt = torch.load(filepath)
    initializer.model.load_state_dict(checkpt['model'])
    return initializer

def get_refinement_args():
    rnet_args = Namespace(
        Xdim = 2,
        classifier_hidden_dim_str = '312-312-312',
        classifier_activation = 'softplus'
    )

    meta_args = Namespace(
        Xdim = 2,
        batch_size_rnet = 200, 
        batch_size_flow = 2000,
        rnet_lr = 1e-3,
        flow_lr = 5e-5,
        tot_flips = 2,
        J1 = 300, # How many epochs to train ratio nets in each flip
        J1_sub = 4, # How many epochs to train ratio nets in each flow net update
        J2 = 50, # How many epochs to train odeblocks in each flip
        Tfor_J2 = 1, # Starting epoch for training T_{1:L}
        Tback_J2 = 1, # Starting epoch for training T_{L:1}
        log_intervals = 10, # How often to examine ODEblock movement and losses
        get_true_KL = False, # If get true KL divergence
        gamma = 0.5, # W2 regularization factor
        save_statedict = True, # If save state_dict of flow net and rnets after every flip
        #####################
        # Default true, to avoid retraining nets at a specific flip
        save_rLp1_statedict = True, 
        save_Tfor_statedict = True,
        save_rtilde0_statedict = True,
        save_Tback_statedict = True,
        #####################
        statedict_path = os.path.join(checkpt_dir, 'ckpt_PtoQ.pth'), # Savename of the state_dict
        load_existing_statedict = True, # If load existing state_dict of flow net and rnets
        flip_start = 1,
        train_rlp1 = True, # Default always true
        train_rtilde0 = True, # Default always true
    ) 
    return rnet_args, meta_args

if __name__ == '__main__':
    ipb.reload(jt)
    ipb.reload(nets)
    master_dir = 'results'
    checkpt_dir = 'checkpoints'
    os.makedirs(master_dir, exist_ok = True)
    os.makedirs(checkpt_dir, exist_ok = True)
    c = 0.5 # Plot a bit larger than the data domain
    PQ_logit_arg = [-3.75-c, 3.75+c, -3.75-c, 3.75+c]
    QP_logit_arg = [-3.75-c, 3.75+c, -3.75-c, 3.75+c]
    init_P, init_Q = get_initial_flow(PtoZ = True), get_initial_flow(PtoZ = False)
    rnet_args, meta_args = get_refinement_args()
    # Get data
    args = init_P.args
    tot_train_samples = 100000; args.train_data_size = tot_train_samples
    tot_test_samples = 200000; args.test_data_size = tot_test_samples
    Pdata, Qdata = data.data_generator(args, train = True)
    Pdata_test, Qdata_test = data.data_generator(args, train = False)
    ######### Define some storing stuff
    # For r_L+1 and tilde{r}_0, it stores vectors of length J1
    # For T_{1:L} and T_{1:L}^{-1}, it store matrices of size (J2, J2) (for KL and W2)
    losses_per_flip = {} 
    # It stores a long matrix of size tot_flips*(J2, J2, J2)
    # Which are [est_KL, trueKL, W2^2] on TEST data
    overall_progress_loss_PQ = [] 
    overall_progress_loss_QP = []
    s=0.01
    ######### 0. Initialization 
    lrs = [meta_args.rnet_lr, meta_args.flow_lr]
    FlowNet_P, FlowNet_Q, args_P, args_Q, optimizer_flow, rnetPQ, rnetQP,  optimizer_logitP, optimizer_logitQ = jt.initialization_nets(rnet_args, lrs, init_P, init_Q)
    print(f'###### Initial trajectory before joint training ######')
    with torch.no_grad():
        full_PQ_traj = jt.flow_P_Q(FlowNet_P, FlowNet_Q, args_P, args_Q, Pdata_test, Qdata_test, 
                                    PtoQ = True, return_W2 = False)
        W2_PtoQ, _ = jt.flow_P_Q(FlowNet_P, FlowNet_Q, args_P, args_Q, Pdata_test, Qdata_test, 
                                    PtoQ = True, return_W2 = True)
        fig_PQ = jt.visualize_trajectory(full_PQ_traj, args_P.num_blocks, args_Q.num_blocks, PtoQ = True, long_row = True,s=s)
        fig_PQ.savefig(f'{master_dir}/initial_traj_PtoQ.png', dpi=100, bbox_inches='tight')
        full_QP_traj = jt.flow_P_Q(FlowNet_P, FlowNet_Q, args_P, args_Q, Pdata_test, Qdata_test, 
                                    PtoQ = False, return_W2 = False)
        W2_QtoP, _ = jt.flow_P_Q(FlowNet_P, FlowNet_Q, args_P, args_Q, Pdata_test, Qdata_test, 
                                    PtoQ = False, return_W2 = True)
        fig_QP = jt.visualize_trajectory(full_QP_traj, args_P.num_blocks, args_Q.num_blocks, PtoQ = False, long_row = True,s=s)
        fig_QP.savefig(f'{master_dir}/initial_traj_QtoP.png', dpi=100, bbox_inches='tight')
        # Scatter P_L on top of Q and \tilde{P} on top of P 
        # To be compared with what we get at the END
        fig_overlay = jt.overlay_on_true(full_PQ_traj, full_QP_traj,s=s)
        fig_overlay.savefig(f'{master_dir}/fig_overlay.png', dpi=100, bbox_inches='tight')
        print(f'Initial trajectory W2^2 movement from P -> Q is {W2_PtoQ.item()}')
        print(f'Initial trajectory W2^2 movement from Q -> P is {W2_QtoP.item()}')
    ####################################### Override initial with current files
    if meta_args.load_existing_statedict and os.path.exists(meta_args.statedict_path):  
        meta_args, losses_per_flip, overall_progress_loss_PQ, overall_progress_loss_QP = jt.load_or_save_checkpt(meta_args.statedict_path, 
                                                                                                            FlowNet_P, FlowNet_Q, rnetPQ, rnetQP, 
                                                                                                            optimizer_flow, optimizer_logitP, optimizer_logitQ,
                                                                                                            load = True)
        meta_args.load_existing_statedict = True
    PQ_loader, Qest_full = jt.get_rnet_loader(FlowNet_P, FlowNet_Q, args_P, args_Q, 
                                   Pdata, Qdata, meta_args.batch_size_rnet, 
                                   PtoQ = True)
    ######### Start alternate joint training over flips
    s=0.01
    fstart = meta_args.flip_start
    print('###### Start joint training ######')
    for flip in range(meta_args.flip_start, meta_args.tot_flips + 1):
        if flip > fstart:
            # Reduce learning rate by 50% after every flip
            optimizer_flow.param_groups[0]['lr'] /= 2
            optimizer_logitP.param_groups[0]['lr'] /= 2
            optimizer_logitQ.param_groups[0]['lr'] /= 2
        losses_rlp1, losses_Tfor, losses_rtilde0, losses_Tback = [], [], [], []
        if f'Flip{flip}' not in losses_per_flip.keys():
            losses_per_flip[f'Flip{flip}'] = {}
        ######### 1. Training r_{L+1} and visualize
        PQ_loader, Qest_full = jt.get_rnet_loader(FlowNet_P, FlowNet_Q, args_P, args_Q, 
                                    Pdata, Qdata, meta_args.batch_size_rnet, 
                                    PtoQ = True)
        start = time.time()
        if meta_args.train_rlp1:
            for _ in range(meta_args.J1):
                losses_per_flip = jt.train_rnet(rnetPQ, rnetQP, optimizer_logitP, optimizer_logitQ,
                                                PQ_loader, Qest_full, Qdata,
                                                losses_per_flip, flip,
                                                PtoQ = True)
        print(f'Done training r_L+1 at flip {flip} for {meta_args.J1} epochs')
        print(f'Time taken: {time.time() - start} secs')
        ################## Save for efficient tuning of FlowNet T_{1:L}
        if meta_args.save_rLp1_statedict and meta_args.train_rlp1:
            # This is because we may terminate next training if wrong, and need not train this again
            meta_args.train_rlp1 = False 
            jt.load_or_save_checkpt(meta_args.statedict_path, FlowNet_P, FlowNet_Q, rnetPQ, rnetQP, 
                                    optimizer_flow, optimizer_logitP, optimizer_logitQ,
                                    losses_per_flip, overall_progress_loss_PQ, 
                                    overall_progress_loss_QP, meta_args,
                                    flip, load = False)
        #
        #
        #
        #
        # ######### Separator for easy debugging by section
        #
        #
        #
        #
        ######### 2. Training T_{1:L} and visualize
        start = time.time()
        for epoch in range(meta_args.Tfor_J2, meta_args.J2):
            start_e = time.time()
            losses_per_flip, overall_progress_loss_PQ = jt.train_PQ_flow(FlowNet_P, FlowNet_Q, args_P, args_Q, Pdata, Qdata,
                                                                    Pdata_test, Qdata_test,
                                                                    optimizer_flow, rnetPQ, rnetQP,
                                                                    losses_per_flip, overall_progress_loss_PQ, flip, meta_args,
                                                                    PtoQ = True)
            print(f'Epoch {epoch+1} took {time.time() - start_e} secs')
            # After epoch of training T_{1:L}, update r_{L+1} with X epoch of training
            PQ_loader, Qest_full = jt.get_rnet_loader(FlowNet_P, FlowNet_Q, args_P, args_Q, 
                                    Pdata, Qdata, meta_args.batch_size_rnet, 
                                    PtoQ = True)
            for _ in range(meta_args.J1_sub):
                losses_per_flip = jt.train_rnet(rnetPQ, rnetQP, optimizer_logitP, optimizer_logitQ,
                                                PQ_loader, Qest_full, Qdata,
                                                losses_per_flip, flip,
                                                PtoQ = True)
            if (epoch+1) % meta_args.log_intervals == 0 or epoch == meta_args.J2-1:
                ################## Visualize losses and movement
                print(f'Flip {flip}, Epoch {epoch}')
                _, fig0, fig1, fig2 = jt.visualize_PQ_flow(FlowNet_P, FlowNet_Q, args_P, args_Q, 
                                        Pdata_test, Qdata_test, flip,
                                        losses_per_flip, overall_progress_loss_PQ,
                                        PtoQ = True,s=s)
                fig0.savefig(f'{master_dir}/PQ_trainloss.png', dpi=100, bbox_inches='tight')
                fig1.savefig(f'{master_dir}/PQ_overallprogress.png', dpi=100, bbox_inches='tight')
                fig2.savefig(f'{master_dir}/PQ_traj.png', dpi=100, bbox_inches='tight')
                ################## Visualize logit, as our T is trained to push to red parts of r_{L+1}
                # We also can check loss of r_{L+1}, which may fluctuate because T_{1:L} is updated,
                # which makes P_L to be different at every epoch
                fig_loss, fig_logit =jt.rnet_visualize(rnetPQ, flip, losses_per_flip, PQ_logit_arg, Qest_full, Qdata, PtoQ = True,s=s)
                fig_loss.savefig(f'{master_dir}/rnetPQ_loss.png', dpi=100, bbox_inches='tight')
                fig_logit.savefig(f'{master_dir}/rnetPQ_logit.png', dpi=100, bbox_inches='tight')
            meta_args.Tfor_J2 += 1
            if meta_args.save_Tfor_statedict:
                jt.load_or_save_checkpt(meta_args.statedict_path, FlowNet_P, FlowNet_Q, rnetPQ, rnetQP, 
                                    optimizer_flow, optimizer_logitP, optimizer_logitQ,
                                    losses_per_flip, overall_progress_loss_PQ, 
                                    overall_progress_loss_QP, meta_args,
                                    flip, load = False)
        print(f'Done training T_1:L at flip {flip} for {meta_args.J2} epochs')
        print(f'Time taken: {(time.time() - start)/60} mins')
        #
        #
        #
        #
        # ######### Separator for easy debugging by section
        #
        #
        #
        #
        ######### 3. Training tilde{r}_0 and visualize
        start = time.time()
        QP_loader, Pest_full = jt.get_rnet_loader(FlowNet_P, FlowNet_Q, args_P, args_Q,
                                        Pdata, Qdata, meta_args.batch_size_rnet, 
                                        PtoQ = False)
        if meta_args.train_rtilde0:
            for _ in range(meta_args.J1):
                losses_per_flip = jt.train_rnet(rnetPQ, rnetQP, optimizer_logitP, optimizer_logitQ,
                                                QP_loader, Pest_full, Pdata,
                                                losses_per_flip, flip,
                                                PtoQ = False)
        print(f'Done training rtilde_0 at flip {flip} for {meta_args.J1} epochs')
        print(f'Time taken: {time.time() - start} secs')
        if meta_args.save_rtilde0_statedict and meta_args.train_rtilde0:
            # This is because we may terminate next training if wrong, and need not train this again
            meta_args.train_rtilde0 = False 
            jt.load_or_save_checkpt(meta_args.statedict_path, FlowNet_P, FlowNet_Q, rnetPQ, rnetQP, 
                                    optimizer_flow, optimizer_logitP, optimizer_logitQ,
                                    losses_per_flip, overall_progress_loss_PQ, 
                                    overall_progress_loss_QP, meta_args,
                                    flip, load = False)
        #
        #
        #
        #
        # ######### Separator for easy debugging by section
        #
        #
        #
        #
        ######### 4. Training T_{1:L}^{-1} and visualize
        start = time.time()
        for epoch in range(meta_args.Tback_J2, meta_args.J2):
            start_e = time.time()
            losses_per_flip, overall_progress_loss_QP = jt.train_PQ_flow(FlowNet_P, FlowNet_Q, args_P, args_Q, Pdata, Qdata,
                                                                    Pdata_test, Qdata_test,
                                                                    optimizer_flow, rnetPQ, rnetQP,
                                                                    losses_per_flip, overall_progress_loss_QP, flip, meta_args,
                                                                    PtoQ = False)
            print(f'Epoch {epoch+1} took {time.time() - start_e} secs')
            # After epoch of training T_{1:L}, update r_{L+1} with X epoch of training
            QP_loader, Pest_full = jt.get_rnet_loader(FlowNet_P, FlowNet_Q, args_P, args_Q,
                                        Pdata, Qdata, meta_args.batch_size_rnet, 
                                        PtoQ = False)
            for _ in range(meta_args.J1_sub):
                losses_per_flip = jt.train_rnet(rnetPQ, rnetQP, optimizer_logitP, optimizer_logitQ,
                                                    QP_loader, Pest_full, Pdata,
                                                    losses_per_flip, flip,
                                                    PtoQ = False)
            if (epoch+1) % meta_args.log_intervals == 0 or epoch == meta_args.J2-1:
                ################## Visualize losses and movement
                print(f'Flip {flip}, Epoch {epoch}')
                _, fig0, fig1, fig2 = jt.visualize_PQ_flow(FlowNet_P, FlowNet_Q, args_P, args_Q, 
                                        Pdata_test, Qdata_test, flip,
                                        losses_per_flip, overall_progress_loss_QP,
                                        PtoQ = False,s=s)
                fig0.savefig(f'{master_dir}/QP_trainloss.png', dpi=100, bbox_inches='tight')
                fig1.savefig(f'{master_dir}/QP_overallprogress.png', dpi=100, bbox_inches='tight')
                fig2.savefig(f'{master_dir}/QP_traj.png', dpi=100, bbox_inches='tight')
                ################## Visualize logit, as our T^{-1} is trained to push to red parts of \tilde{r}_0
                # We also can check loss of \tilde{r}_0, which may fluctuate because T_{1:L}^{-1} is updated,
                # which makes \tilde{P}_0 to be different at every epoch
                fig_loss, fig_logit = jt.rnet_visualize(rnetQP, flip, losses_per_flip, QP_logit_arg, Pest_full, Pdata, PtoQ = False,s=s)
                fig_loss.savefig(f'{master_dir}/rnetQP_loss.png', dpi=100, bbox_inches='tight')
                fig_logit.savefig(f'{master_dir}/rnetQP_logit.png', dpi=100, bbox_inches='tight')
            meta_args.Tback_J2 += 1
            if meta_args.save_Tback_statedict:
                jt.load_or_save_checkpt(meta_args.statedict_path, FlowNet_P, FlowNet_Q, rnetPQ, rnetQP, 
                                    optimizer_flow, optimizer_logitP, optimizer_logitQ,
                                    losses_per_flip, overall_progress_loss_PQ, 
                                    overall_progress_loss_QP, meta_args,
                                    flip, load = False)
        print(f'Done training inv(T_1:L) at flip {flip} for {meta_args.J2} epochs')
        print(f'Time taken: {(time.time() - start)/60} mins')
        #
        #
        #
        #
        # ######### Separator for easy debugging by section
        #
        #
        #
        #
        ######### 5. Save models and losses and args, the same way we load above
        if meta_args.save_statedict:
            # Flip to True, o/w next flip has error
            meta_args.train_rlp1 = True
            meta_args.train_rtilde0 = True
            meta_args.Tfor_J2 = 1
            meta_args.Tback_J2 = 1
            if flip == 1:
                # For later flips, just train 50% of original epochs in first flip
                meta_args.J1 = int(meta_args.J1/2)
            jt.load_or_save_checkpt(meta_args.statedict_path, FlowNet_P, FlowNet_Q, rnetPQ, rnetQP, 
                                    optimizer_flow, optimizer_logitP, optimizer_logitQ,
                                    losses_per_flip, overall_progress_loss_PQ, overall_progress_loss_QP, 
                                    meta_args, flip+1, load = False)
    print('###### Final visualize ######')
    plot_rloss = True
    plot_Tloss = True
    plot = True
    s=0.01
    flip = 1 # Placeholder
    #### T_{1:L}
    full_PQ_traj, fig0, fig1, fig2 = jt.visualize_PQ_flow(FlowNet_P, FlowNet_Q, args_P, args_Q, 
                                    Pdata_test, Qdata_test, flip,
                                    losses_per_flip, overall_progress_loss_PQ,
                                    PtoQ = True,
                                    return_full_traj = True,
                                    plot_loss = plot_Tloss,s=s)
    fig0.savefig(f'{master_dir}/PQ_trainloss.png', dpi=100, bbox_inches='tight')
    fig1.savefig(f'{master_dir}/PQ_overallprogress.png', dpi=100, bbox_inches='tight')
    fig2.savefig(f'{master_dir}/PQ_traj.png', dpi=100, bbox_inches='tight')
    #### T_{1:L}^{-1}
    full_QP_traj, fig0, fig1, fig2 = jt.visualize_PQ_flow(FlowNet_P, FlowNet_Q, args_P, args_Q, 
                                    Pdata_test, Qdata_test, flip,
                                    losses_per_flip, overall_progress_loss_QP,
                                    PtoQ = False,
                                    return_full_traj = True,
                                    plot_loss = plot_Tloss,s=s)
    fig0.savefig(f'{master_dir}/QP_trainloss.png', dpi=100, bbox_inches='tight')
    fig1.savefig(f'{master_dir}/QP_overallprogress.png', dpi=100, bbox_inches='tight')
    fig2.savefig(f'{master_dir}/QP_traj.png', dpi=100, bbox_inches='tight')
    #### Scatter P_L on top of Q and \tilde{P} on top of P 
    W2_PtoQ, _ = jt.flow_P_Q(FlowNet_P, FlowNet_Q, args_P, args_Q, Pdata_test, Qdata_test, 
                                PtoQ = True, return_W2 = True)
    W2_QtoP, _ = jt.flow_P_Q(FlowNet_P, FlowNet_Q, args_P, args_Q, Pdata_test, Qdata_test, 
                                PtoQ = False, return_W2 = True)
    # Scatter P_L on top of Q and \tilde{P} on top of P 
    # To be compared with what we get at the END
    fig_overlay = jt.overlay_on_true(full_PQ_traj, full_QP_traj,s=s)
    fig_overlay.savefig(f'{master_dir}/fig_overlay.png', dpi=100, bbox_inches='tight')
    print(f'After joint training, W2^2 movement from P -> Q is {W2_PtoQ.item()}')
    print(f'After joint training, W2^2 movement from Q -> P is {W2_QtoP.item()}')

    # Get data for infinitesimal DRE
    print('###### Get data for infinitesimal DRE ######')
    tot_train_samples = 100000; args.train_data_size = tot_train_samples
    tot_test_samples = 50000; args.test_data_size = tot_test_samples
    Pdata, Qdata = data.data_generator(args, train = True)
    Pdata_test, Qdata_test = data.data_generator(args, train = False)
    full_PQ_traj = jt.flow_P_Q(FlowNet_P, FlowNet_Q, args_P, args_Q, Pdata, Qdata, 
                                    PtoQ = True, return_W2 = False)
    data_dict = {'data':full_PQ_traj}
    torch.save(data_dict, os.path.join(checkpt_dir, 'moon_to_checkerboard_data.pt'))
    full_PQ_traj_test = jt.flow_P_Q(FlowNet_P, FlowNet_Q, args_P, args_Q, Pdata_test, Qdata_test, 
                                    PtoQ = True, return_W2 = False)
    data_dict_test = {'data':full_PQ_traj_test}
    torch.save(data_dict_test, os.path.join(checkpt_dir, 'moon_to_checkerboard_data_test.pt'))