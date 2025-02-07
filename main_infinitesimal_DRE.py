import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from tqdm import tqdm
from argparse import Namespace
import multiprocessing
from sklearn.neighbors import KernelDensity
from sklearn.metrics import pairwise_distances

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
            dims = (input_dim+1,) + tuple(hidden_dims) + (1,)
            layers_in_block = []
            for (in_dim, out_dim) in zip(dims[:-1], dims[1:]):
                layers_in_block.append(nn.Linear(in_dim, out_dim))
                if out_dim != 1:
                    layers_in_block.append(activation_dict[classifier_activation])
            self.classifier = nn.Sequential(*layers_in_block)
        def forward(self, x, t):
            tt = torch.ones_like(x[:,:1]) * t
            ttx = torch.cat([tt, x], 1)
            ttx = self.classifier(ttx)
            return ttx

    classifier = LinearClassifier(args).to(device)
    return classifier
        
def rnet_integral(score_model, x, t, num_int_pts = 1):
    # Runge-Kutta 3/8 Method
    # http://www.mymathlib.com/diffeq/runge-kutta/runge_kutta_3_8.html
    # t here is [t0, t1] for how long to integrate
    # Here, score model: (x,t) -> score \in R, where x is the input and t is the time
    outputs = [score_model(x, t[0])] 
    h = t[1] - t[0]
    if num_int_pts > 1:
        h = (t[1] - t[0]) / num_int_pts
    for i in range(num_int_pts):
        t_now = t[0] + i*h
        # print(f'Starting at {t_now} and ending at {t_now + h} with step size {h}.')
        k1 = score_model(x, t_now)
        k2 = score_model(x, t_now + h/3)
        k3 = score_model(x, t_now + 2*h/3)
        k4 = score_model(x, t_now + h)
        if i > 0:
            # This is because we break the integral into smaller pieces, so we need to 
            # add the previous output to the current output for cumulative integration
            outputs.append(outputs[-1] + h/8 * (k1 + 3*k2 + 3*k3 + k4))
        else:
            outputs.append(h/8 * (k1 + 3*k2 + 3*k3 + k4))
    return torch.stack(outputs)

# For continuous time training
def cont_t_train(train_loader, time_ls, rnet, optimizer, num_int_pts = 1):
    '''
        # train_loader consistent of batches of (x0, ..., x_{L+1})
        # time_ls = [[t_{k-1}, t_k]], k=1,...,L+1
        # rnet is a continuous time score function
    '''
    softplus = torch.nn.Softplus(beta = 1)
    loss_tot = []
    for batch in train_loader:
        # Here, batch consists of ALL samples from P0, P1, ..., P_L+1
        optimizer.zero_grad()
        loss_batch = 0
        for i, t in enumerate(time_ls):
            x, y = batch[i], batch[i+1]
            t = torch.tensor(t).to(device)
            output_xy = rnet_integral(rnet, x, t, num_int_pts)
            output_yx = rnet_integral(rnet, y, torch.flip(t, [0]), num_int_pts)
            loss_X = softplus(output_xy[-1]).mean()
            loss_Y = softplus(output_yx[-1]).mean()
            loss_batch += loss_X + loss_Y
        loss_batch.backward()
        optimizer.step()
        loss_tot.append(loss_batch.item())
    return np.mean(loss_tot)

### Save or load from path
def save_or_load(save = False, load = True, 
                 filepath = None, score_model = None, 
                 optimizer = None, loss_score = None):
    if save:
        save_obj = {'score_model': score_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss_score': loss_score}
        torch.save(save_obj, filepath)
    if load:
        print(f'### Load rnets to evaluate or resume')
        save_obj = torch.load(filepath)
        score_model.load_state_dict(save_obj['score_model'])
        optimizer.load_state_dict(save_obj['optimizer'])
        loss_score = save_obj['loss_score']
        return loss_score
            
### Visualize
def visualize_rnets_on_data(score_model, full_PQ_traj, all_t, s=1):
    num_nets = len(all_t)
    fig, ax = plt.subplots(1,num_nets, figsize = (3*num_nets,3))
    start = 1
    max_vals = min(10000, len(full_PQ_traj[0]))
    indices = torch.randperm(len(full_PQ_traj[0]))[:max_vals] # Max 10000 points
    for a, t_now in zip(ax, all_t):
        xinput_PQ = full_PQ_traj[start-1][indices]
        xinput_PQ_ = xinput_PQ.cpu().detach().numpy()
        with torch.no_grad():
            logit_PQ = rnet_integral(score_model, xinput_PQ, t_now)[-1]
        logit_PQ = logit_PQ.cpu().numpy()
        # Use the smaller of the two absolute values for the colorbar
        v_val = min(np.abs(logit_PQ.min()), np.abs(logit_PQ.max()))
        sc = a.scatter(xinput_PQ_[:,0], xinput_PQ_[:,1], c = logit_PQ, cmap = 'bwr', vmin = -v_val, vmax = v_val, s=s)
        fig.colorbar(sc, ax=a)
        t_now_ = [f'{t:.2f}' for t in t_now]
        a.set_title(f'Logit at t = {t_now_}')
        start += 1
        a.grid()
    fig.tight_layout()
    fig.savefig('results/DRE_intermediate_logit.png', dpi=100, bbox_inches='tight', pad_inches=0.02)
    plt.close()

def get_KDE_estimator(X):
    ## Get bandwidth
    # Compute the pairwise distances between all points in the dataset
    idx = np.random.choice(X.shape[0], min(10000, X.shape[0]), replace=False)
    distances = pairwise_distances(X[idx], metric='euclidean')
    # Take the upper triangle of the distance matrix (excluding the diagonal) since it's symmetric
    upper_triangle_distances = distances[np.triu_indices_from(distances, k=1)]
    # Calculate the median distance
    median_distance = np.median(upper_triangle_distances)
    ## Get KDE
    kde = KernelDensity(bandwidth = 0.1*median_distance, rtol = 0.1, atol = 0.1) # Same bandwidth as in JKO-iFlow (0.1*median bandwidth; see rose)
    kde.fit(X)
    return kde

def parrallel_score_samples(kde, samples, thread_count=int(0.875 * multiprocessing.cpu_count())):
    with multiprocessing.Pool(thread_count) as p:
        return np.concatenate(p.map(kde.score_samples, np.array_split(samples, thread_count)))

def visualize_rnets_on_PQ(score_model, full_PQ_traj, s=1):
    plot_KDE = False # If True, plot KDE-then-DRE results here.
    if plot_KDE:
        fig, ax = plt.subplots(1,4, figsize = (16, 4))
    else:
        fig, ax = plt.subplots(1,3, figsize = (12, 4))
    max_vals = min(10000, len(full_PQ_traj[0]))
    indices = torch.randperm(len(full_PQ_traj[0]))[:max_vals] # Max 10000 points
    xinput_P = full_PQ_traj[0][indices] # Data from Q
    xinput_P_ = xinput_P.cpu().detach().numpy()
    xinput_Q = full_PQ_traj[-1][indices] # Data from Q
    xinput_Q_ = xinput_Q.cpu().detach().numpy()
    xinput_PQ = torch.cat([xinput_P, xinput_Q], 0)
    xinput_PQ_ = xinput_PQ.cpu().detach().numpy()
    with torch.no_grad():
        logit_PQ = rnet_integral(score_model, xinput_PQ, t = [0,1], num_int_pts = 9)[-1]
    logit_PQ = logit_PQ.cpu().numpy()
    ## Plot
    ax[0].scatter(xinput_P_[:, 0], xinput_P_[:, 1], s=s)
    ax[0].set_title(r'Data from $P$')
    ax[1].scatter(xinput_Q_[:, 0], xinput_Q_[:, 1], s=s)
    ax[1].set_title(r'Data from $Q$')
    def plot_dre(x, logit, ax):
        # Use the smaller of the two absolute values for the colorbar
        v_val = min(np.abs(logit.min()), np.abs(logit.max()))
        sc = ax.scatter(x[:,0], x[:,1], c = logit, cmap = 'bwr', vmax = v_val, s=s)
        fig.colorbar(sc, ax=ax)
    # Ours
    plot_dre(x = xinput_PQ_, logit = logit_PQ, ax = ax[2])
    ax[2].set_title(r'Infinitesimal DRE: $\log q(x) - \log p(x)$')
    if plot_KDE:
        ax[2].set_title(r'Ours: $\log q(x) - \log p(x)$')
        # Aside: add the DRE via fitting KDE
        logit_PQ_kde = parrallel_score_samples(kde_Q, xinput_PQ_) - parrallel_score_samples(kde_P, xinput_PQ_)
        plot_dre(x = xinput_PQ_, logit = logit_PQ_kde, ax = ax[3])
        ax[3].set_title(r'KDE: $\log q(x) - \log p(x)$')
    for a in ax.flatten():
        a.grid()
    fig.tight_layout()
    fig.savefig('results/DRE_PQ_logit.png', dpi=100, bbox_inches='tight', pad_inches=0.02)
    plt.close()

if __name__ == '__main__':
    dir = 'checkpoints'
    data_file = torch.load(os.path.join(dir, 'moon_to_checkerboard_data.pt'))
    full_PQ_traj = data_file['data']
    data_loaded = torch.load(os.path.join(dir, 'moon_to_checkerboard_data_test.pt'))
    full_PQ_traj_test = data_loaded['data']
    ### (Can delete) DRE via KDE for comparison
    xinput_P_ = full_PQ_traj[0].cpu().detach().numpy()
    xinput_Q_ = full_PQ_traj[-1].cpu().detach().numpy()
    kde_P = get_KDE_estimator(xinput_P_)
    kde_Q = get_KDE_estimator(xinput_Q_)
    ###
    bsize = 500
    train_loader_full = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*full_PQ_traj),
                                                batch_size=bsize, shuffle=True)
    # Score net initialization
    rnet_args = Namespace(
        Xdim = 2,
        classifier_hidden_dim_str = '312-312-312',
        classifier_activation = 'softplus'
    )
    score_model = def_classifier_net(rnet_args)
    optimizer_score = torch.optim.Adam(score_model.parameters(), lr=1e-3)
    # Time ls [t_{k-1}, t_k], k=1,...,L+1
    t_discretize = torch.linspace(0, 1, len(full_PQ_traj)).to(device)
    all_t = []
    for t0, t1 in zip(t_discretize[:-1], t_discretize[1:]):
        all_t.append([t0.item(), t1.item()])
    print(score_model)
    # Training, with saving option
    ## Hyperparameters
    num_int_pts = 1 # Namely, how far to break up the integral. For harder examples, can increase this
    num_epochs = 500
    log_freq = 25
    load = True
    filepath = os.path.join(dir, 'moon_to_checkerboard_score_net.pt')
    #########
    ## Start training
    if load and os.path.exists(filepath):
        loss_score = save_or_load(save = False, load = load,
                            filepath = filepath, score_model = score_model,
                            optimizer = optimizer_score, loss_score = None)
    else:
        loss_score = []
    epoch_now = len(loss_score)
    print(f'### Start training score net at epoch {epoch_now} out of {num_epochs}')
    for enow in tqdm(range(epoch_now, num_epochs), position=0, leave=True):
        start = time.time()
        loss_score.append(cont_t_train(train_loader_full, 
                                        time_ls = all_t, 
                                        rnet = score_model, 
                                        optimizer = optimizer_score,
                                        num_int_pts = num_int_pts))
        if enow % log_freq == 0 or enow == num_epochs-1:
            save_or_load(save = True, load = False,
                        filepath = filepath, score_model = score_model,
                        optimizer = optimizer_score, loss_score = loss_score)
            plt.figure(figsize = (10,4))
            plt.plot(loss_score)
            plt.title('Loss score')
            plt.xlabel('Epoch')
            plt.savefig('results/DRE_loss.png', dpi=100, bbox_inches='tight', pad_inches=0.02)
            plt.close()
            visualize_rnets_on_data(score_model, full_PQ_traj_test, all_t, s=0.01)
            visualize_rnets_on_PQ(score_model, full_PQ_traj_test, s=0.01)
    #########
    # Eval alone on test data
    plt.figure(figsize = (10,4))
    plt.plot(loss_score)
    plt.title('Loss score')
    plt.xlabel('Epoch')
    plt.savefig('results/DRE_loss.png', dpi=100, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    visualize_rnets_on_data(score_model, full_PQ_traj_test, all_t, s=0.01)
    visualize_rnets_on_PQ(score_model, full_PQ_traj_test, s=0.01)