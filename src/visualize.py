'''Contains visualization tools'''
from sklearn.decomposition import PCA
import torchvision
import torch
import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
from moviepy.editor import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' New ones I added for this new work'''

def plot_losses(ls_all, args, show = True):
    titlesize = 20
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    errs = np.array(ls_all)
    msize = 3
    label_t = ''
    ax[3].plot(errs[:, -1].flatten(), '-o', markersize=msize, label = label_t, color='blue')
    ax[3].set_title('Sum of three', fontsize=titlesize)
    ax[2].plot(errs[:, 0].flatten(), '-o', markersize=msize, label = label_t, color='blue')
    ax[2].set_title(
        r'$-\int_0^1 \nabla \cdot f_b(X_b(s),s)ds$', fontsize=titlesize)
    ax[1].plot(errs[:, 1].flatten(), '-o', markersize=msize, label = label_t, color='blue')
    ax[1].set_title(
        r'$V(X_b+\int_0^1 f_b(X_b(s),s)ds)/2$', fontsize=titlesize)
    ax[0].plot(errs[:, 2].flatten(), '-o', markersize=msize, label = label_t, color='blue')
    # ax[0].set_title(r'$W_2^2(f_b)/(2T_b)$', fontsize=titlesize)
    ax[0].set_title(r'$W_2^2(f_b)$', fontsize=titlesize)
    fig.suptitle(
        f'Training metrics over {args.niters} training epochs\n per block over {args.num_blocks} blocks \n each epoch has {args.tot_batches} batches', 
        y=1.02, fontsize=titlesize)
    for a in ax.flatten():
        # Multiply tick label by args.num_batches
        import matplotlib.ticker as ticker
        a.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y*args.num_batches_switch:.0f}'))
        a.set_xlabel('Num batches trained', fontsize=titlesize)
    fig.tight_layout()
    if show:
        plt.show()
        plt.close()
    return fig

def for_and_back(self, args, block_now = None, cont_t = False,s=1):
    if self.flow_P_to_Z:
        xraw = self.X_train
    else:
        xraw = self.Y_train
    if cont_t == False:
        FlowNet = self.FlowNet
        num_blocks = len(FlowNet)
        z_est_ls = utils.map_for_or_back(input=xraw,
                                        num_blocks=num_blocks, FlowNet=FlowNet,
                                        args=args, reverse=False, test = True)
    else:
        num_blocks = args.num_blocks
        z_est_ls = utils.map_for_or_back_cont_t(input=xraw,
                                        num_blocks=num_blocks, model=self.model,
                                        args=args, reverse=False, test = True)
    zhat = z_est_ls[-1]
    zsamples = torch.randn_like(zhat).to(device)
    if cont_t == False:
        x_est_ls = utils.map_for_or_back(input=zsamples,
                                        num_blocks=num_blocks, FlowNet=FlowNet,
                                        args=args, reverse=True, test = True)
    else:
        x_est_ls = utils.map_for_or_back_cont_t(input=zsamples,
                                        num_blocks=num_blocks, model=self.model,
                                        args=args, reverse=True, test = True)
    xhat = x_est_ls[-1]
    # Start plotting
    fig, ax = plt.subplots(1, 4, figsize=(14, 4))
    plot_dict = {0: xraw, 1: xhat, 2: zsamples, 3: zhat}
    title_dict = {0: 'X', 1: r'$\hat{X}$', 2: 'Z', 3: r'$\hat{Z}$'}
    for i in plot_dict.keys():
        samples = plot_dict[i].cpu().detach().numpy()
        if args.Xdim == 1:
            ax[i].hist(samples)
        else:
            ax[i].scatter(samples[:, 0],samples[:, 1], s=s)
        ax[i].set_title(title_dict[i], fontsize = 24)
        ax[i].tick_params(axis='both', which='major', labelsize=14)
        ax[i].tick_params(axis='both', which='minor', labelsize=14)
    ax[3].get_shared_x_axes().join(ax[3], ax[2])
    ax[3].get_shared_y_axes().join(ax[3], ax[2])
    ax[1].get_shared_x_axes().join(ax[1], ax[0])
    ax[1].get_shared_y_axes().join(ax[1], ax[0])
    fig.tight_layout()
    plt.show()
    plt.close()
    

def plot_panels(arr_list, share_axes = True, flow_P_to_Z = True,s=1,forward = True):
    # The function takes in a list of 2D numpy arrays and plots them using matplotlib. 
    # The function uses subplots to create a grid of panels with a maximum of 8 panels per row. 
    # Each panel is titled with its index in the input list.
    n_panels = len(arr_list)
    max_per_row = n_panels
    n_rows = 1
    sharex, sharey = True, True
    if share_axes == False:
        sharex, sharey = False, False
    fig, ax = plt.subplots(n_rows, max_per_row, figsize=(3*max_per_row, 3), sharex=sharex, sharey=sharey)
    ax = ax.flatten()
    for i, arr in enumerate(arr_list):
        # Handle case where there's only one row of panels
        if arr.shape[1] == 1:
            ax[i].hist(arr)
        else:
            ax[i].scatter(arr[:,0],arr[:,1],s=s)
        if forward:
            ax[i].set_title(f'(F) Block {i}')
        else:
            ax[i].set_title(f'(B) Block {n_panels-i-1}')
        if i == 0:
            if forward:
                if flow_P_to_Z:
                    ax[i].set_title(r'$X\sim P$')
                else:
                    ax[i].set_title(r'$Y\sim Q$')
            else:
                ax[i].set_title(r'$Z\sim N(0,I_d)$')
        if i == len(arr_list) - 1:
            if forward:
                ax[i].set_title(r'$Z\sim N(0,I_d)$')
            else:
                ax[i].set_title(r'$X\sim P$')
    # Remove unused axes
    for j in range(n_panels, n_rows * max_per_row):
        ax[j].axis('off')
    fig.tight_layout()
    plt.show()


''' Visualize projection to 2D '''
def get_PCA_plot(self, X_test, X_test_hat):
    if self.X_test_PCA is None:
        pca = PCA(n_components=2)
        X_test_tmp = X_test.view(X_test.shape[0], -1).cpu().detach().numpy()
        pca.fit(X_test_tmp)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f'Top 2 components explain {explained_var*100:.2f}% variance')
        self.V_two_dim = pca.components_.T
        self.X_test_PCA = X_test_tmp.dot(self.V_two_dim)
    # Project to 2D by PCA
    X_test_hat_PCA = X_test_hat.view(
        X_test.shape[0], -1).cpu().detach().numpy().dot(self.V_two_dim)
    visualize_after_PCA(self.X_test_PCA, X_test_hat_PCA,
                        self.args.word, use_kde = True, add_title = True)
    
def visualize_after_PCA(X_test_PCA, X_test_hat_PCA, dataname, use_kde=False, add_title=True):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3), sharex = True, sharey = True)
    if dataname == 'gas':
        # Remove strange outlier
        X_test_PCA = X_test_PCA[X_test_PCA[:, 0] < 10]
    KDE(X_test_PCA, ax[0], use_kde)
    KDE(X_test_hat_PCA, ax[1], use_kde)
    fontsize = 22
    name0, name1 = '$X$', '$F^{-1}(Z)$'
    if 'solar' in dataname:
        name0, name1 = '$X|Y$', '$F^{-1}(H|Y)$'
    if add_title:
        ax[0].set_title(r'True $X$', fontsize=fontsize)
        ax[1].set_title('JKO-iFlow', fontsize=fontsize)
    else:
        for a in ax.ravel():
            a.axes.get_yaxis().set_visible(False)
            a.axes.get_xaxis().set_visible(False)
    plt.show()
    plt.close()

def KDE(data, ax, use_kde=False, cmap='viridis'):
    # NOTE, run kde takes a while so we not always run it
    # # ax.set_facecolor('lightblue')

    x, y = data[:, 0], data[:, 1]
    if use_kde:
        xy = np.vstack([x, y])
        k = kde.gaussian_kde([x, y])(xy)
        ax.scatter(x, y, c=k, s=2, cmap=cmap)
    else:
        ax.scatter(x, y, s=2)
    # ax.get_yaxis().set_visible(False)
    # ax.get_xaxis().set_visible(False)


def display_mult_images(images, rows, cols, figsize = 0.5, show = True):
    fig, ax = plt.subplots(rows, 1, figsize=(int(figsize*cols), int(figsize*rows)))
    num_per_row = cols
    for i in range(rows):
        start = i*num_per_row
        end = (i+1)*num_per_row
        grid_img_gen = torchvision.utils.make_grid(
            images[start:end], nrow=num_per_row)
        grid_img_gen = grid_img_gen.permute(1, 2, 0).detach().cpu().numpy()
        ax[i].imshow(grid_img_gen)
    for a in ax.ravel():
        a.get_yaxis().set_visible(False)
        a.get_xaxis().set_visible(False)
        a.set_aspect('equal')
    fig.tight_layout(h_pad=0.0, w_pad=0.0)
    if show:
        plt.show()
        plt.close()
    # plt.colorbar()
    return fig