from sklearn.datasets import make_moons
import numpy as np
import torch
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#### 1d or 2d examples, from either image or Gaussian

def np_to_torch(x, y):
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    return x.to(device), y.to(device)


def data_generator(args, train = True):
    train_data_size = args.train_data_size if train else args.test_data_size
    seed = 1103 if train else 1111
    if args.word == 'moon_to_checkerboard':
        xraw, _ = make_moons(noise=0.05,
                                 n_samples=train_data_size // 2, random_state=seed)
        # Scale to same domain as checkerboard
        mean = xraw.mean(axis=0)
        std = xraw.std(axis=0) / np.array([np.sqrt(4), np.sqrt(5)])
        xraw = (xraw - mean) / std
        xraw = torch.from_numpy(xraw).float().to(device)
        if train:
            args.train_data_size = train_data_size // 2
        else:
            args.test_data_size = train_data_size // 2
        args.word = 'img_checkerboard'
        yraw = inf_train_gen(args, train = train)
        args.word = 'moon_to_checkerboard'
        if train:
            args.train_data_size = train_data_size
        else:
            args.test_data_size = train_data_size
    else:
        raise NotImplementedError
    if isinstance(xraw, np.ndarray):
        xraw, yraw = np_to_torch(xraw, yraw)
    return xraw, yraw

def gen_data_from_img(args, train_data_size, seed):
    ''' From FFJORD '''
    def sample_data(train_data_size):
        np.random.seed(seed)
        inds = np.random.choice(
            int(probs.shape[0]), int(train_data_size), p=probs)
        m = means[inds] # Pre-noise image
        samples = np.random.randn(*m.shape) * std + m # Add tiny noise
        return samples
    img = args.image_mask
    h, w = img.shape
    xx = np.linspace(-4, 4, w)
    yy = np.linspace(-4, 4, h)
    xx, yy = np.meshgrid(xx, yy)
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)
    means = np.concatenate([xx, yy], 1) # (h*w, 2)
    img = img.max() - img
    probs = img.reshape(-1) / img.sum() 
    std = np.array([8 / w / 2, 8 / h / 2])
    args.h, args.w = h, w
    full_data = sample_data(train_data_size)
    return full_data

def inf_train_gen(args, train=True):
    train_data_size = args.train_data_size if train else args.test_data_size
    seed = 1103 if train else 1111
    # More complex 2D data, using masks from images
    args.image_mask = np.array(Image.open(f'./{args.word}.png').rotate(
        180).transpose(0).convert('L'))
    dataset = gen_data_from_img(args, train_data_size, seed)
    X_full = torch.from_numpy(dataset).float().to(device)
    return X_full