import os
import sys
import random
import webbrowser

from datetime import datetime
import numpy as np
import math
from numpy import argmin, linalg as la
import scipy
import matplotlib
from scipy.linalg import block_diag
import torch
from torch.linalg import matrix_norm as mn
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam, LBFGS
from torch.utils.data import DataLoader
import torchvision
import scipy.io as sio
from torchvision.datasets import MNIST  # type: ignore
from torchvision.transforms import Compose, Normalize, ToTensor  # type: ignore
from tqdm import tqdm
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import argparse


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=3, verbose=True, delta=0, path="checkpoint.pt", mode="min"
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.metric_best = np.Inf if mode == "min" else -np.Inf
        self.delta = delta
        self.path = path
        self.mode = mode

    def __call__(self, current_metric, model):

        score = -current_metric if self.mode == "min" else current_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(current_metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"Patience used: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(current_metric, model)
            self.counter = 0

    def should_stop(self):
        return self.early_stop

    def save_checkpoint(self, current_metric, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.metric_best:.6f} --> {current_metric:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.metric_best = current_metric


def parse_args():
    parser = argparse.ArgumentParser("Network's parameters")
    parser.add_argument(
        "--layers",
        type=int,
        default=10,
        help="Number of layers/iterations",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate",
    )

    parser.add_argument(
        "--t10",
        type=float,
        default=1,
        help="Initial step size 1",
    )

    parser.add_argument(
        "--t20",
        type=float,
        default=1,
        help="Initial step size 2",
    )

    parser.add_argument("--mu", type=float, default=100, help="Smoothing parameter")

    parser.add_argument(
        "--red", type=int, default=7840, help="Redundancy factor. Default value =10*amb"
    )

    parser.add_argument("--meas", type=float, default=0.25, help="CS ratio")

    parser.add_argument("--a", type=float, default=0.5, help="Step size 1 parameter")

    parser.add_argument("--b", type=float, default=0.3, help="Step size 2 parameter")

    parser.add_argument("--epochs", type=int, default=300, help="Epochs for training")

    parser.add_argument(
        "--init",
        type=str,
        default="kaiming",
        help="Type of initialization for W: He or Beta",
    )

    parser.add_argument(
        "--init-a",
        type=float,
        default=0.5,
        help="Alpha parameter to be used with beta initialization",
    )

    parser.add_argument(
        "--init-b",
        type=float,
        default=0.7,
        help="Beta parameter to be used with beta initialization",
    )

    parser.add_argument("--batch", type=int, default=128, help="Batch size")

    parser.add_argument(
        "--early-stopping",
        type=int,
        default=0,
        help="Type of early stopping. The default mode (according to paper) is 0",
    )

    return parser.parse_args()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG = False
ARGS = parse_args()

# Model parameters
ACF_ITERATIONS = ARGS.layers  # Number of TFOCS iterations/layers during forward
AMBIENT_DIM = 28 * 28  # AMBIENT_DIM -> Vectorized image pixels
SPARSE_DIM = ARGS.red
NUM_MEASUREMENTS = round(
    ARGS.meas * AMBIENT_DIM
)  # Number of measurements to use for CS
ALPHA = ARGS.a
BETA = ARGS.b

INIT = ARGS.init
INIT_ALPHA = ARGS.init_a
INIT_BETA = ARGS.init_b
EARLY_ID = ARGS.early_stopping

# Training parameters
LEARNING_RATE = ARGS.lr  # Adam Learning rate
BATCH_SIZE = ARGS.batch  # How many images to process in parallel
NUM_EPOCHS = ARGS.epochs  # Epochs to train


#######################################################################################
# Data Loading & Utility functions                                                    #
#######################################################################################
def get_data_loaders(train_batch_size, val_batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train = MNIST(download=True, root=".", transform=data_transform, train=True)
    val = MNIST(download=False, root=".", transform=data_transform, train=False)
    train_loader = DataLoader(
        train,
        num_workers=2,
        batch_size=train_batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val, num_workers=2, batch_size=val_batch_size, shuffle=False
    )

    return train_loader, val_loader


def safe_mkdirs(path: str) -> None:
    """! Makes recursively all the directory in input path """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            log.warning(e)
            raise IOError((f"Failed to create recursive directories: {path}"))


def date_fname():
    uniq_filename = (
        str(datetime.now().date()) + "_" + str(datetime.now().time()).replace(":", ".")
    )

    return uniq_filename


def save_image(grid, fname):
    from PIL import Image

    ndarr = (
        grid.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )

    im = Image.fromarray(ndarr)
    im.save(fname)


def save_examples(model, val_loader, epoch, algo="ACF_MNIST", device="cpu"):
    psnr_fn = PSNR()
    idxes = random.sample(range(len(val_loader.dataset)), 16)
    original = torch.stack([val_loader.dataset[i][0] for i in idxes]).to(device)
    reconstructed = model(original).view(original.size(0), 1, 28, 28)
    original = original.detach().cpu()
    reconstructed = reconstructed.detach().cpu()
    img_orig = torchvision.utils.make_grid(original, nrow=4)
    img_recon = torchvision.utils.make_grid(reconstructed, nrow=4)
    psnr = psnr_fn(img_orig, img_recon)
    folder = f"results_ACF_MNIST/{algo}_{date_fname()}_epoch.{epoch}"
    safe_mkdirs(folder)
    save_image(img_orig, f"{folder}/original_epoch_{epoch}.jpg")
    save_image(img_recon, f"{folder}/reconstructed_epoch_{epoch}/PSNR_{psnr}.jpg")
    html = """
    <!DOCTYPE html>
    <html>
    <body>

    <h2>Original</h2>
    <img src="original_epoch_{epoch}.jpg" width="500" height="500">


    <h2>Reconstructed</h2>
    <img src="reconstructed_epoch_{epoch}.jpg" width="500" height="500">

    </body>
    </html>    
    """.format(
        epoch=epoch
    )
    html_file = f"{folder}/epoch_{epoch}.html"
    with open(html_file, "w") as fd:
        fd.write(html)
    # webbrowser.open(html_file)


def eye_like(tensor):
    return torch.eye(*tensor.size(), out=torch.empty_like(tensor)).to(DEVICE)


class PSNR:
    # Peak Signal to Noise Ratio img1 and img2 have range [0, 1]

    def __init__(self):
        self.name = "PSNR"

    # @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))


#######################################################################################
# Model Implementation                                                                #
#######################################################################################


class ShrinkageActivation(nn.Module):
    def __init__(self):
        super(ShrinkageActivation, self).__init__()

    def forward(self, x, epsilon):
        return torch.sign(x) * torch.max(torch.zeros_like(x), torch.abs(x) - epsilon)


class TruncationActivation(nn.Module):
    def __init__(self):
        super(TruncationActivation, self).__init__()

    def forward(self, x, epsilon):
        return torch.sign(x) * torch.min(torch.abs(x), epsilon * torch.ones_like(x))


class DECONET(nn.Module):
    def __init__(
        self,
        ambient=28 * 28,
        measurements=round(0.25 * 784),
        sparse_dim=7840,
        acf_iterations=10,
        alpha=0.7,
        beta=0.5,
        init_alpha=0.5,
        init_beta=0.7,
        initial="kaiming",
    ):
        super(DECONET, self).__init__()
        print("Model Hyperparameters:")
        print(f"\tambient_dim={ambient}")
        print(f"\tmeasurements={measurements}")
        print(f"\tredundancy={sparse_dim}")
        print(f"\tDECONET_iterations={acf_iterations}")
        print(f"\tStep size 1 parameter={alpha}")
        print(f"\tStep size 2 parameter={beta}")
        print(f"\tType of initialization={initial}")
        print(f"\tBeta parameter a={init_alpha}")
        print(f"\tBeta parameter b={init_beta}")
        self.alpha = alpha
        self.beta = beta
        self.init_alpha = init_alpha
        self.init_beta = init_beta
        self.sparse_dim = sparse_dim
        self.acf_iterations = acf_iterations
        self.measurements = measurements
        self.ambient = ambient
        self.initial = initial
        self.first_activation = TruncationActivation()
        self.second_activation = ShrinkageActivation()
        A = torch.randn(measurements, ambient) / np.sqrt(self.measurements)
        self.register_buffer("A", A)
        phi = nn.Parameter(self._init_phi())
        self.register_parameter("phi", phi)

    def _init_phi(self):

        init = torch.empty(self.sparse_dim, self.ambient)

        if self.initial == "kaiming":
            init = torch.nn.init.kaiming_normal_(init)
            init = init  # / mn(init, ord=2)
        else:  ## BETA
            with torch.no_grad():
                beta_dist = torch.distributions.beta.Beta(
                    torch.tensor(self.init_alpha), torch.tensor(self.init_beta)
                )
                init = beta_dist.sample((self.sparse_dim, self.ambient))
                init = init / mn(init, ord=2)

        return init

    def extra_repr(self):
        return "(phi): Parameter({}, {})".format(self.sparse_dim, self.ambient)

    def measure_x(self, x):
        # Create measurements y
        y = torch.einsum("ma,ba->bm", self.A, x)

        return y

    def noisy_measure(self, y):
        # add Gaussian noise to y
        y_noisy = y + 0.0001 * torch.randn_like(y)

        return y_noisy

    def affine_transform1(self, theta1, u1, z1, t1, x):
        affine1 = (
            (1 - theta1) * u1
            + theta1 * z1
            - (t1 / theta1) * torch.einsum("sa,ba->bs", self.phi, x)
        )

        return affine1.detach()

    def affine_transform2(self, theta2, u2, z2, t2, y, x):
        affine2 = (
            (1 - theta2) * u2
            + theta2 * z2
            - (t2 / theta2) * (y - torch.einsum("sa,ba->bs", self.A, x))
        )

        return affine2.detach()

    def decode(self, y, epsilon, mu, x0, z1, z2, min_x, max_x):
        u1 = z1
        u2 = z2
        t1 = ARGS.t10
        t2 = ARGS.t20
        theta1 = 1
        theta2 = 1
        Lexact = torch.tensor([1000.0]).to(DEVICE)

        for _ in range(self.acf_iterations):
            x_hat = (
                x0
                + (
                    (1 - theta1) * torch.einsum("as,bs->ba", self.phi.t(), u1)
                    + theta1 * torch.einsum("as,bs->ba", self.phi.t(), z1)
                    - (1 - theta2) * torch.einsum("am,bm->ba", self.A.t(), u2)
                    - theta2 * torch.einsum("am,bm->ba", self.A.t(), z2)
                )
                / mu
            )

            w1 = self.affine_transform1(theta1, u1, z1, t1, x_hat)
            w2 = self.affine_transform2(theta2, u2, z2, t2, y, x_hat)

            z1 = self.first_activation(w1, t1 / theta1)
            z2 = self.second_activation(w2, t2 * epsilon / theta2)
            u1 = (1 - theta1) * u1 + theta1 * z1
            u2 = (1 - theta2) * u2 + theta2 * z2

            t1 = self.alpha * t1
            t2 = self.beta * t2
            muL = torch.sqrt(mu / Lexact).to(DEVICE)
            theta_scale = (1 - muL) / (1 + muL).to(DEVICE)
            theta1 = torch.min(torch.tensor([1.0]).to(DEVICE), theta1 * theta_scale)
            theta2 = torch.min(torch.tensor([1.0]).to(DEVICE), theta2 * theta_scale)

        return torch.clamp(x_hat, min=min_x, max=max_x)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        min_x = torch.min(x)
        max_x = torch.max(x)

        y = self.measure_x(x)
        y_noisy = self.noisy_measure(y)

        x0 = torch.einsum("am,bm->ba", self.A.t(), y_noisy)
        phix0 = torch.einsum("sa,ba->bs", self.phi, x0)
        mu = ARGS.mu

        z1 = torch.zeros_like(phix0)
        z2 = torch.zeros_like(y)

        epsilon = torch.norm(y - y_noisy)

        x_hat = self.decode(y_noisy, epsilon, mu, x0, z1, z2, min_x, max_x)

        return x_hat


#######################################################################################
# Training Functions                                                                  #
#######################################################################################


def train_step(
    model,
    optimizer,
    criterion,
    batch,
    device="cpu",
):
    x_original, _ = batch
    x_original = x_original.to(device)
    optimizer.zero_grad()
    x_pred = model(x_original)
    mse = criterion(x_pred, x_original.view(x_original.size(0), -1))

    loss = mse
    loss.backward()
    optimizer.step()

    return loss


def train_epoch(
    model,
    optimizer,
    criterion,
    train_loader,
    device="cpu",
):
    avg_train_loss = 0
    n_proc = 0

    train_iter = tqdm(train_loader, desc="training", leave=True)

    model.train()

    for i, batch in enumerate(train_iter):
        n_proc += 1
        loss = train_step(
            model,
            optimizer,
            criterion,
            batch,
            device=device,
        )
        avg_train_loss += loss.item()
        train_iter.set_postfix({"loss": "{:.3}".format(avg_train_loss / n_proc)})
    avg_train_loss = avg_train_loss / len(train_loader)

    return avg_train_loss


def val_step(model, criterion, batch, device="cpu"):
    x_original, _ = batch
    x_original = x_original.to(device)
    x_pred = model(x_original)
    mse = criterion(x_pred, x_original.view(x_original.size(0), -1))

    return mse


def val_epoch(model, criterion, val_loader, device="cpu"):
    avg_val_mse = 0
    n_proc = 0
    val_iter = tqdm(val_loader, desc="validation", leave=True)

    model.eval()

    for batch in val_iter:
        n_proc += 1
        mse = val_step(model, criterion, batch, device=device)
        avg_val_mse += mse.item()
        val_iter.set_postfix(
            {
                "mse": "{:.3}".format(avg_val_mse / n_proc),
            }
        )

    avg_val_mse = avg_val_mse / len(val_loader)

    return avg_val_mse


def train(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    epochs,
    early_stop,
    checkpoint_name,
    device="cpu",
):

    early_stopping = EarlyStopping(patience=5, path=checkpoint_name, mode="min")
    min_val_mse = np.Inf

    for e in range(epochs):
        avg_train_mse = train_epoch(
            model,
            optimizer,
            criterion,
            train_loader,
            device=device,
        )

        avg_val_mse = val_epoch(model, criterion, val_loader, device=device)
        gen_mse = np.abs(avg_train_mse - avg_val_mse)
        normalized_gen_mse = gen_mse / min_val_mse

        # save_examples(model, val_loader, e, algo="acf", device=device) # uncomment if you want to see pairs of original+reconstructed images

        print("--------------------------------------")
        print("Average Train MSE = {:.6f}".format(avg_train_mse))
        print("--------------------------------------")
        print("Average Test MSE = {:.6f}".format(avg_val_mse))
        print("--------------------------------------")
        print("Epoch no. = ", e)
        print("--------------------------------------")
        print("Generalization error = {:.6f}".format(gen_mse))
        print("--------------------------------------")

        if early_stop == 0:
            early_stopping(gen_mse, model)
        elif early_stop == 1:
            early_stopping(normalized_gen_mse, model)
        else:
            print(
                "Wrong choice for early stopping. Choose 0 for val, 1 for gen or 2 for normalized gen."
            )
            break

        if early_stopping.should_stop():
            break


#######################################################################################
# Main                                                                                #
#######################################################################################


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = get_data_loaders(BATCH_SIZE, BATCH_SIZE)

    model = DECONET(
        ambient=AMBIENT_DIM,
        measurements=NUM_MEASUREMENTS,
        sparse_dim=SPARSE_DIM,
        acf_iterations=ACF_ITERATIONS,
        alpha=ALPHA,
        beta=BETA,
        init_alpha=INIT_ALPHA,
        init_beta=INIT_BETA,
        initial=INIT,
    ).to(device)

    print(model)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    early_stop = EARLY_ID
    epochs = NUM_EPOCHS

    checkpoint_name = f"DECONET(MNIST)-{ARGS.layers}L-red{ARGS.red}-lr{ARGS.lr}-mu{ARGS.mu}-init{ARGS.init}.pt"

    train(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        epochs,
        early_stop,
        checkpoint_name,
        device=device,
    )
