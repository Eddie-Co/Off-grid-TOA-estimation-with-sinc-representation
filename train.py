import os
import sys
import time
import argparse
import logging
import torch
import numpy as np
import data.loss as loss
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from data import dataset
import modules
import util
from data import re

logger = logging.getLogger(__name__)


def train_off_grid_TOA(args, tau_module, tau_optimizer, tau_criterion, tau_scheduler, train_loader, val_loader,
                                   xgrid, epoch, tb_writer):
    """
    Train the off-grid-TOA-values-representation module for one epoch
    """
    epoch_start_time = time.time()
    tau_module.train()
    loss_train_tau = 0
    for batch_idx, (train_signal, target_tau) in enumerate(train_loader):
        if args.use_cuda:
            train_signal, target_tau = train_signal.cuda(), target_tau.cuda()
        tau_optimizer.zero_grad()
        output_tau = tau_module(train_signal)
        loss_tau = tau_criterion(output_tau, target_tau)
        loss_tau.backward()
        tau_optimizer.step()
        loss_train_tau += loss_tau.data.item()

    tau_module.eval()
    loss_val_tau, fnr_val, c = 0, 0, 0
    for batch_idx, (test_signal, target_tau, tau) in enumerate(val_loader):
        if args.use_cuda:
            test_signal, target_tau = test_signal.cuda(), target_tau.cuda()
        with torch.no_grad():
            output_tau = tau_module(test_signal)
        loss_tau = tau_criterion(output_tau, target_tau)
        loss_val_tau += loss_tau.data.item()
        ntau = (tau >= -0.5).sum(dim=1)
        f_hat, f_hat_off = re.find_tau(output_tau.cpu().detach().numpy(), ntau, xgrid)
        f_hat_plus = (f_hat_off-2)/100
        f_hat = f_hat+f_hat_plus
        c += loss.mse(f_hat, tau.cpu().numpy())
    loss_train_tau /= args.nums_data*(1-args.test_ratio)
    loss_val_tau /= args.nums_data*args.test_ratio
    mse = c / (args.nums_data*args.test_ratio*args.n_tau)

    tb_writer.add_scalar('tau_l2_training', loss_train_tau, epoch)
    tb_writer.add_scalar('tau_l2_validation', loss_val_tau, epoch)
    tb_writer.add_scalar('tau_FNR', fnr_val, epoch)
    tb_writer.add_scalar('MSE', mse, epoch)

    tau_scheduler.step(loss_val_tau)
    logger.info("Epochs: %d / %d, Time: %.1f, TOA training L2 loss %.2f, TOA validation L2 loss %.2f, MSE %.6f",
                epoch, args.n_epochs_tau, time.time() - epoch_start_time, loss_train_tau, loss_val_tau, mse)
    return mse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # basic parameters
    parser.add_argument('--output_dir', type=str, default='./checkpoint/experiment_name', help='output directory')
    parser.add_argument('--no_cuda', action='store_true', help="avoid using CUDA when available")
    # dataset parameters
    parser.add_argument('--nums_data', type=int, default=200000, help='Numbers of dataset')
    parser.add_argument('--test_ratio', type=int, default=0.01, help='Test ratio of dataset')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size used during training')
    parser.add_argument('--signal_dim', type=int, default=50, help='dimension of the input signal')
    parser.add_argument('--tau_size', type=int, default=1000, help='size of the TOA values representation')
    parser.add_argument('--n_tau', type=int, default=3,
                        help='The number of multipath of signal')
    parser.add_argument('--min_sep', type=float, default=2.5,
                        help='minimum separation between spikes, normalized by signal_dim')
    parser.add_argument('--noise', type=str, default='gaussian_blind', help='kind of noise to use')
    parser.add_argument('--snr', type=float, default=20, help='snr parameter')
    # Time delay vaules-representation (tau) module parameters
    parser.add_argument('--tau_module_type', type=str, default='tau', help='type of the tau module: [tau]')
    parser.add_argument('--tau_n_layers', type=int, default=10, help='number of convolutional layers in the tau module')
    parser.add_argument('--tau_n_filters', type=int, default=64, help='number of filters per layer in the tau module')
    parser.add_argument('--tau_kernel_size', type=int, default=3,
                        help='filter size in the convolutional blocks of the fr module')
    parser.add_argument('--tau_kernel_out', type=int, default=25, help='size of the conv transpose kernel')
    parser.add_argument('--tau_inner_dim', type=int, default=125, help='dimension after first linear transformation')
    parser.add_argument('--tau_upsampling', type=int, default=8,
                        help='stride of the transposed convolution, upsampling * inner_dim = tau_size')
    parser.add_argument('--gaussian_std', type=float, default=10,
                        help='std of the gaussian kernel')
    # kernel parameters used to generate the ideal tauuency representation
    parser.add_argument('--kernel_type', type=str, default='sinc',
                        help='type of kernel used to create the ideal tau representation [gaussian, triangle, sinc or one-hot]')
    # training parameters
    parser.add_argument('--lr_tau', type=float, default=0.0005,
                        help='initial learning rate for adam optimizer used for the tau-representation module')
    parser.add_argument('--n_epochs_tau', type=int, default=100, help='number of epochs used to train the tau module')
    parser.add_argument('--save_epoch_tau', type=int, default=2,
                        help='tau of saving checkpoints at the end of epochs')
    parser.add_argument('--numpy_seed', type=int, default=100)
    parser.add_argument('--torch_seed', type=int, default=76)

    np.random.seed(100)
    torch.manual_seed(76)

    args = parser.parse_args()

    if torch.cuda.is_available() and not args.no_cuda:
        args.use_cuda = True
    else:
        args.use_cuda = False

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    file_handler = logging.FileHandler(filename=os.path.join(args.output_dir, 'run.log'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

    tb_writer = SummaryWriter(args.output_dir)
    util.print_args(logger, args)

    train_loader, val_loader = dataset.make_train_test_data(args)

    tau_module = modules.set_tau_module(args)
    tau_optimizer, tau_scheduler = util.set_optim(args, tau_module)
    start_epoch = 1

    logger.info('[Network] Number of parameters in the sinc coding module : %.3f M' % (
                util.model_parameters(tau_module) / 1e6))

    tau_criterion = torch.nn.MSELoss(reduction='sum')

    xgrid = np.linspace(0, 20, args.tau_size, endpoint=False)

    a = np.zeros(args.n_epochs_tau)
    for epoch in range(start_epoch, args.n_epochs_tau + 1):

        a[epoch-1] = train_off_grid_TOA(args=args, tau_module=tau_module, tau_optimizer=tau_optimizer, tau_criterion=tau_criterion,
                                           tau_scheduler=tau_scheduler, train_loader=train_loader, val_loader=val_loader,
                                           xgrid=xgrid, epoch=epoch, tb_writer=tb_writer)

    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.plot(a, label='off-grid Rational Quadratic kernel estimation')
    plt.show()