import os
import torch
import errno
import modules

def model_parameters(model):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    return num_params


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def save(model, optimizer, scheduler, args, epoch, module_type):
    checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'args': args,
    }
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if not os.path.exists(os.path.join(args.output_dir, module_type)):
        os.makedirs(os.path.join(args.output_dir, module_type))
    fn = os.path.join(args.output_dir, module_type, 'epoch_'+str(epoch)+'.pth')
    torch.save(checkpoint, fn)


def load(fn, device = torch.device('cuda')):
    checkpoint = torch.load(fn, map_location=device)
    args = checkpoint['args']
    if device == torch.device('cpu'):
        args.use_cuda = False
    model = modules.set_tau_module(args)
    model.load_state_dict(checkpoint['model'])
    optimizer, scheduler = set_optim(args, model)
    if checkpoint["scheduler"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, scheduler, args, checkpoint['epoch']


def set_optim(args, module):
    optimizer = torch.optim.Adam(module.parameters(), lr=args.lr_tau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.8, verbose=True)
    return optimizer, scheduler


def print_args(logger, args):
    message = ''
    for k, v in sorted(vars(args).items()):
        message += '\n{:>30}: {:<30}'.format(str(k), str(v))
    logger.info(message)

    args_path = os.path.join(args.output_dir, 'run.args')
    with open(args_path, 'wt') as args_file:
        args_file.write(message)
        args_file.write('\n')
