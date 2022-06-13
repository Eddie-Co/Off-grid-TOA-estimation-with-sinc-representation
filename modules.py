import torch.nn as nn
import torch.nn.functional as F

def set_tau_module(args):
    """
    Create a TOA-values-representation module
    """
    net = None
    net = tauRepresentationModule(signal_dim=args.signal_dim, n_filters=args.tau_n_filters,
                                            inner_dim=args.tau_inner_dim, n_layers=args.tau_n_layers,
                                            upsampling=args.tau_upsampling, kernel_size=args.tau_kernel_size,
                                            kernel_out=args.tau_kernel_out)
    if args.use_cuda:
        net.cuda()
    return net


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = (self.avg_pool(x)+self.max_pool(x)).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SELayer(out_channels, reduction)

    def forward(self, X):
        Y = self.conv1(F.relu(self.bn1(X)))
        Y = self.conv2(F.relu(self.bn2(Y)))
        Y = self.se(Y)
        return (Y + X)


class tauRepresentationModule(nn.Module):
    def __init__(self, signal_dim=50, n_filters=8, n_layers=7, inner_dim=125,
                 kernel_size=3, upsampling=8, kernel_out=25):
        super().__init__()
        self.tau_size = inner_dim * upsampling
        self.n_filters = n_filters
        self.in_layer = nn.Linear(2 * signal_dim, inner_dim * n_filters, bias=False)
        mod = []
        for n in range(n_layers):
            mod.append(Residual(n_filters, n_filters))
        self.mod = nn.Sequential(*mod)
        self.out_layer = nn.ConvTranspose1d(n_filters, 1, kernel_out, stride=upsampling,
                                            padding=(kernel_out - upsampling + 1) // 2, output_padding=1, bias=False)

    def forward(self, inp):
        bsz = inp.size(0)
        inp = inp.view(bsz, -1)
        x = self.in_layer(inp).view(bsz, self.n_filters, -1)
        x = self.mod(x)
        x = self.out_layer(x).view(bsz, -1)
        return x
