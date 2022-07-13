from model import common
import torch
import torch.nn as nn
import functools
import torch.nn.functional as F


def make_model(args, parent=False):
    return EGASR(args)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CoreModule(nn.Module):
    def __init__(self, features, M, G, r, stride=1, L=32):
        super(CoreModule, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features

        self.edgeconvs = nn.ModuleList([])

        self.recon_trunk = common.ResidualBlock_noBN(nf=features, at='prelu')

        self.conv3x3 = nn.Conv2d(features, features, kernel_size=3, padding=1)

        EC_combination = ['conv1-sobelx', 'conv1-sobelx', 'conv1-laplacian']
        for i in range(len(EC_combination)):
            self.edgeconvs.append(nn.Sequential(
                common.EdgeConv(EC_combination[i], features, features),
            ))

        self.conv_reduce = nn.Conv2d(features * len(EC_combination), features, kernel_size=1, padding=0)
        self.sa = SpatialAttention()

        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        based_x = x

        out = self.recon_trunk(based_x)

        for i, edgeconv in enumerate(self.edgeconvs):
            fea = edgeconv(x)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        feas = self.conv_reduce(feas)
        feas = self.sa(feas) * feas

        feas_f = torch.cat([out.unsqueeze_(dim=1), feas.unsqueeze_(dim=1)], dim=1)
        fea_f_U = torch.sum(feas_f, dim=1)

        fea_f_s = fea_f_U.mean(-1).mean(-1)
        fea_f_z = self.fc(fea_f_s)
        for i, fc in enumerate(self.fcs):
            vector_f = fc(fea_f_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors_f = vector_f
            else:
                attention_vectors_f = torch.cat([attention_vectors_f, vector_f], dim=1)
        attention_vectors_f = self.softmax(attention_vectors_f)
        attention_vectors_f = attention_vectors_f.unsqueeze(-1).unsqueeze(-1)
        fea_v_out = (feas_f * attention_vectors_f).sum(dim=1)

        return fea_v_out


# # Edge-guided Residual Attention Block (EGRAB)
class EGRAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.PReLU(), res_scale=1):

        super(EGRAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)

        modules_body.append(CoreModule(n_feat, M=2, G=8, r=2, stride=1, L=32))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# ## Residual Edge-enhanced Group (REG)
class REG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(REG, self).__init__()
        modules_body = []
        modules_body = [
            EGRAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.PReLU(), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# ## Edge-guided Attention-based SR Network (EGASR)
class EGASR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EGASR, self).__init__()

        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale[0]
        self.sf = args.scale[0]
        act = nn.PReLU()
        self.gamma = nn.Parameter(torch.zeros(1))
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        self.REGs = nn.ModuleList([
            REG(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)])

        self.conv = conv(n_feats, n_feats, kernel_size)

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x_h = self.head(x)
        xx = self.conv(x_h)
        residual = xx

        for i, l in enumerate(self.REGs):
            xx = l(xx) + self.gamma * residual

        xx = self.conv(xx)

        out = self.tail(xx)
        base = F.interpolate(x, scale_factor=self.sf, mode='bilinear', align_corners=False)
        out += base
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
