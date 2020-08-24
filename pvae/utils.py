import sys
import math
import time
import os
import shutil
import torch
import torch.distributions as dist
from torch.autograd import Variable, Function, grad
import numpy as np
import torch.nn as nn


def lexpand(A, *dimensions):
    """Expand tensor, adding new dimensions on left."""
    return A.expand(tuple(dimensions) + A.shape)


def rexpand(A, *dimensions):
    """Expand tensor, adding new dimensions on right."""
    return A.view(A.shape + (1,) * len(dimensions)).expand(A.shape + tuple(dimensions))


def assert_no_nan(name, g):
    if torch.isnan(g).any(): raise Exception('nans in {}'.format(name))


def assert_no_grad_nan(name, x):
    if x.requires_grad: x.register_hook(lambda g: assert_no_nan(name, g))


# Classes
class Constants(object):
    eta = 1e-5
    log2 = math.log(2)
    logpi = math.log(math.pi)
    log2pi = math.log(2 * math.pi)
    logceilc = 88  # largest cuda v s.t. exp(v) < inf
    logfloorc = -104  # smallest cuda v s.t. exp(v) > 0
    invsqrt2pi = 1. / math.sqrt(2 * math.pi)
    sqrthalfpi = math.sqrt(math.pi / 2)


def logsinh(x):
    # torch.log(sinh(x))
    return x + torch.log(1 - torch.exp(-2 * x)) - Constants.log2


def logcosh(x):
    # torch.log(cosh(x))
    return x + torch.log(1 + torch.exp(-2 * x)) - Constants.log2


class Arccosh(Function):
    # https://github.com/facebookresearch/poincare-embeddings/blob/master/model.py
    @staticmethod
    def forward(ctx, x):
        ctx.z = torch.sqrt(x * x - 1)
        return torch.log(x + ctx.z)

    @staticmethod
    def backward(ctx, g):
        z = torch.clamp(ctx.z, min=Constants.eta)
        z = g / z
        return z


class Arcsinh(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.z = torch.sqrt(x * x + 1)
        return torch.log(x + ctx.z)

    @staticmethod
    def backward(ctx, g):
        z = torch.clamp(ctx.z, min=Constants.eta)
        z = g / z
        return z


# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.begin = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.begin
        self.elapsedH = time.gmtime(self.elapsed)
        print('====> [{}] Time: {:7.3f}s or {}'
              .format(self.name,
                      self.elapsed,
                      time.strftime("%H:%M:%S", self.elapsedH)))


# Functions
def save_vars(vs, filepath):
    """
    Saves variables to the given filepath in a safe manner.
    """
    if os.path.exists(filepath):
        shutil.copyfile(filepath, '{}.old'.format(filepath))
    torch.save(vs, filepath)


def save_model(model, filepath):
    """
    To load a saved model, simply use
    `model.load_state_dict(torch.load('path-to-saved-model'))`.
    """
    save_vars(model.state_dict(), filepath)


def log_mean_exp(value, dim=0, keepdim=False):
    return log_sum_exp(value, dim, keepdim) - math.log(value.size(dim))


def log_sum_exp(value, dim=0, keepdim=False):
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))


def log_sum_exp_signs(value, signs, dim=0, keepdim=False):
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(torch.sum(signs * torch.exp(value0), dim=dim, keepdim=keepdim))


def get_mean_param(params):
    """Return the parameter used to show reconstructions or generations.
    For example, the mean for Normal, or probs for Bernoulli.
    For Bernoulli, skip first parameter, as that's (scalar) temperature
    """
    if params[0].dim() == 0:
        return params[1]
    # elif len(params) == 3:
    #     return params[1]
    else:
        return params[0]


def probe_infnan(v, name, extras={}):
    nps = torch.isnan(v)
    s = nps.sum().item()
    if s > 0:
        print('>>> {} >>>'.format(name))
        print(name, s)
        print(v[nps])
        for k, val in extras.items():
            print(k, val, val.sum().item())
        quit()


def has_analytic_kl(type_p, type_q):
    return (type_p, type_q) in torch.distributions.kl._KL_REGISTRY



def find_point_in_cone(model, initial_point, sampling='random', t=2, **kwargs):
    # assumes lorentzian point as input
    samples_i_want = 2
    if sampling == 'informed':
        qz_x = kwargs.get('qz_x', None)
    found = False
    attempts = 0
    how_many_have_i_found = 0
    samples_return = []
    cone_idx_return = []
    while not found:
        attempts += 1
        if sampling == 'random':
            samples = model.sample_new(10000000)
        elif sampling == 'informed':
            samples = qz_x.rsample(torch.Size([100000])).squeeze(0)

        new_point = model.manifold.from_poincare_ball_to_lorentz(samples)

        sum = 0
        for jj in range(1, initial_point.shape[0]):
            sum += (new_point[:, jj] - initial_point[jj]) ** 2
        sum = sum - t ** 2

        neg_idx = (sum <= 0).nonzero()
        sum_neg = sum[neg_idx]
        new_points = new_point[neg_idx]

        if len(neg_idx) > 0:
            for oo, ff in enumerate(new_points):
                cone_idx_return.append(sum_neg[oo, 0].view(-1, 1))
                samples_return.append(ff.view(-1, initial_point.shape[0]))

            how_many_have_i_found += neg_idx.shape[0]
            if how_many_have_i_found >= samples_i_want:
                attempts = 0
                found = True
        if attempts == 1000:
            attempts = 0
            found = True

    return samples_return, cone_idx_return


def find_interescting_cones(model, initial_point, sampling='random', t=2, **kwargs):
    cones_to_intersect = kwargs.get('cones_to_intersect', 2)
    images_to_keep = initial_point
    cone_origin = initial_point
    # assumes images as inputs

    lorentz_points_to_keep = []
    method_for_comparison = kwargs.get('comparison', 'whites')

    for i in range(cones_to_intersect):
        qz_x = model.qz_x(*model.enc(initial_point))
        embedded = qz_x.rsample(torch.Size([1])).squeeze(0)
        lorentz_point = model.manifold.from_poincare_ball_to_lorentz(embedded)

        if i == 0: next_lorentz_point = lorentz_point

        lorentz_points_to_keep = [next_lorentz_point] + lorentz_points_to_keep

        samples_return, cone_idx_return = find_point_in_cone(model, next_lorentz_point, sampling, t, qz_x=qz_x)
        # failsafe switch
        if len(samples_return) <= 1:
            break
        new_points = torch.stack(samples_return).view(len(samples_return), -1)

        for time_dif, cone_points in enumerate(lorentz_points_to_keep):
            # give slack due to failing theoretical assumption of perfect knowledge
            new_points = get_in_cone_points(new_points, cone_points, t * (1 + time_dif))

        #failsafe switch
        if len(new_points) <= 1:
            break

        # back to poincare ball
        new_qz_x = model.manifold.to_poincare_ball_from_lorentz(new_points)
        new_qz_x.to(initial_point.device)

        reconstructed = get_mean_param(model.dec(new_qz_x.squeeze(0)))

        # find closest image

        smallest_indx = compare_images(reconstructed, cone_origin, method_for_comparison)

        initial_point = reconstructed[smallest_indx]

        next_lorentz_point = new_points[smallest_indx]


        images_to_keep = torch.cat([initial_point, images_to_keep])
        sampling = 'random'

    return images_to_keep


def assert_in_cone(point_to_check, cone_point, t):
    # assumes points on lorentzian manifold
    sum = 0
    for jj in range(1, cone_point.shape[0]):
        sum += (point_to_check[jj] - cone_point[jj]) ** 2
    sum = sum - t ** 2

    if sum <= 0:
        return True
    else:
        return False


def get_in_cone_points(points_to_check, cone_point, t):
    # assumes points on lorentzian manifold

    verified = []
    for point in points_to_check:
        is_it_in = assert_in_cone(point, cone_point, t)
        if is_it_in:
            verified.append(point)
    if len(verified) > 0:
        verified = torch.stack(verified).view(len(verified), -1)
    return verified


def compare_images(images_to_compare, gt, method_for_comparison):
    if method_for_comparison == 'mse':
        mse = nn.MSELoss(reduction='none')

        comparisons = mse(images_to_compare, gt)
        comparisons = comparisons.mean(axis=[1, 2, 3])
        indeces = torch.argsort(comparisons)
        return indeces[0]
    elif method_for_comparison=='whites':
        gold_standard = (gt >= 0.5).sum()
        minimum = 1e10
        arg_min = 0
        for i, img in enumerate(images_to_compare):
            whites = (img >= 0.5).sum()
            dif = torch.abs(gold_standard - whites)
            if dif <= minimum:
                minimum = dif
                arg_min = i
        return arg_min
    else:
        raise NotImplementedError



