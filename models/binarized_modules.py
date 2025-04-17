import torch
import torch.nn as nn
from sympy.codegen.cnodes import sizeof
from torch.autograd.function import Function, InplaceFunction


class Binarize(InplaceFunction):

    def forward(ctx, input, quant_mode='det', allow_scale=False, inplace=False):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        scale = output.abs().max() if allow_scale else 1

        if quant_mode == 'det':
            return output.div(scale).sign().mul(scale)
        else:
            return output.div(scale).add_(1).div_(2).add_(torch.rand(output.size()).add(-0.5)).clamp_(0,
                                                                                                      1).round().mul_(
                2).add_(-1).mul(scale)

    def backward(ctx, grad_output):
        #STE 
        grad_input = grad_output
        return grad_input, None, None, None


class Quantize(InplaceFunction):
    def forward(ctx, input, quant_mode='det', numBits=4, inplace=False):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        scale = (2 ** numBits - 1) / (output.max() - output.min())
        output = output.mul(scale).clamp(-2 ** (numBits - 1) + 1, 2 ** (numBits - 1))
        if quant_mode == 'det':
            output = output.round().div(scale)
        else:
            output = output.round().add(torch.rand(output.size()).add(-0.5)).div(scale)
        return output

    def backward(grad_output):
        #STE 
        grad_input = grad_output
        return grad_input, None, None


def binarized(input, quant_mode='det'):
    return Binarize.apply(input, quant_mode)


def quantize(input, quant_mode, numBits):
    return Quantize.apply(input, quant_mode, numBits)


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
        self.margin = 1.0

    def hinge_loss(self, input, target):
        #import pdb; pdb.set_trace()
        output = self.margin - input.mul(target)
        output[output.le(0)] = 0
        return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input, target)


class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction, self).__init__()
        self.margin = 1.0

    def forward(self, input, target):
        output = self.margin - input.mul(target)
        output[output.le(0)] = 0
        self.save_for_backward(input, target)
        loss = output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self, grad_output):
        input, target = self.saved_tensors
        output = self.margin - input.mul(target)
        output[output.le(0)] = 0
        grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
        grad_output.mul_(output.ne(0).float())
        grad_output.div_(input.numel())
        return grad_output, grad_output


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if input.size(1) != 16:
            input_b = binarized(input)
        else:
            input_b = input
        weight_b = binarized(self.weight)
        out = nn.functional.linear(input_b, weight_b)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out


class CoarseNormalization(nn.Module):

    def __init__(self, eps=1e-5, elementwise_affine=True, bias=True):
        super(CoarseNormalization, self).__init__()
        self.eps = eps

    def forward(self, input):
        mean = torch.mean(input, dim=1, keepdim=True)
        centered = input - mean
        centered_ap2 = self.ap2_tensor(centered)
        deviation = torch.mean(centered * centered_ap2, dim=1, keepdim=True) + self.eps
        invert_std = self.fast_inv_sqrt(deviation)
        invert_std_ap2 = self.ap2_tensor(invert_std)
        out = centered * invert_std_ap2
        return out


    @staticmethod
    def ap2_tensor(x):
        signs = torch.sign(x)
        abs_x = torch.abs(x) + 1e-10
        return signs * torch.pow(2, torch.round(torch.log2(abs_x)))

    @staticmethod
    def fast_inv_sqrt(x):
        x = torch.clamp(x, min=1e-10)
        i = x.int().type_as(x)
        magic_constant = torch.tensor(0x5f3759df).type_as(i)
        i = magic_constant - (i >> 1)
        return i.type_as(x)


class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if input.size(1) != 3:
            input_b = binarized(input)
        else:
            input_b = input
        weight_b = binarized(self.weight)

        out = nn.functional.conv2d(input_b, weight_b, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
