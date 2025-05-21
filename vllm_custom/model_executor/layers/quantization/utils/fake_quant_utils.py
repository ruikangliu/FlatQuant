import torch

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def get_qmin_qmax(bits, sym):
    if sym:
        q_max = torch.tensor(2 ** (bits - 1) - 1)
        q_min = -q_max -1
    else:
        q_max, q_min = torch.tensor(2 ** bits - 1), 0
    return q_max, q_min


def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(round_ste(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))


def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(round_ste(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))


class ActivationQuantizer(torch.nn.Module):
    '''
        A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
        for the activations.
    '''
    def __init__(self, bits, sym=False, lac=False, groupsize=-1, clip_ratio=None, ):
        super(ActivationQuantizer, self).__init__()
        self.bits = bits
        self.q_max, self.q_min = get_qmin_qmax(bits, sym)
        self.sym = sym
        self.groupsize = groupsize
        self.lac = lac
        self._clip_ratio = clip_ratio
        if self.lac:
            init_value = 4.
            self.sigmoid = torch.nn.Sigmoid()
            self.clip_factor_a_max = torch.nn.Parameter(torch.ones((1, ))*init_value, requires_grad=True)
            self.clip_factor_a_min = torch.nn.Parameter(torch.ones((1, ))*init_value, requires_grad=True)
        
        self.enable = True

    def forward(self, x, scale=None, zero=None):
        if self.bits == 16 or (not self.enable):
            return x
        if self.groupsize != -1:
            init_shape = x.shape
            x = x.reshape(-1, self.groupsize)
        fq_x = self.fake_quant(x, scale, zero)
        if self.groupsize != -1:
            fq_x = fq_x.reshape(*init_shape)
        return fq_x

    def fake_quant(self, x, scale=None, zero=None):
        x_dtype = x.dtype
        if scale is None or zero is None:
            scale, zero = self.get_scale_zero(x)
        if self.sym:
            return sym_quant_dequant(x, scale, self.q_max.to(x)).to(x_dtype)
        else:
            return asym_quant_dequant(x, scale, zero, self.q_max.to(x)).to(x_dtype)  # TODO

    def get_scale_zero(self, x):
        q_max = self.q_max.to(x)
        init_shape = x.shape
        reshaped_x = x.reshape((-1, x.shape[-1]))
        xmax, xmin = reshaped_x.amax(1, keepdim=True), reshaped_x.amin(1, keepdim=True)
        tmp = torch.zeros_like(xmax)
        xmax, xmin = torch.maximum(xmax, tmp), torch.minimum(xmin, tmp)
        # # if self.groupsize > 0:
        # #     assert x.shape[-1] % self.groupsize == 0
        # #     x = x.reshape((-1, self.groupsize))
        # #     # TODO: add padding
        if self.lac:
            xmax = xmax * self.sigmoid(self.clip_factor_a_max)
            xmin = xmin * self.sigmoid(self.clip_factor_a_min)
        elif self._clip_ratio is not None:
            xmax = xmax * self._clip_ratio
            xmin = xmin * self._clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            scale = (xmax / q_max)
            scale[tmp] = 1
            scale = scale.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            zero = torch.zeros_like(scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            scale = (xmax - xmin) / q_max
            zero = torch.round(-xmin / scale)

            scale = scale.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            zero = zero.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)

        return scale, zero


class WeightQuantizer(torch.nn.Module):
    '''From GPTQ Repo'''

    def __init__(self, shape=1):
        super(WeightQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

        self.enable = True

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8
    ):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if sym:
            self.maxq = torch.tensor(2**(bits-1)-1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

    def find_params(self, x):
        if self.bits == 16 or (not self.enable):
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:

                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:

            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    def quantize(self, x):
        x_dtype = x.dtype
        if self.enable and self.ready() and self.bits < 16:
            if self.sym:
                return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)
        return x
    
    def forward(self, x):
        return self.quantize(x)

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


def set_quantizer_state(model, enable=True):
    for m in model.modules():
        if isinstance(m, (WeightQuantizer, ActivationQuantizer)):
            m.enable = enable
    return model


def set_weight_quantizer_state(model, enable=True):
    for m in model.modules():
        if isinstance(m, WeightQuantizer):
            m.enable = enable
    return model


def set_act_quantizer_state(model, enable=True):
    for m in model.modules():
        if isinstance(m, ActivationQuantizer):
            m.enable = enable
    return model
