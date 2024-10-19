import deploy
import torch


def get_decompose_dim(n):
    a = int(n ** 0.5)
    if a * a < n:
        a += 1
    while True:
        tmp = a * a - n
        b = int(tmp ** 0.5)
        if b * b == tmp:
            break
        a += 1
    return a - b, a + b 


class OnlineTrans(torch.nn.Module):
    def __init__(self, trans_dim, force_fp32=False, trans="had", decompose=True):
        super().__init__()
        self.fp32_trans = force_fp32
        self.trans = trans
        self.decompose = decompose
        self.trans_dim = trans_dim
        if trans == "had":
            had_rem_dim, self.rem_dim = deploy.functional.online_trans.get_hadK(trans_dim)
            if had_rem_dim is not None:
                self.register_buffer("had_rem_dim", had_rem_dim)
                if not self.fp32_trans:
                    self.had_rem_dim = self.had_rem_dim.to(torch.float16)
            else:
                self.had_rem_dim = None
        elif trans == "matmul":
            if decompose:
                left_size, right_size = get_decompose_dim(trans_dim)
                left_matrix = torch.randn([left_size, left_size], dtype=torch.float16)
                right_matrix = torch.randn([right_size, right_size], dtype=torch.float16)
                self.register_buffer("left_matrix", left_matrix)
                self.register_buffer("right_matrix", right_matrix)
            else:
                right_matrix = torch.randn([trans_dim, trans_dim], dtype=torch.float16)
                self.register_buffer("right_matrix", right_matrix)
            # if decompose:
            #     left_size, right_size = get_decompose_dim(trans_dim)
            #     self.left_matrix = torch.nn.Parameter(torch.randn([left_size, left_size], dtype=torch.float16), requires_grad=True)
            #     self.right_matrix = torch.nn.Parameter(torch.randn([right_size, right_size], dtype=torch.float16), requires_grad=True)
            # else:
            #     self.right_matrix = torch.nn.Parameter(torch.randn([trans_dim, trans_dim], dtype=torch.float16), requires_grad=True)

    def forward(self, x):
        if self.fp32_trans:
            x = x.float()
        if self.trans == "had":
            x = deploy.functional.matmul_hadU_cuda(x, self.had_rem_dim, self.rem_dim)
        elif self.trans == "matmul":
            invs = []
            if hasattr(self, "left_matrix"):
                invs.append(self.left_matrix)
            if hasattr(self, "right_matrix"):
                invs.append(self.right_matrix)
            x = deploy.functional.online_trans.kronecker_matmul(x, invs)
        return x
