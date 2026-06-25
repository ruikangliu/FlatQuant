import json, os
R = json.load(open("outputs/sensitivity_rtn.json"))
F = json.load(open("outputs/sensitivity_flatquant.json"))
L = R["L"]

def diff(c):  # 累积曲线 -> 边际差分 ΔE(n)=E(n)-E(n-1)
    return [c[0]] + [c[i]-c[i-1] for i in range(1, len(c))]

# B 前缀差分 = 第 n 层 (前面已量化前提下) 的边际敏感度
Rb, Fb = diff(R["B_prefix"]), diff(F["B_prefix"])
# C 后缀差分: C[n-1] 对应量化后 n 层 [L-n..L-1]; 差分对应第 (L-n) 层加入
Rc, Fc = diff(R["C_suffix"]), diff(F["C_suffix"])

print("== B 前缀差分 ΔE (浅->深累积语境的边际敏感度) ==")
print(" 第0层  RTN %.3f Flat %.5f"%(Rb[0],Fb[0]))
print(" 末层(差分末点) RTN %.4f Flat %.5f"%(Rb[-1],Fb[-1]))
print("== C 后缀差分 (深->浅累积; 末层为首点) ==")
print(" 末层(31, C首点) RTN %.4f Flat %.5f"%(Rc[0],Fc[0]))

# 画图 (log 轴, RTN/Flat 量级差百倍)
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = list(range(L))
    fig, ax = plt.subplots(1, 3, figsize=(16,4.2))
    ax[0].semilogy(x, R["A_single"], "o-", label="RTN", color="crimson")
    ax[0].semilogy(x, F["A_single"], "s-", label="FlatQuant", color="steelblue")
    ax[0].set_title("A: single-layer KL (isolated)"); ax[0].set_xlabel("layer"); ax[0].set_ylabel("KL (log)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].semilogy(x, [max(v,1e-9) for v in Rb], "o-", label="RTN", color="crimson")
    ax[1].semilogy(x, [max(v,1e-9) for v in Fb], "s-", label="FlatQuant", color="steelblue")
    ax[1].set_title("B: prefix marginal dKL (shallow->deep)"); ax[1].set_xlabel("layer n"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].semilogy(x, [max(v,1e-9) for v in Rc], "o-", label="RTN", color="crimson")
    ax[2].semilogy(x, [max(v,1e-9) for v in Fc], "s-", label="FlatQuant", color="steelblue")
    ax[2].set_title("C: suffix marginal dKL (deep->shallow)"); ax[2].set_xlabel("step (last layers first)"); ax[2].legend(); ax[2].grid(alpha=.3)
    plt.tight_layout()
    out = "/home/maomaotat/xwh/analysis/sensitivity_flatquant_vs_rtn.png"
    plt.savefig(out, dpi=110); print("saved figure:", out)
except Exception as e:
    print("matplotlib 不可用, 跳过画图:", e)
