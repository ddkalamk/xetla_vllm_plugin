import torch, time, xetla_pt_ext
dev='xpu'
def bench(fn,n=100):
    for _ in range(5): fn()
    torch.xpu.synchronize(); t=time.perf_counter()
    for _ in range(n): fn()
    torch.xpu.synchronize(); return (time.perf_counter()-t)/n*1e6
for K,N,name in [(5120,16384,'in_proj_qkvz'),(5120,34816,'gate_up'),(17408,5120,'down'),(6144,5120,'out_proj')]:
    W=torch.randint(-2**31,2**31-1,(K//16,N),dtype=torch.int32,device=dev); S=(torch.rand(K//128,N,device=dev)*0.01).half()
    row=[]
    for M in (1,2,4,8,16):
        x=torch.randn(M,K,device=dev).half()
        t_up=bench(lambda: torch.ops.xetla_int2.int2_fp16_upcvt_gemm_run(x,W,S,None))
        t_dp=bench(lambda: torch.ops.xetla_int2.int2_fp16_dpas_gemm_run(x,W,S,None)) if M>1 else float('nan')
        row.append(f"M={M}: upcvt {t_up:6.1f}us dpas {t_dp:6.1f}us")
    print(f"{name:13s} K={K} N={N}\n   "+"\n   ".join(row))
