import torch
from torch.utils.cpp_extension import load
import sys
import os
import gc

gc.collect()
torch.cuda.empty_cache()
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


fused_qkv = load(
    name='fused_qkv',
    sources=['fused_qkv.cu', 'fused_qkv_binding.cpp'],
    # extra_cuda_cflags=['-03', '],
    verbose=True
)

batch_size = 1
seq_len = 16
hidden_dim = 3072
dim_q = 3072 
dim_k = 1024 
dim_v = 1024
X = torch.randn(batch_size * seq_len, hidden_dim, dtype=torch.bfloat16, device="cuda")

w_q = torch.randn(hidden_dim, dim_q, dtype=torch.bfloat16, device='cuda')  # Full 24 heads
w_k = torch.randn(hidden_dim, dim_k, dtype=torch.bfloat16, device='cuda')  # 8 kv heads
w_v = torch.randn(hidden_dim, dim_v, dtype=torch.bfloat16, device='cuda')  # 8 kv heads

print(f"===Weights before===")
print(f"w_q shape: {w_q.shape}, w_k shape: {w_k.shape}, w_v shape: {w_v.shape}")
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


# Fuse weights
w_qkv = torch.cat([w_q, w_k, w_v], dim=1).contiguous()  # [5120, 3072]
print(f"size of w_qkv weight matrix {w_qkv.shape}")

print(f"\n=== Dimension Check ===")
print(f"X.shape: {X.shape}")  # Should be [16, 3072]
print(f"w_qkv.shape: {w_qkv.shape}")  # Should be [3072, 5120]
print(f"Expected output: [{X.shape[0]}, {w_qkv.shape[1]}]")  # Should be [16, 5120]

# Manually verify one element
print(f"\n=== Manual verification ===")
expected_00 = (X[0, :] @ w_qkv[:, 0]).item()
print(f"Expected output[0,0] from PyTorch: {expected_00}")


output = fused_qkv.forward(X, w_qkv)
print(f" Size of output from the cuda kernel {output.shape}")

print(f"Actual output[0,0] from kernel: {output[0, 0].item()}")
print(f"Difference: {abs(expected_00 - output[0, 0].item())}")
Q = torch.matmul(X, w_q)
K = torch.matmul(X, w_k)
V = torch.matmul(X, w_v)
torch.cuda.synchronize()
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

q_fused, k_fused, v_fused = torch.split(output, [dim_q, dim_k, dim_v], dim=-1)
print(f" Size of q split after fusion {q_fused.shape}")
print(f" Size of k split after fusion {k_fused.shape}")
print(f" Size of v split after fusion {v_fused.shape}")
# Verify correctness
q_match = torch.allclose(Q, q_fused, atol=1e-3)
k_match = torch.allclose(K, k_fused, atol=1e-3)
v_match = torch.allclose(V, v_fused, atol=1e-3)

print(f"\n✓ Results match:")
print(f"  Q: {q_match}")
print(f"  K: {k_match}")
print(f"  V: {v_match}")

print(f"Max Q diff: {(Q - q_fused).abs().max()}")
print(f"Max K diff: {(K - k_fused).abs().max()}")

# fused_output.cuda()
# print(f" Are the two outputs similar, {torch.allclose(output, fused_output, atol=1e-3)}")
# print("===Weights after===")
# print(f"w_qkv shape: {w_qkv.shape}")