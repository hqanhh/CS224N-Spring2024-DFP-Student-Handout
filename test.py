import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

device = "cuda" if torch.cuda.is_available() else "cpu"

# Lets define a helpful benchmarking function:
import torch.utils.benchmark as benchmark
def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6

# Lets define the hyper-parameters of our input
batch_size = 32
max_sequence_len = 1024
num_heads = 32
embed_dimension = 128

dtype = torch.float16

query = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)
key = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)
value = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


print(f"The python implementation runs in {benchmark_torch_function_in_microseconds(scaled_dot_product_attention, query, key, value):.3f} microseconds")

print(f"The default implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")

# Lets explore the speed of each of the 3 implementations
from torch.nn.attention import SDPBackend, sdpa_kernel


with sdpa_kernel(SDPBackend.MATH):
    math_time=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value)
    print(f"The math implementation runs in {math_time:.3f} microseconds")

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    try:
        flash_time=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value)
        print(f"The flash attention implementation runs in {flash_time:.3f} microseconds")
    except RuntimeError:
        print("FlashAttention is not supported. See warnings for reasons.")

with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    try:
        efficient_time=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value)
        print(f"The memory efficient implementation runs in {efficient_time:.3f} microseconds")
    except RuntimeError:
        print("EfficientAttention is not supported. See warnings for reasons.")