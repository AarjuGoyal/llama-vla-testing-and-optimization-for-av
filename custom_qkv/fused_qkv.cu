#include <torch/torch.h>
#include <torch/extension.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
// #include <stdfloat>


/*
M: Dimension of the input tokens (n)
K: Dimension of embedding space (e)
N: Sum of dimensions of Q, K, V q_dim + k_dim + v_dim. With MHA all equal, with GQA q_dim = k*(k_dim, v_dim)
*/
__global__ void fused_qkv_kernel(
    const __nv_bfloat16* __restrict__ input, //[M , K]
    const __nv_bfloat16* __restrict__ weight, // [K, N]
    __nv_bfloat16* __restrict__ output,      // [M, N]
    int M, int N, int K //N dimension of input, E hidden dimension/ embedding dimension, D is addition of Q, K, V dimension
)
{
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("CUDA Kernel - M: %d, N: %d, K: %d\n", M, N, K);
    }
    
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    
    output[x*N + y] = __float2bfloat16(0.0f);;
    
    if(x < M && y < N)
    {
        float sum = 0.0f;
        for (uint i=0; i< K; i++ )
        {
            sum += __bfloat162float(input[x*K + i]) *__bfloat162float(weight[i*N + y]);
        }
        output[x*N + y] = __float2bfloat16(sum);
    }
    
    
}

torch::Tensor fused_qkv_cuda(
    torch::Tensor input, //[batch*seq, hidden_dim]
    torch::Tensor weight //[q_dim+ k_dim + v_dim, hidden_dim]
)
{
    const int M = input.size(0); //batch * seq
    const int K = input.size(1); // hidden_dim
    const int N = weight.size(1);

    auto output = torch::zeros({M,N}, input.options());

    //Launch Kernel
    dim3 block(16,16);
    dim3 grid((M + block.x - 1) / block.x,( N + block.y - 1)/ block.y);

    fused_qkv_kernel<<<grid, block>>> (
        reinterpret_cast<__nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        M, N, K
    );

    return output;

}