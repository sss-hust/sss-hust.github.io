---
title: GPU体系学习
date: 2026-03-18 19:28:45
categories:
---
## 引言

最近写triton总是没有一个比较好的优化方向，遂系统学习，参考[高性能矩阵乘法内核解剖](https://www.aleksagordic.com/blog/matmul?trk=public_post_comment-text)

在LLM中，最主要的操作就是矩阵乘法，而用GPU进行高效矩阵乘法主要有两点——计算并行，缓存命中。所有的优化都是围绕这两点进行展开。

## GPU架构

下图为H100基本架构，总的来说就是最外层是显存空间，容量最大，但访存速度最慢，依次向内的存储空间为——L2cache ，L1cache（和共享内存共用一个硬件内存，后者可以由程序员使用），寄存器。速度依次递增，容量递减。GPU的核心为SM（流式多处理器），其内部包含一个共享内存，以及众多运算器件，TMA。

运算器件中，tensor是主要运算器件，负责大规模张量运算，此外还有些特殊运算（如超越函数）由专门的器件负责。warp scheduler是并行运算的核心，SM最小调度单元就是一个warp，warp中包含32个线程（因此分块时最好是32的倍数）；LD/ST则负责辅助装卸数据。同时TMA就是负责绕开寄存器，直接将显存中数据加载到共享内存中，方便线程进行传输-计算重叠（triton中使用tl.make\_block\_ptr快速得到数据）。在SM外层，还有一个DSMEM（分布式共享内存），是在GPC维度上的存储（GPC是以前的概念，将SM划分组），相较于L2 cache，物理位置更近。

![Figure 1: Model of the NVIDIA Hopper H100 GPU](https://www.aleksagordic.com/blog/matmul/h100_model.png)

![Figure 2: Memory hierarchy of the H100 (SXM5) GPU](https://www.aleksagordic.com/blog/matmul/mem_hierarchy.png)

## 物理与逻辑映射

![Figure 6: CUDA's built-in variables: how threads know where they are](https://www.aleksagordic.com/blog/matmul/cuda_model2.png)

一层嵌套一层，然后逐渐定位，写cuda时需要给每个线程制定好计划，写triton时只需要看block层级，其余的交给编译器。

这里需要提一下，SRAM中存在Bank conflict的问题，SARM内部被组织成32个bank，每个bank 4字节，一个内存事务可以读取所有32个bank的数据，但是如果有相同的内存访问，即不同线程访问同一个bank的不同字节，就会发生严重的踩踏，导致需要多个内存事务才可以完成读取，降低了并行性。但如果是访问同一个bank的同一地址，就可以通过广播，把数据传送给其它线程。

```c
// __global__ keyword declares a GPU kernel
__global__ void naive_kernel(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  int BLOCKSIZE=32;

  const int row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (row < M && col < N) {  // guard in case some threads are outside the range
    float tmp = 0.0;
    // compute dot product
    for (int i = 0; i < K; ++i) {
      tmp += A[row * K + i] * B[i * N + col];
    }
    // GEMM: C = alpha * A @ B + beta * C
    C[row * N + col] = alpha * tmp + beta * C[row * N + col];
  }
}
// create as many blocks as necessary to map all of C
dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
// 32 * 32 = 1024 thread per block
dim3 blockDim(32 * 32);
// launch the asynchronous execution of the kernel on the device
// the function call returns immediately on the host
naive_kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```

对于矩阵乘法，写kernel一般视角是A@B=C，只看A的行和B的列，因为这是最后保留的维度，在这个维度上划分块，每个线程负责块中的一个位置，这个位置是由AB行列相乘得来的。

对于任意时刻，一个warp中的所有线程都会读取一段连续的内存，例如对A的读取，同一行都在读取i，那就会通过广播机制把结果发送给每个线程；对B，此时需要访问的是相同行不同列，多个线程的访存指令会被合并成一个大指令。

由于上面我们分配了众多block（远超SM的数量，一个block只能在一个SM上运行，但SM可以运行多个block），但GPU的SM是有限的，为了尽可能多的利用资源，我们需要了解有哪些资源会限制SM的并发性（即影响运行block的数量），主要有如下三点——寄存器、共享内存、线程/warp，它们将一起决定同一时刻能有多少个block在SM中运行（occupancy）。

## 开始优化（warp-tiling+thread-tiling）

针对Hopper架构以前的GPU，我们需要减少访存并提高运算强度，上面基础实现中，每个线程都只负责结果的一块操作，读取2k但只计算1，计算强度较低，因此需要重新规划线程读取。由于分块机制的存在，可以如下规划：放弃点积思想，转向外积——每次遍历A的列，B的行，得到的是C结果的一部分，多次遍历后累加就可以得到最终的结果。

![Figure 31: Loading a chunk of B (GMEM) into Bs (SMEM)](https://www.aleksagordic.com/blog/matmul/warp_tiling_pt2_smem_load_b.png)

为了高效读取B，每个线程负责读取一行中的4个float，这样就可以触发一次大访存指令；由于一个warp具有32个线程，一次就可以读取128个float，4个warp就是4*128个，并且每个warp都是读取相邻的地址块，访存地址相邻，访存极其舒服，这样四次循环就可以读取B中一个16\*128的块了。

而高效读取A的列，其大小最好是B的转置（如果此时B访存最优，A相应也优秀），每个warp负责读取8行16列，具体分配到线程中，一个线程使用大访存指令读取4个float，一行就是4个线程，总共8行，刚好分配完线程。迭代四次也刚好读取完一个块。但与B的读取不同的是，由于后续访问的是A的列，因此如果把现在读取的存进SRAM，之后访存每次都会发生不连续的情况，导致非常慢，于是把A进行转置，这样在列维度上就是连续的了。但这样在SRAM转置会触发bank conflict，降低速度，现在也只能暂时忍受了。![Figure 32: Loading a chunk of A (GMEM) into As (SMEM)](https://www.aleksagordic.com/blog/matmul/warp_tiling_pt2_smem_load.png)

从整体的视角看，我们先把A按列分块，B按行分块，依次访问A同行不同列的块以及B中同列不同行的块，每个块进行点积，但内部点积方法使用的是外积，以此提高计算强度，内部按照上面的读取方式，一个线程分四次访问A的列，四次访问B的行，最终生成结果中的多个小正方形，后续相加到最终结果上去（这里需要同步），由于所有线程是一起的，可以迅速算完这个块。

## 进一步优化（Hopper架构）

Hopper引入的TMA可以方便异步运输数据，在一个warp中，让一个线程去负责拿数据，其它线程等着使用就可以（更细节来说，还有双缓冲，将共享内存分块，一块用于搬未来数据，一块用来计算先前搬得数据，为此建立流水线策略）。

同时为了解决上述bank conflict问题，还引入了一种swizz技术，可以实现不管访问哪一行那一列，都可以一个事务内完成（有点像八皇后问题），这么复杂的任务，TMA居然可以自动完成！设计这个的真是天才（群论拉丁方阵），大致思想是把每行的bank号与上一行错开，使用xor极其优雅的实现。

![Figure 35: Swizzling example](https://www.aleksagordic.com/blog/matmul/swizzle_example.png)
