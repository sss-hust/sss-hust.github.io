---
title: nano-vllm搭建
date: 2026-03-05 17:25:33
categories:
- AI与深度学习
tags:
- 大模型推理
- 推理优化
- Triton
- 模型量化
- 投机解码
mathjax: true
---
[toc]

了解nano-vllm后，着手开始构建简单推理框架，现在已经有最基本的加速部件，现在打算添加算法层面优化方法——SD。

当前系统支持模型为qwen3-0.6B，在4090 24G显卡上跑绰绰有余，于是把它当成草稿模型，找一个更大的模型，先不更改太多模型架构，当前只支持稠密模型，因此使用Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4，未量化版本也才14G，量化版本在5G左右，用以试试水。

### SD（spectaculate decode）

根据当前知识，前向推理时激活两个模型，一个草稿一个目标模型，但是两个模型前向推理过程不太熟悉，是两者同步还是一前一后呢？同时需要维护两个kvblock，管理上更复杂些，相应的也会影响长上下文的表现。不管运行顺序是什么，目标模型都会对草稿模型提交的矩阵做一次前向传播，再判断是否接受，如不接受就需要回退。

进一步细究SD的原理，有一个非常重要的点——在数学上是无损的，即草稿模型选中与否，其总概率是和原先一样的（对于按概率采样），数学原理如下：

目标模型前向计算出下一个token的真实分布$p(x)$，此时以一定概率接受草稿模型生成的token，$q(x)$为草稿模型生成的采样分布。

α=min(1,q(x)p(x))这里包含两种情况：

* 如果$p(x) \ge q(x)$：大模型认为这个 Token 出现的概率比小模型认为的还要高，或者一样高。此时 $\alpha = 1$，我们 **100% 接受**这个候选 Token。
* 如果 $p(x) < q(x)$：小模型“过于自信”了，高估了这个 Token 的概率。此时我们要按比例惩罚它，以$\alpha = \frac{p(x)}{q(x)}$的概率**接受**它，以$1 - \frac{p(x)}{q(x)}$ 的概率**拒绝**它。

拒绝后重采样的概率空间需要改变，需要将之前被草稿模型高估的token概率降低，维持总的分布不变。

新的采样分布$p'(x)$定义为：

$$
p'(x) \propto \max(0, p(x) - q(x))
$$

这意味着，我们只从那些“大模型给的概率高于小模型给的概率”的 Token 中去抽卡，从而弥补小模型未覆盖到的概率空间。标准化后的重采样概率为：

$$
p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_{y} \max(0, p(y) - q(y))}
$$

上面的设计非常巧妙，首先是接受概率，不管q是大是小，都尽量把它往标准的概率去按，如果更小，那就需要补偿；如果更大，就强制按到p，不需要补偿了。综合来看，需要补偿的就是前者，因此重采样$p'(x)$才是这样定义的，从每一个token的视角来看都是如此，因为没选择之前，每个token都有可能选中。

最后可以证明，在这些数学限制下，最终生成的token分布不会有任何变化，

### GPTQ（int4）

#### 基本原理

简单介绍下量化原理，GPTQ这里采用的是非对称量化，将其变形到0-15之间，首先先确定scale

$$
Scale = \frac{Max - Min}{15}
$$

即找到缩放系数，然后就可以定义零点了

$$
Zero\_point = \text{round}\left(0-\frac{Min}{Scale}\right)
$$

于是就可以把所有的数塞进更小的数组里

$$
W_{4bit} = \text{round}\left(\frac{W-0}{Scale} + Zero\_point\right)
$$

反量化公式自然就是：

$$
W_{真实} \approx Scale \times (W_{4bit} - Zero\_point)
$$

但这样会发现如果全是正数，那定出来的零点将会是负值（下溢），或者全是负数，零点将上溢，这两种情况都无法表示。GPTQ采用的是强制拉回含0的区间，即让原始值最大最小值包含0，这样量化后的0也会落在对应区间。

#### 参数读取

首先先支持量化版模型，需要修改三个文件，一是config.py，记录一些关于量化的信息，包括——量化方法、精度、每组大小。直接从文件中读取就可以。由于python会把这些自动改成dict，无法通过属性访问，采用get方法

~~~python
        if quant_cfg is not None:
            self.quant_method = quant_cfg.get("quant_method")
            self.quant_bits = quant_cfg.get("bits")
            self.quant_group_size = quant_cfg.get("group_size")
~~~

然后是设计具体的量化线性层，包括注意力层和MLP层都需要添加上量化算子，现在框架将qkv合并为一个大矩阵，还有注意力层的o矩阵和MLP层的gate_up、down矩阵，并且它们都是支持并行运算的，因此我们在linear.py中添加对量化算子的支持。

首先分析下原有的linear，基本的线性层是直接读入safetensor中的数据的，加上并行则是在weight_load中加上额外的切分逻辑，由于每台机器上都会有一个模型实例，因此都会读取其对应的数据，前向传播时处理这部分数据即可。

所以我们要做的就是增加对量化层的读取，同时加入W4A16的计算内核，先实现一个简化版本的代码，纯用pytorch进行反量化。

参照原有架构，先实现一个基础类，用于后续类的继承，与前面不同的地方在于需要多个矩阵——qweight、qzeros、scales、g_idx，都是由GPTQ量化算法决定的。注意其中矩阵的大小，qweight中权重量化成了INT4，因此可以8个一组变成INT32（查看safetensor中具体张量形状知道是行打包，一开始当成列打包十分痛苦，后续很多都需要修改，让claude sonnet修改，改了半天没发现是divisor的原因）；qzeros存的是每个量化组的0点，那有多少个量化组呢？首先是每个通道一组（列，让特征之间不互相影响），然后在行的维度分组（可以是乱序，idx就是用来索引的），因此总的组为in/group*out，又因为0点也分布在0~16，自然也是INT4，8个一组；scales因为受精度影响非常大，因此必须使用fp16.

~~~python
        self.qweight = nn.Parameter(torch.empty(input_size // self.pack_factor, output_size, dtype=torch.int32), requires_grad=False)
        self.qzeros  = nn.Parameter(torch.empty(input_size // self.group_size, output_size // self.pack_factor, dtype=torch.int32), requires_grad=False)
        self.scales  = nn.Parameter(torch.empty(input_size // self.group_size, output_size, dtype=torch.float16), requires_grad=False)
        self.g_idx   = nn.Parameter(torch.empty(input_size, dtype=torch.int32), requires_grad=False)
~~~

解包操作，GPTQ中为了最大程度利用并行化，将量化打包时是跳着来打包的，这样解包时每个线程读取的恰好是自己需要的数据，如果是连续打包，则需要线程间的通信。（注意！这里的算法有一个大坑，所有存储的数值，在“出厂”的时候都减去了1，如果不加上这个1，所有数字都会发生偏移，从而导致最终的输出崩盘！）

~~~python
        # qweight: 行方向打包，第 i 个 INT4 分量存储在每 pack_factor 行的第 i 位
        #   qweight[r, c] 存储了原始行 r*pack_factor ~ r*pack_factor+(pack_factor-1) 在列 c 上的 INT4 值
        for i in range(self.pack_factor):
            qweight_unpacked[i::self.pack_factor, :] = (self.qweight >> (i * 4)) & 0xF
        # qzeros: 列方向打包，第 i 个 INT4 分量存储在每 pack_factor 列的第 i 位
        #   qzeros[g, c] 存储了 group g 在列 c*pack_factor ~ c*pack_factor+(pack_factor-1) 上的零点
        for i in range(self.pack_factor):
            qzeros_unpacked[:, i::self.pack_factor] = ((self.qzeros >> (i * 4)) & 0xF) + 1
~~~

其余逻辑就是建立几个矩阵，把数据展开，然后再根据公式反量化，后续考虑使用triton优化。

有了父类后，后续只需要在此基础上进行数据填充即可，子类每次都会初始化父类的矩阵，然后再根据自身性质读取load_weight放入对应位置。如单个按列分的矩阵，计算出大小和偏移位置就可以直接塞入了。

~~~python
        divisor = self.pack_factor if (param is self.qzeros) else 1
        shard_size = param_data.size(self.tp_split_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_split_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)
~~~

如果是合并的按列分的矩阵，则稍微复杂些，需要先将参数进行TP分割，然后再填入到对应位置，注意，这里的参数都是量化后的参数，因此计算偏移以及大小时都要注意（按槽位分）。

~~~python
        inner_shard_size = self.output_sizes[loaded_shard_id]
        shard_offset = (sum(self.output_sizes[:loaded_shard_id]) // self.tp_size) // divisor
        shard_size = (inner_shard_size // self.tp_size) // divisor
  
        param_data = param_data.narrow(self.tp_split_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_split_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
~~~

QKV合并的矩阵也类似（注意有bias，需要单独处理下），最后是按行并行的线性层，其一般位于一个大层的最后一块，用于减少通信，前面按列并行的结果可以直接与我们按行并行的矩阵相乘得到中间结果，最后采用all-reduce即可。

~~~python
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param is self.bias:
            if loaded_weight.size(0) == param_data.size(0):
                param_data.copy_(loaded_weight)
            return
    
        shard_size = param_data.size(self.tp_split_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_split_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize()
        y = F.linear(x, weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
~~~

最后改造loader，新加量化权重读取逻辑，与原有逻辑二选一。整体思路是先把safetensor里面的数据改名，按照上述层名将QKV进行对应合并等，然后遍历模型层上的属性，一一调用模型层其自带的weight_loader函数就可以衔接上上面的逻辑了。

#### 初期结果

![20260307170533479.png](20260307170533479.png)

最终结果如上，prefill阶段挺快，29tps，但是decode阶段就非常慢了，只有4tps，被反量化操作严重拖累。同时注意到回答质量非常低，原因不明（反量化时出现了问题，见上）。

![20260307171238915.png](20260307171238915.png)

并行也成功运行了，prefill速度反而更慢了，decode阶段倒是变得更快了。分析——prefill阶段是计算密集型，分开来后通信代价变大了，导致速度变慢，同时decode阶段是内存密集型，加上反量化算子很慢（需要大量内存访问），因此分开后压力更小，速度更快。

奇怪的是，用完nvidia-smi后，prefill运行速度翻三倍了，但是decode阶段降三倍了

![20260307203844829.png](20260307203844829.png)

![20260307203846262.png](20260307203846262.png)

#### 优化算子

接下来是优化反量化算子，现在的逻辑非常简单，创造几个大矩阵，填进去就可以，先不改整体逻辑，就先优化掉for循环，此时是串行的，将其变成张量操作试试——原理是通过辅助张量与广播机制，原先要一列列访问，现在有大矩阵，就可以直接访问了，并行度++，显存压力++，decode加速2->3,prefill加速60->70，

~~~python
 def dequantize(self) -> torch.Tensor:
        """
        反量化过程（向量化高效版）：利用 PyTorch 向量化和广播机制将压缩的 INT32 权重还原。
        返回的权重形状: [out_features, in_features]
        """
        in_features = self.qweight.size(0) * self.pack_factor
        out_features = self.scales.size(1)
        # 1. 移位标量向量: [0, 4, 8, 12, 16, 20, 24, 28] 
        shifts = torch.arange(0, 32, 4, device=self.qweight.device, dtype=torch.int32)
  
        # 2. 向量化解包：避免昂贵且访存不连续的 Python for 循环
        # qweight: 按「行」打包。unsqueeze 并右移位后:
        # [in_f//8, out_f] -> [in_f//8, 1, out_f] >> [8, 1] -> [in_f//8, 8, out_f]
        # 使用 .view 展平时，由于 PyTorch 在内存上按行主序，它刚好将这 8 个值按 in_features 顺序交织。
        qw = ((self.qweight.unsqueeze(1) >> shifts.unsqueeze(1)) & 0xF).view(in_features, out_features)
  
        # qzeros: 按「列」打包。unsqueeze 并右移位后:
        # [in_f//group, out_f//8] -> [..., ..., 1] >> [8] -> [in_f//group, out_f//8, 8]
        # 展平时在列方向把压缩的值拉平。注意 GPTQ 需要隐式的 + 1 操作！
        qz = (((self.qzeros.unsqueeze(2) >> shifts) & 0xF) + 1).view(self.qzeros.size(0), out_features)
  
        # 3. 补齐分组维度 (Group broadcast)
        qz = qz.repeat_interleave(self.group_size, dim=0)
        sc = self.scales.repeat_interleave(self.group_size, dim=0)
  
        # 4. 反量化核心计算: fp16 = (int4_weight - int4_zero) * scale
        weight_fp16 = (qw.to(sc.dtype) - qz.to(sc.dtype)) * sc
  
        return weight_fp16.t()
~~~

#### 最终优化

在当前宏观层面，似乎做的已经到极限了，那就深入到底部去做优化，我们可以发现，对于数据的访问，永远都是先访问safetensor中的，然后将其读进预先设定好的矩阵里，最后再用矩阵进行计算，这样一来二去平白无辜的多出了很多的内存访问代码，直接深入到其中，取出来立刻算，当然中间结果也是要存的，但是把它们存到SM的SRAM中去，访问速度快快的。

写triton不难，关键在于划分边界以及处理边界情况，这里首先把输入变成一个大矩阵，利于后续计算与访问。同时还有个比较巧妙的，由于kernel内部假设权重分组是按照相邻来分的，因此没有使用g_idx，刚好对于这个模型是符合的，但对于其它模型就不一定符合了，因此需要引入idx，一般我们会想把qweight之类的统统转化下，但由于乱序排序，会出现跨线程访问的问题，代码极其难写，效率还不一定高，所以！我们直接把输入给排序了，即A@B=C中的A，如此我们就只需要改一个地方，完美适配kernel内部逻辑。

B中的行是乱序的，调整它并不会改变最终的结果，因为MK*KN=MN。同理，改变A的列也不会影响最终结果。从数学角度，A的列与B的行相乘，得到的是一个与结果矩阵一样大的矩阵，因此不管是第几列和第几行相乘，都不会影响形状，也就不会影响位置了，所以我们可以随意调整列的位置，将其按照正确位置摆放即可。

其中判断是否是递增序列的技巧非常秒，学到了！代码中还可以看到很多BLOCK大小，这里是根据经验写的，后续优化可能需要调整，当然这里调整了，kernel内部也需要调整。

~~~python
def triton_w4a16_gemm(a: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, group_size: int = 128, g_idx: torch.Tensor | None = None) -> torch.Tensor:
    """
    Triton-based W4A16 GEMM 算子的前端包装调度函数。
    a: [..., K] 激活张量
    qweight: [K//8, N] 打包好的权重
    qzeros: [K//group_size, N//8] 打包好的零点
    scales: [K//group_size, N] 缩放因子
    g_idx: [K] 可选，每个 in_feature 所属的 group 索引。
           当 desc_act=True 时，g_idx 不是顺序排列的，需要对 A 的列进行重排序。
    """
    # 将多维的 A (例如 bs, sequence_length, K) 展平为 2D 供底层的 GEMM 吃入
    # 这里为什么需要展平成二维？底层是无数个矩阵乘，直接拼接成一个大矩阵是等价的，
    # 但是展平成二维可以避免在 GEMM 时的内存访问模式不连续，从而提高访问效率。
    orig_shape = a.shape
    if len(orig_shape) > 2:
        a = a.view(-1, orig_shape[-1])
  
    # 处理 g_idx 重排序：
    # 当 g_idx 不是简单顺序 [0,0,...,0,1,1,...,1,...] 时（即 desc_act=True），
    # Triton 内核内部假设 K 维度按 group 顺序排列。因此我们需要在前端对 A 的列进行重排序，
    # 使得 A 的第 i 列对应 qweight 解包后的第 i 行，且 group 索引保持顺序。
    # 具体做法：根据 g_idx 排序得到 perm，然后 A = A[:, perm]。
    # 注意：qweight/qzeros/scales 已经按打包后的顺序存储，与 g_idx 排序后的顺序一致，
    # 因此只需要重排 A 的列即可。
    if g_idx is not None:
        # 检查 g_idx 是否已经是顺序的（即 desc_act=False）
        # 顺序的 g_idx 形如 [0,0,...,0,1,1,...,1,2,2,...] 即单调不递减
        # 好秀的代码，好巧妙的排序对象
        is_sequential = torch.all(g_idx[1:] >= g_idx[:-1])
        if not is_sequential:
            # g_idx 非顺序，需要重排 A 的列
            # argsort 得到排列顺序：将 g_idx 排序后，perm[i] 表示排序后第 i 个位置对应原始的哪个列
            # 即返回排列后的索引
            perm = torch.argsort(g_idx)
            a = a[:, perm]
  
    assert a.is_contiguous(), "Matrix A must be contiguous to ensure correct addressing."
  
    M, K = a.shape
    N = qweight.shape[1]
  
    # 预先分配空内存块用于装载产出结果
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # 分块策略 (Block sizes config)
    # M 分块大小根据是否有大批量的 Prefill 决定。
    # 较长上下文 (M大) 则选用 32 可增加复用，单步解码 (M很小一般等于1) 用 16 防止 padding 计算浪费
    # 这些参数有什么作用，为什么就是这些数字？
    # 根据 M 的大小，选择不同的 BLOCK_SIZE_M，以平衡内存使用和计算效率。

    BLOCK_SIZE_M = 32 if M > 16 else 16
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 128 
  
    # Triton 网格分配：M 分成多块，N 分成多块。这是一个 1维 (Tuple维度=1) 的启动网格配置
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # 启动内核 (Launch Triton kernel)
    _w4a16_gemm_kernel[grid](
        a, qweight, c,
        scales, qzeros,
        M, N, K,
        a.stride(0), a.stride(1),
        qweight.stride(0), qweight.stride(1),
        c.stride(0), c.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        group_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=4,     # 一个 Warp 是 32 线程。配置 4 个 Warps (即 128 线程)
        num_stages=2,    # 开启流沙缓冲(Software Pipelining)，重叠访存与计算
    )
  
    # 在最后把它复原成原始送进来的多维度
    if len(orig_shape) > 2:
        c = c.view(*orig_shape[:-1], N)
  
    return c

~~~

kernel内部就是找到地址，然后读取计算。草图辅助理解![20260308161056511.png](20260308161056511.png)

~~~python
@triton.jit
def _w4a16_gemm_kernel(
    # 指针
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr,
    # 矩阵维度 a: [M, K], b_packed: [K//8, N], c: [M, N]
    M, N, K,
    # 步长信息 (Stride) - 用于计算非连续内存的确切物理地址
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_sk, stride_sn,
    stride_zk, stride_zn,
    # Meta-parameters
    group_size: tl.constexpr,   # 常见的 group_size 是 128
    BLOCK_SIZE_M: tl.constexpr, # M 维度的分块大小 (通常在 prefill 时较大，decode时等于 16 或 32)
    BLOCK_SIZE_N: tl.constexpr, # N 维度的分块大小
    BLOCK_SIZE_K: tl.constexpr, # K 维度的分块大小 
):
    """
    Triton 算子网格 (Grid) 级别调度：
    每个 Program ID (pid) 负责计算 C 矩阵上的一个 BLOCK_SIZE_M x BLOCK_SIZE_N 块。
    由于 A 矩阵形状为 M x K，B 矩阵（逻辑上）为 K x N，我们要在这个内核里循环跨越 K 维度。
    """
    pid = tl.program_id(axis=0)
    # 按从左到右，从上到下(行主序)划分网格区块
    # 因为c中的每一列都依赖B中的每一列，因此如果按列顺序排序，有助于利用空间局限性
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    # 计算出每一个grid负责处理的 M 和 N 的具体坐标索引
    # tl.arange 产生诸如 [0,1,2,3...] 的连续数组，常用于向量化取数据
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    # K 维度的索引。每次内循环(循环一次K即为一块处理)，我们步进 BLOCK_SIZE_K 个元素
    offs_k = tl.arange(0, BLOCK_SIZE_K)
  
    # 针对打包后的 qweight，它的行(in_features维度)被压缩了 8 倍
    offs_k_packed = tl.arange(0, BLOCK_SIZE_K // 8)
  
    # 移位量预计算 [0, 4, 8, 12, 16, 20, 24, 28] 
    # 每个 INT32 可含 8 个 4-bit
    b_shifts = tl.arange(0, 8) * 4

    # 初始化累加器，必须为高精度以防溢出 (float32 常规用法)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 主循环 (Inner Loop)：沿着 K 维度累加 (A 的列和 B 的行做内积)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # -----------------------------------------------------
        # 1. 载入 A 的一部分 (激活 Tensor)
        # a_ptrs 将指向外存 A 矩阵在当前(偏移m, 偏移k)的地址
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + (k * BLOCK_SIZE_K + offs_k[None, :]) * stride_ak)
        # 用 mask 防止在尾部遇到不能被分块整除的 K 导致的越界段越界读取
        a = tl.load(a_ptrs, mask=(k * BLOCK_SIZE_K + offs_k[None, :]) < K, other=0.0)

        # -----------------------------------------------------
        # 2. 载入量化权重 B 及拆包 
        # qweight 格式(HuggingFace GPTQ)：[in_features//pack_factor, out_features] 即按 K 打包
        b_ptrs = b_ptr + ((k * (BLOCK_SIZE_K // 8) + offs_k_packed[:, None]) * stride_bk + offs_bn[None, :] * stride_bn)
        b_packed = tl.load(b_ptrs) # [BLOCK_SIZE_K // 8, BLOCK_SIZE_N]
  
        # 向量化位移拆包: 利用广播把包含8个参数的INT32劈裂开
        # shape 变为 [BLOCK_SIZE_K // 8, 8, BLOCK_SIZE_N]
        b_unpacked = (b_packed[:, None, :] >> b_shifts[None, :, None]) & 0xF
  
        # 使用 reshape 揉合成连续排列的二维块：[BLOCK_SIZE_K, BLOCK_SIZE_N]
        # 设置 can_reorder=True 可让 Triton 编译器依据硬件情况优化布局警告
        b_unpacked = tl.reshape(b_unpacked, (BLOCK_SIZE_K, BLOCK_SIZE_N), can_reorder=True)

        # -----------------------------------------------------
        # 3. 计算本块所处的 Group Index (量化是对一定大小内进行共享标量的组量化)
        # 通常 BLOCK_SIZE_K <= group_size (如 128) 并且 K 恰好是倍数。为了简单这里假设块与分组对齐。
        # 如果没有对齐，计算稍微复杂一点，按照第几行去求，并且可能被分到不同组，后续逻辑会更复杂
        k_group = (k * BLOCK_SIZE_K) // group_size
  
        # 载入 Scales (FP16格式)
        s_ptrs = scales_ptr + (k_group * stride_sk + offs_bn * stride_sn)
        scales = tl.load(s_ptrs) # [BLOCK_SIZE_N]

        # -----------------------------------------------------
        # 4. 载入 Zeros 及拆包
        # Zeros 的打死包格式是按 `out_features`(N) 方向压缩的！这跟 qweight 正好相反！
        # zeros_packed shape: [in_features//group_size, out_features//pack_factor]
        offs_bn_packed = offs_bn // 8
        z_shifts = (offs_bn % 8) * 4 # 计算出每个列索引被塞在哪一个 INT32 中的哪一阵 4 bit 里
  
        z_ptrs = zeros_ptr + (k_group * stride_zk + offs_bn_packed * stride_zn)
        zeros_packed = tl.load(z_ptrs)
  
        # 拆解出本块所有 N 的零点，加上极端的 GPTQ `+1` 偏移约定
        zeros_unpacked = ((zeros_packed >> z_shifts) & 0xF) + 1

        # -----------------------------------------------------
        # 5. 最后就位：反量化转浮点及矩阵乘
        # 在这最最关键的一步，所有的变量(b_unpacked, zeros_unpacked, scales) 目前都安然寄居在极高带宽的 SRAM 里。
        # 利用 (int - zero) * scale 得到真实的重构权值。
        b_fp16 = (b_unpacked - zeros_unpacked[None, :]) * scales[None, :]
        b_fp16 = b_fp16.to(tl.float16) # Cast 到 16bit 以喂入 Tensor Cores
  
        # 使用极其高效的 Tensor Core 乘加：[BLOCK_M, K] @ [K, BLOCK_N]
        accumulator += tl.dot(a, b_fp16)

    # -----------------------------------------------------
    # 主循环结束后：把当前 Thread 累积的最终结果塞回 HBM
    # 将高精度的 accumulator 转回 FP16 存储。
    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator.to(tl.float16))
~~~

最终结果，prefill从70-80，decode从6-8

![20260308161356375.png](20260308161356375.png)

### MoE

为了进一步支持投机解码，必须要词表相互适配的MoE模型，于是Qwen3-30B-A3B-GPTQ-Int4来了，与此前不同的是，MLP层变成了需要门控的多个FFN，修改主要也是针对它进行适配。
首先是loader函数，使用正则判定找到对应experts的参数，调用模型层的weight_load函数；模型层方面，只需要修改MLP，先构造门控，再进行多选，初始阶段使用的是Huggingface/Qwen模式，即循环顺序访问每一个experts进行前向传播，显然太木讷了，后续需要优化。MLP层读取参数需要注意都需要增加一个experts维度，除此之外没有什么特殊的了。

上面提到的优化experts前向传播，我们想要的是一次性把所有计算算完，因此把每个experts负责的矩阵变成一个大矩阵，同时传入experts间的分割线，在内部算子中，为每个experts都线启动一个线程，然后像之前的FFN层一样再进行划分，这里肯定会有多余的线程，因此只需要把越界的block提前返回，当然，这样还是进行了实际分配，存在大量的资源浪费，是后续可以优化的点。（kernel的对齐还是有些恶心的，并且随着矩阵的形状改变，线程的分配也会变得复杂，性能也没有那么好。

对于MoE的并行，需要注意门控层对于精度特别敏感，而这点由于不同机器对于浮点数的执行可能不同，存在一定的误差，同时由于硬件对矩阵乘的执行策略不同，被拆分乘了多个小block，执行顺序不保证，且浮点数加不满足交换律，存在误差。将route_logits多方同步就可以解决了。

### 调度系统优化

当前调度策略是始终优先prefill阶段，如果prefill计算量比较大同时有源源不断的prefill，将会带来大量的decode等待时间，导致整体吞吐会变低，TPOT变得很长，测试结果如下：

为优化，采取现在普遍使用的chunk_prefill，将第一次请求分成一块块的，可以减少每次请求需要的资源以及计算时间
