---
title: nano-vllm
date: 2026-02-13 22:04:12
categories:
- AI与深度学习
tags:
- vLLM
- PagedAttention
- KV Cache
- Triton
- CUDA Graph
---
[toc]

正式开始学习nano-vllm，希望通过这个项目，对推理这个方向能有个大概的认识。

首先程序入口是bench.py和example.py这两个文件，区别在于两者的输入prompt属性，前者是自动测试并发能力，生成大量的无意义id，无须准确回答，只是用来看数据；后者可以测试具体的输入，让模型回答准确内容，由于后端能力，也支持并发输入。

## llm_engine

接着就进入引擎的总入口——llm_engine.py，它负责将用户传过来的prompt转化成输出，因此它的任务就是这么简单，因此只提供相应的接口——首先是最经典和最主要的generate，它把收到的输入全部传入调度器，然后再由调度器给出先计算谁，接着执行单步推理就可以了。如此简单的背后蕴含着丰富的封装，首先每个输入都会先被sequence包装好再送到调度器，其次调度器实现continually batch，最后是运行模型前向传播等的类，我们将持续展开。

## schedule

Q：为什么需要max_batch和max_tokens的限制？

A：前者控制decode阶段的并发数，防止并发数量过多导致管理开销过大，后者无法实现这个功能；后者则应对prefill阶段的大批量数据，防止由于单批次的输入过长导致显存爆炸。

Q：如何确定这两个参数的最优值？

A：根据需求和最佳性能来判断，如果是对响应速度要求高，那就需要减小并发数；如果对吞吐量有要求，就需要调大两个值，并且需要和两个阶段的特点进行适配，并且还可以运行基准测试，寻找最佳表现点。

调度器调度基本单位是seq，维护等待、工作队列，上层添加的请求在这里都会一一加入等待队列，等需要调度时，从这个等待队列里面去拿。调度的策略是优先保证prefill请求,只要有请求在队列中，就意味着是prefill请求，因为prefill结束后对应seq依然在run队列中，只不过没有让它去运行。

prefill调度比较简单，如果有足够多的块，就可以运行，为其分配块，然后加入run队列，注意这里是一直循环，直到批次上限或者没有请求了才会停止。然后将所有这轮调度的请求发回给engine，engine再进行前向传播。

```python
    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        核心调度逻辑。
      
        Returns:
            tuple[list[Sequence], bool]: 
                - 本次被调度的序列列表
                - bool 值: True 表示当前是 Prefill 阶段，False 表示 Decode 阶段。
        """
        # ===========================
        # 1. 尝试调度 Prefill (预填充) 任务
        # ===========================
        # 策略：只要有等待的任务，且显存/BatchSize允许，就优先执行 Prefill。
        # 这种实现方式意味着 Prefill 和 Decode 不会在同一个 step 中混合执行。
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
      
        # 遍历等待队列
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
          
            # 检查1: 是否超过了最大 batched token 限制
            # (Prefill 阶段 token 数增长很快，需要严格限制以防 OOM)
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens:
                break
              
            # 检查2: 询问 BlockManager 是否有足够的空闲显存块来容纳该序列的 KV Cache
            if not self.block_manager.can_allocate(seq):
                break
          
            # --- 资源检查通过，开始分配 ---
            num_seqs += 1
            # 在 BlockManager 中实际分配物理块
            self.block_manager.allocate(seq)
          
            # 计算本次 batch 增加的 token 数
            # 注意：如果使用了前缀缓存(Prefix Caching)，实际计算的只有未缓存的部分
            num_batched_tokens += len(seq) - seq.num_cached_tokens
          
            # 更新状态为 RUNNING
            seq.status = SequenceStatus.RUNNING
            # 从等待队列移除，加入运行队列
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
          
        # 如果调度到了 Prefill 任务，立即返回。
        # 第二个返回值 True 告诉 ModelRunner 执行 Prefill 模式的前向传播。
        if scheduled_seqs:
            return scheduled_seqs, True

        # ===========================
        # 2. 调度 Decode (解码) 任务
        # ===========================
        # 只有在没有 Prefill 任务（或显存不足以运行新的 Prefill）时，才执行 Decode。
      
        while self.running and num_seqs < self.max_num_seqs:
            # 取出一个正在运行的序列
            seq = self.running.popleft()
          
            # 检查显存: 下一个生成的 token 是否需要分配新的显存块？
            # 如果当前块还没满，can_append 为 True；如果满了且有空闲块，也为 True。
            while not self.block_manager.can_append(seq):
                # --- 显存不足，触发抢占 (Preemption) ---
                # 显存不够当前序列生成下一个 token 了，需要踢掉一些正在运行的任务来腾位置。
              
                if self.running:
                    # 策略：牺牲运行队列尾部（最近加入或优先级最低）的任务
                    victim_seq = self.running.pop()
                    self.preempt(victim_seq)
                else:
                    # 极端情况：队列里只剩当前这一个任务了，还是显存不足
                    # 只能抢占自己，停止运行
                    self.preempt(seq)
                    break # 跳出内层循环，该序列本次无法调度
            else:
                # --- 显存充足 (或通过抢占腾出了空间) ---
                # python 的 while-else 语法：如果 while 没有被 break 中断，则执行 else 块。
              
                num_seqs += 1
                # 告知 BlockManager 准备追加 token (可能会分配新块或更新哈希)
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
      
        # 确保肯定调度到了任务（除非 running 队列本来就是空的）
        assert scheduled_seqs
      
        # 将本次调度的序列放回 running 队列的头部
        # 这样保持了队列的循环顺序，或者是为了让活跃的任务保持在队列前列
        self.running.extendleft(reversed(scheduled_seqs))
      
        # 返回 False，告诉 ModelRunner 执行 Decode 模式
        return scheduled_seqs, False
```

即另外一个接口，同时还引入了block管理类，实现pageattention，在这些基础上添加运行队列，像操作系统那样去调度每一个请求，请求的优先级是先处理prefill，有prefill就不调度decode；同时调度decode时，会把其放在运行队列最前端，保证短任务优先完成。如果没有足够显存，就实行抢占策略，优先提供decode，保证用户体验感。

调度回来后由schedule进行后处理，输入是前向传播得到的新token_id，这里先调用seq，将新生成的token_id加入到原先的序列中，然后检查是否到达结束条件，如果到达，就从运行队列中移除，并释放显存，否则就放回运行队列。

如果是decode调度，就是从running队列中拿出想要的，然后为其分配显存，如果显存不够，就进行抢占（被抢占的回到running队头，下次优先调度），优先保证当前decode。进行调度时会先通知block_manager告诉它可能要更新块了。当然，新被调度的seq还是会加入running队列，并且挤在最前头，资源不够的话，后面的请求只能等了。

### senquece

承接请求的物理实体，将每个请求变成一段段管理，方便kvcache以及信息输出，同时还提供了分布式通信的方法，可以把prefill阶段全部prompt发给其它机器，或者发送decode阶段最新的，同时也可以接收别人传过来的数据并还原。（其实就是把下面记录的东西都发过去）

每个请求进来后，保存其对应token_id，其中单独保存最后一个token的id，用于decode阶段。并计算当前token需要多少个块，以及一些用于统计输出的变量。

### block

Q：大数据hash是如何实现的？

A：将大数据分块，通过对每一段的二进制进行不可逆的数学运算，再将最终结果合并

Q：为什么block里面的块不是二对一（K、V对应Q）？

A：这里只是逻辑块号，在实际的物理块中，有两个数据结构用于存储K和V。

整体分为两部分，管理块的类和块类，所有存储都按块来管理，将其挂在不同队列上（使用或者未使用），一个块包含这些内容——引用计数（用来判断当前块是否被多个token块共享，即具有相同前缀）、hash索引（用以快速找到实际的物理块号）、token_id（用以hash索引碰撞时判断是否相同），整个大类都是围绕这三个维度进行的，包含三类函数——分配块、回收块、判断是否可分配，同时对于decode长度动态性，单独用两个函数判断，如果发现当前块不够，就把这个请求先搁置，否则根据decode长度判断是否需要新分配块。

在分配块时，先每个块确定哈希索引（如果块未满就不算，等后续加入再算；同时会传入之前的哈希索引，用以代表其前缀），然后去找是否有匹配的，匹配上就无需重复分配，直接引用，并记录当前seq使用了多少个缓存token（这里还有细节，命中分为正在使用的块和用完的块，用完的块采用懒回收，仅仅把块放到空闲队列，不直接回收）；未命中就直接分配。最后将分配的快好加入到seq的块表中。

释放显存时，会从最后一个块开始释放，因为它被共享的可能性最低，尽快腾出空闲块，同时只清理seq中的块表，但是块里保留的hash以及tokenid并没有释放，如果后续对应上了，就可以直接引用，无需重新计算。只有后续被分配给别的块时才真正删除。

当收到可能更新块的信号，就会进行判断如果已满，就分配新块。如果未满且新增也不满，就什么都不做，否则更新hash

## model_runner

Q:为什么要使用cuda_graph？它有什么缺点？

A：GPU的行为需要由CPU来控制，而在decode阶段，每个计算时间非常短，等待CPU发命令时间变得很长，因此不如把先前的计算图依赖关系记录下来，当下次需要前向传播时，CPU只需要下一条命令就可以全部执行，缺点是批次变化后需要重新录制图，显存空间占用会变大，用空间换时间。

Q：建立分布式通信时，为什么使用TCP，这样不会更慢嘛？

A：TCP只是一开始使用，用于让大家都知道有谁上线了，后续由后端NCCL自动判断走哪里最快。

在engine中，每一个进程都会单独建立一个model_runner，进程间通信通过共享内存实现。

runner负责上层与模型层的交接，模型骨架以及参数读取都是在这进行的，初始化时就会进行一次热身，执行一次最大允许的prefill，以确定显存占用情况，运行时多余的空间完全分配给KVblock，这里采用所有层使用同一个块号，在前向传播时就可以使用一个统一的块号进行每一层的kvcache访问，真是一个好设计。kvcache的物理块已经在这里就分配好了，后续attention层只需要填进对应的块就行（已经映射好了，每一层都觉得自己在访问本地创建的，其实是统一创建的），以及逻辑kvcache块号也是在这里确定的。

它给外界的接口就是call，通过它调用内部函数，同时把指令写到其它rank的共享内存中，一起同步执行。

prefill请求进入run后，会依次访问每一个token_id，并为后续attention计算做好准备，比如将需要计算的tokenid放入input中（缓存命中的就不需要计算了，直接访问对应的kvcache），以及各token的绝对位置，同时记录其对应长度（便于后续展平寻找对应请求块），以及最关键的kvcache要写入的位置（通过block_ids与block_size算出每一个token要写入的绝对位置）。最后如果存在缓存命中，就需要将所有请求的块号对齐（flashattention需要，简化kernel逻辑）。将以上所有信息都写入全局上下文，就可以实现引擎层的参数快速简洁传递到模型的每一层。

decode进入run，依旧是准备好未来需要的信息，相较于prefill，decode每个请求只有一个token（但是不要陷入误区，由于token是用seq管理的，seq中还是保留了全部的token），因此需要的信息只有——token_id、绝对位置、kvcache写入位置,以及请求的总长度（用于告诉flashattention回看多久的历史）。如果使用cuda graph技术，就会预先把所有需要用到的信息留好位置，并把计算过程记录下来，下次直接调用就行，但是缺点是批次变化后需要重新录制图，显存空间占用会变大，用空间换时间。

从底层返回的是logits，在这里进行采样，并把新生成的token_id传给上层。

## 模型层

从runner进入模型层后，上层传递的是输入的id和对应位置，首先经过embedding层，将id转换为词向量（这里采用词表并行，每一个GPU负责一部分词表，id变词表的过程就是找到对应序号就可以，多卡注意区间，最后累加），然后依次经过注意力层、MLP层（非常多这样的块），每层之间都会有归一化层。

注意力层中，输入的是token索引号、隐藏向量，进去后首先得到QKV（这里的行长度是多个请求混杂在一起的），然后进入关键的attn计算，这里一进来会调用triton kernel，将刚刚新计算的kv放进kv缓存中（利用全局上下文中的slot_mapping，如下，每个线程负责读取写入一个token的kvcache，这也是物理位置的最小计算单位，由于在triton看来，所有矩阵都是一片连续的地址，因此想要读出新算的kv，就要根据起始点和偏移量进行计算，当然，要写入的位置也是需要这样算，然后一个一个写入）。decode阶段也类似，只不过是计算量以及并行度变少，不用因果掩码了。

```python
@triton.jit  # @triton.jit: 标记该函数为 Triton JIT 内核，会被编译成 GPU 机器码。
def store_kvcache_kernel(
    # --- 输入参数 ---
    key_ptr,            # 当前 Token 的 Key 向量的起始显存地址（指针）
    key_stride,         # Key 张量中相邻 Token 之间的步长（stride），即从 Token i 跳到 Token i+1 要跨越多少元素
    value_ptr,          # 当前 Token 的 Value 向量的起始显存地址
    value_stride,       # Value 张量的步长
    k_cache_ptr,        # KV Cache 中 Key 部分的起始显存地址（整个 Cache 的入口）
    v_cache_ptr,        # KV Cache 中 Value 部分的起始显存地址
    slot_mapping_ptr,   # slot_mapping 数组的起始显存地址。
                        # slot_mapping[i] 表示第 i 个 Token 应该写入 Cache 中的第几个 slot。
    D: tl.constexpr,    # 每个 Token 的 KV 向量总维度 = num_kv_heads * head_dim。
                        # tl.constexpr: 编译时常量，Triton 编译器会据此优化循环展开和寄存器分配。
):
    """
    GPU 并行写入内核：将每个 Token 的 Key/Value 向量写入 KV Cache 的指定位置。
  
    并行策略：每个 GPU 线程块（Program）负责处理 1 个 Token。
    总共启动 N 个线程块（N = Token 总数），所有 Token 的写入并行执行。
    """
    # tl.program_id(0): 获取当前线程块的编号（类似 CUDA 中的 blockIdx.x）。
    # 每个线程块负责将第 idx 个 Token 的 KV 写入 Cache。
    idx = tl.program_id(0)
  
    # 从 slot_mapping 中读取第 idx 个 Token 应该写入 Cache 的哪个 slot。
    # slot 是 KV Cache 中的一个"格子编号"，由 BlockManager 预先计算好。
    # 例如 slot = 512 表示要写入 KV Cache 的第 512 个位置。
    slot = tl.load(slot_mapping_ptr + idx)
  
    # 如果 slot == -1，说明这个 Token 不需要写入（可能是 CUDA Graph padding 出来的无效 Token）。
    if slot == -1: return
  
    # 计算当前 Token 的 Key 向量在输入张量中的偏移地址。
    # key_offsets = 第 idx 个 Token 的起始位置 + [0, 1, 2, ..., D-1]
    # tl.arange(0, D): 生成一个从 0 到 D-1 的序列（类似 Python 的 range(D)）。
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
  
    # tl.load(): 从显存中批量读取 D 个元素（整个 KV 向量）。
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
  
    # 计算写入 Cache 中的目标偏移地址。
    # cache_offsets = slot * D + [0, 1, 2, ..., D-1]
    # 即 Cache 中第 slot 个位置的起始地址加上向量内部的偏移。
    cache_offsets = slot * D + tl.arange(0, D)
  
    # tl.store(): 将 Key/Value 向量批量写入 KV Cache 的指定位置。
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

写完kvcache后，就要进行计算了，prefill阶段对应的是变长矩阵乘法，这里以自己写的为例，输入是所有请求的qkv以及对应的全局信息，同时注意grid的切分，首先是最小的维度，因为在SM内部，先变化x维度去更换block，所以为了L2缓存最大命中，把小的放前面比较合适，同时后面的设计稍微有点问题，因为正常来说先head这个更低的维度会更好，但这里为了简化kernel内部逻辑，采用更低效的写法，高效写法是直接传输batch与head相乘，内部再用逻辑进行区分。算子整体思路就是把Q矩阵按照batch和head进行分配，同时内部按照行再进行切割，一个块就是一个block，kv也是这样分块，但是在block内部分的，利用for循环去遍历这一部分（因为KVcache一般比较大，无法全部放在SM内，所以需要分块计算），更新就是按照数学原理进行了。MLP层就没什么说的了，不管是稠密模型还是稀疏模型，本质上都还是FFN层，无脑乘就可以了，最后每个token都变成了一次前向传播后的结果，我们只需要取每个请求的最后一个token即可，将其传回runner。

```python
   grid = (triton.cdiv(max_seqlen_q, BLOCK_M), batch_size, num_heads)
```

## 并行通信部分

每当主进程调用run中的接口时，会自动触发seqence中的上下文函数，将其发送给其它机器，其它机器就不光会收到指令，还有执行下去的全部数据。

## 运行结果

![20260307171529592.png](20260307171529592.png)

![20260307171543190.png](20260307171543190.png)

decode阶段变慢很好理解，但是prefill阶段反而变快了，并且是线性关系，同时输入的上下文如下：

```python
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
```

ai是这样解释的——prefill阶段分两张卡读权重，相当于带宽翻倍了，在计算量并不大的时候，变成了内存限制，因此分卡有收益，是这三者的权衡：内存访问、矩阵运算、卡间通信。

以下是自定义triton flashattn算子与官方算子的比较，最后是开启cuda graph的结果。实在是恐怖如斯，官方算子比我的接近一倍提速

![20260314162119054.png](20260314162119054.png)
