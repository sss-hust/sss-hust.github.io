---
title: transformer优化学习
date: 2025-10-10 19:29:51
tags:
- FlashAttention
- KV Cache
- MQA/GQA
- 在线Softmax
- 注意力优化
mathjax: true
categories:
- AI与深度学习
---
注意力机制出现后，模型参数数量得到极大的提高，日渐庞大的参数量，使得训练与推理的性能成为一个瓶颈，因此出现了很多优化方案。由于注意力分数矩阵大小为n*n，则时间、空间复杂度都是O($n^2$)

多头注意力机制其实就是把特征分块，然后每个头只聚焦一部分的特征，相比于之前的大矩阵，更加细腻。

## 注意力机制源码实现

```python
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # 简洁的写法，乘完后一刀切
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        # view在连续内存中改变组织方式，transpose只改变步长（读的跳跃步数），reshape可以改变不连续的内存（深copy）
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
```

## KV cache

当前主流语言模型使用的架构为单decode模型，首先是prefill阶段,将输入的prompt转化为注意力矩阵（详解见引用），这时候就可以将计算的K、V存储在cache中。因为生成都是依靠最后一个token的概率进行采样，因此下一个输出的计算只和当前Q以及过去K、V，因此存在大量的重复计算，所以把KV缓存起来，大致原理如下图

> prefill阶段，输入的prompt分解成一个个token，按行排布，然后计算KQV，同时KQ相乘得到注意力分数，这时掩码矩阵将不该计算的置为负无穷后再softmax计算。同时这时的K可以存储起来，接着注意力分数作为权重，将当前token之前的V进行加权求和，算出来的V也存储起来，等待推理时使用，最终注意力向量加到原先的向量上，完成注意力层计算。

![](https://pic2.zhimg.com/v2-655b95ebfb7808563bead28bc89bb459_1440w.jpg)

在多头注意力中，Q头和KV头是一一对应的，产出的KVcache非常大，于是就有人提出了多查询单键值（MQA）和多查询多键值（MQA），其根本思想是压缩KV的系数矩阵，使得token与它们相乘的结果变小，比如32头原先K大小为[4096,4096]，现在变成8头后，就是[4096,1024]，kvcache大小缩小到原来的四分之一，当然代价是模型表达能力下降。此时Q头依然是32，怎么乘呢？只需要复用K头，比如前四个头用K的第一个头，以此类推。

## FlashAttention

技巧一，将大量softmax操作分块，让小块能够存进最快但最小的SRAM中，从而提高IO速度，原理如下

$$
m(x):=\max_{i}x_{i},\quad f(x):=\begin{bmatrix}e^{x_{1}-m(x)}&\ldots&e^{x_{B}-m(x)}\end{bmatrix},\quad\ell(x):=\sum_{i}f(x)_{i},\quad\mathrm{softmax}(x):=\frac{f(x)}{\ell(x)}.
$$

单独的块可以这样求出局部值，每个块都保存局部$l(x)、m(x)、softmax=O$

$$
\begin{aligned}&m(x)=m(\left[x^{(1)}\:x^{(2)}\:\right])=\max(m(x^{(1)}),m(x^{(2)})),\quad f(x)=\left[e^{m(x^{(1)})-m(x)}f(x^{(1)})\quad e^{m(x^{(2)})-m(x)}f(x^{(2)})\right],\\&\ell(x)=\ell(\left[x^{(1)}\:x^{(2)}\right])=e^{m(x^{(1)})-m(x)}\ell(x^{(1)})+e^{m(x^{(2)})-m(x)}\ell(x^{(2)}),\quad\mathrm{softmax}(x)=\frac{f(x)}{\ell(x)}.\end{aligned}
$$

$$
O_{new} = \frac{l_{old} \cdot e^{m_{old} - m_{new}} \cdot O_{old} + \tilde{l} \cdot e^{\tilde{m} - m_{new}} \cdot \tilde{O}}{l_{new}}
$$

融合两个块时，分子只需要乘以真正的缩放系数，分母统一更新即可。

其中涉及的小技巧如下：

$$
e^{x_i-m}=e^{(x_i-m^{(k)})+(m^{(k)}-m)}=e^{x_i-m^{(k)}}\cdot e^{m^{(k)}-m}
$$

问题——为什么要把k分成这样，以及q的分块是否有技巧；flashattention怎么和kvcache配合
