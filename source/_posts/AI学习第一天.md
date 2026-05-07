---
title: AI学习第一天
date: 2025-07-29 15:19:42
categories:
- AI与深度学习
tags:
- Embedding
- Transformer架构
- 注意力机制实现
- GQA
- 多头注意力
---
现在AI发展的太快了，原本以为CV只能解决图像识别、分割等任务，没想到还可以用来生成2D，3D图像；音频处理就更是强大了，自动配音，自动将音频与视频对齐，行业壁垒要被冲烂了。

AI领域相当重要的嵌入（embedding），目的是让机器可以看懂人类语言，最初想法是每一个词就是一个数字，但这样一会导致数字无比巨大，二会导致语义间的联系不大（语义割裂），例如a和an距离就会比a和abandon更大，但这些都还不能解释为什么数字要和词具有关联。

根据[没有思考过Embedding，不足以谈AI](https://mp.weixin.qq.com/s/7kPxUj2TN2pF9sV06Pd13Q)所说，

+ 让AI更好理解句子语义，即通过词的相近语义，让整个句子在空间里表现出形状。
+ 让模型结构设计更加自由——将文本变为”模糊“的概率列，再通过映射成文字，这样就让模型发挥空间更大了。

> 因为词是离散分布的，而计算模型的输出 —— 除非只使用非常简单的运算并且约束参数的权重 —— 很难恰好落在定义好的量化值中。

在我的理解中，AI还处于模仿人类思维，因此其知识空间应该也与人类相似，文本向量可以理解为情感空间，每个词在情感空间都有一个对应的点，全部维度可以唯一确定一个文字。自然，情感也包含里语义。

transformer架构看起来很简洁，一个编码器一个解码器，编码器是解析文字输入，解码器把已生成的输出变回向量，编码器就是简单的注意力层加全连接层；解码器首先用多头注意力层，加上注意力层和全连接层，每层连接都加上了残差连接，且第二个注意力层加上了掩码机制

![1755418619305](1755418619305.png)

Key、Value、Query，是注意力层从左到右的三个输出，在自然语言处理中，key和value是同一个，query表示当前输入向量，通过query和key进行操作，得到加权系数，再用加权系数把value聚合得到最终的注意力分数。

注意力机制的实现：

~~~python
def scaled_dot_product_attention(Q, K, V):
    attn = Q @ K.transpose(-1,-2) /math.sqrt(K.size(-1))
    score = torch.softmax(attn,dim = -1)
    return score @ V
~~~

多头注意力机制实现：

~~~python
class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        # Initialize W_q, W_k, W_v, W_o
        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)
        self.num_heads = num_heads

    def forward(self, Q, K, V):
        b , sq , d = Q.shape
        sk = K.size(1)
        q = self.W_q(Q).view(b,sq,self.num_heads,-1).transpose(1,2) # seq_q,hidden/head
        k = self.W_k(K).view(b,sk,self.num_heads,-1).transpose(1,2) # head,seq_k,hidden/head
        v = self.W_v(V).view(b,sk,self.num_heads,-1).transpose(1,2)
        attn = q @ k.transpose(-1,-2) / math.sqrt(K.shape[-1]/self.num_heads) # head,seq_q,seq_k
        score = torch.softmax(attn,dim=-1)
        o = score @ v # head, seq_q,hidden/head
        return self.W_o(o.transpose(1,2).reshape(b,sq,-1))
~~~

掩码注意力机制的实现

~~~python
def causal_attention(Q, K, V):
    attn = Q @ K.transpose(-1,-2) / math.sqrt(K.shape[-1])
    mask = torch.triu(torch.ones_like(attn,device=attn.device, dtype=torch.bool),diagonal=1)
    attn = attn.masked_fill(mask,float("-inf"))
    score = torch.softmax(attn,dim = -1)
    return score@ V
~~~

组查询注意力（GQA）机制的实现

~~~python
class GroupQueryAttention:
    def __init__(self, d_model, num_heads, num_kv_heads):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_kv= nn.Linear(d_model, 2 * num_kv_heads * d_model // num_heads)
        self.W_o= nn.Linear(d_model, d_model) 

    def forward(self, x):
        b,sq,d = x.shape
        q = self.W_q(x).view(b,sq,self.num_heads,-1).transpose(1,2) # batch,heads,seq,hidden
        k ,v= self.W_kv(x).chunk(2,dim = -1)
        k = k.view(b,sq,self.num_kv_heads,-1).transpose(1,2).repeat_interleave(self.num_heads // self.num_kv_heads,dim = 1) # batch,heads_kv,seq,hidden
        v = v.view(b,sq,self.num_kv_heads,-1).transpose(1,2).repeat_interleave(self.num_heads // self.num_kv_heads,dim = 1) # batch,heads_kv,seq,hidden
        attn = q @ k.transpose(-1,-2) / math.sqrt(k.size(-1)) # batch,heads,seq,seq
        score = torch.softmax(attn,dim=-1) 
        o = self.W_o((score @ v).transpose(1,2).contiguous().view(b,sq,-1)) # b,h,s,d
        return o
~~~
