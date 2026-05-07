---
title: 手撕MoE代码
date: 2025-10-07 20:04:39
categories:
- AI与深度学习
tags:
- MoE门控机制
- 负载均衡损失
- SwiGLU
- Top-K路由
- Mixtral实现
---
此前略读了moe发展历程的论文，发现存在一大段的空白时间，在1991年jordan就已经提出了专家系统的雏形，但是直到2017年才由谷歌实现第一个大规模MoE+RNN模型，不由得去思考，为什么是在这个时间点出现。

通过AI辅助分析，我认为有以下几点（暂未深入分析，个人猜测）：

- 显卡发展较慢，在2016年以前显卡暂不具备大规模运算能力，与moe巨大参数相违背，在2016年出现GTX系列，晶体管数量暴涨，才具备一定的基础能力。
- 分布式训练技术不成熟，由于单卡显存限制，大参数模型训练存在较大问题，在2016年pytorch发布与tensorflow的发展，相关生态才开始好起来。
- 开始有大规模模型的需求，同年谷歌发布至今仍沿用的transformer架构，让巨大参数成为可能，也引发moe的相关研究。

moe模型与稠密模型对比，具有以下几个优点：

- 训练速度快，仅仅激活部分专家，不必经过所有参数，降低训练成本
- 推理速度快，在相同参数下，稀疏模型天然具有巨大优势，激活参数少可以大幅降低推理延迟和带宽压力。
- 扩展成本低，计算成本不变，参数量可以无限复制同时保持模型性能
- 多任务学习能力强

缺点也明显，训练不稳定（专家负载问题，容易一个专家一直干活）；并行通信代价高（EP并行需要大量通信带宽）；模型复杂度高，研发成本相对高。

## sMoE（实现Outrageously-Large-Neural-Networks中的专家均衡）

接下来开始手撕MoE代码——sMoE

首先定义专家结构（简单MLP层）

```python
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.soft(out)
        return out
```

接着定义专家系统的核心逻辑（包含噪声与重要性机制），核心思想是为了防止某些专家过于强大，导致其它专家激活不了，从而失去稀疏性的特点，引入负载均衡机制，其包含两个维度，一是专家之间负载均衡，二是一个专家内对于不同token的激活分数不要差太多。第一个维度通过重要性解决，即各专家的门控分数的变异系数的平方乘以手动调节的参数；第二个维度是为了均衡训练中各GPU的压力，单只有第一个维度，可能出现少量token高门控分数的专家，相对的是大量token低门控分数的专家，从而也形成了平衡，因此必须引入token级别的负载均衡，通过引入噪声机制，将其变为可导的损失——衡量一个专家进入top-k的概率，由于top-k需要与其它专家在token级别进行比较，自然就能通过反向传播优化了。其公式如下：

$$
P(x, i) = \Pr\left( (x \cdot W_g)_i + \text{StandardNormal}() \cdot \text{Softplus}\big((x \cdot W_{\text{noise}})_i\big) > \text{kth\_excluding}\big(H(x), k, i\big) \right)
$$

公式一为定义的概率值，即当前专家i被tokenx选择的概率，由于其中存在一个标准正态分布变量，因此可以将其转化为如下随机变量，进而可以求导。分子为当前专家得分减去进入top-k的阈值得分，分母为噪声大小参数（可学习）。这儿主要因子为分子部分，如前面分析，直接比较token级别的分配；分母主要用于稳定梯度。公式运用了正态分布函数特性简化。

$$
P\left(x,i\right)=\Phi\left(\frac{\left(x\cdot W_g\right)_i-\text{kth}\_\text{excluding }\left(H\left(x\right),k,i\right)}{\text{Softplus}\left(\left(x\cdot W_\text{ noise }\right)_i\right)}\right)
$$

```python
class MoE(nn.Module):
    def __init__(self, input_size, output_size, num_experts, hidden_size, noisy_gating=True, k=4):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        # instantiate experts
        self.experts = nn.ModuleList([MLP(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def forward(self, x, loss_coef=1e-2):
        # 1. 计算noisy top k gating决定哪几个expert会进行计算
        gates, load = self.noisy_top_k_gating(x, self.training)
        importance = gates.sum(0)
        # 负载均衡loss
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        # 2. batch dispatcher 分配不同的数据给不同的专家，提高训练并行性
        # dispatcher类在本文不介绍，看本文图解更加清晰
        dispatcher = SparseDispatcher(self.num_experts, gates) #创建分配器
        expert_inputs = dispatcher.dispatch(x) #给每个专家分配x，整理输入
        gates = dispatcher.expert_to_gates() 
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)] # 每个专家计算对应x的输出
        y = dispatcher.combine(expert_outputs) #对于xi，计算xi在top-k专家的mixture输出

        # 这里的y是sMoE的预测， loss为 load balance，并不是最终的loss
        return y, loss

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    # 估计Prob从而计算估计的Load Loss
    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        # 原论文在进入函数前会将(B,N,...)reshape为(T = B*N,...),我们在这里是简化实现，因此无需展平，参数最后一个包含k+1专家，方便计算概率
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()   # (T * K+1，)

        # top-k时会把无关的expert的gating置为0， 这时要填补一些随机值，使得参数是可导的
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k   # 阈值k专家所在位置
        # gather的作用为将index对应output位置的值作为维度dim的值，dim=2，out[i][j][k] = input[i][j][index[i][j][k]] 
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)   # (T*K,1)
        is_in = torch.gt(noisy_values, threshold_if_in)   # 逐元素比较>，(T,E)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        # 这里计算每个专家的估计负载值,对于模型来说，并不是只有前k个才有机会选入，这个阈值是我们人为定制的，因此可以用差值的概率作为损失，让每个专家对于一个token都尽可能均衡，减少出现极大或极小gate函数
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob   # (T,E)

    # 完整计算noisy top k gating
    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        #shape(x)=(Batch,Num,Input)
        clean_logits = x @ self.w_gate    # (B,N,E)
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise   # (B,N,E)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))   # (B,N,E)
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)   # (B,N,E),逐个相乘
            logits = noisy_logits
        else:
            logits = clean_logits
        # 训练阶段在topk之前可以加入一个极小的噪声，打破可能出现的平局界面
        # 选出top-k gating值和序号
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]   # (B, N, k)
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)   # (B, N, k)

        zeros = torch.zeros_like(logits, requires_grad=True) # 创建gating 0值
        gates = zeros.scatter(1, top_k_indices, top_k_gates) # 在0值上填top-k gating值

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

def train(x, y, model, loss_fn, optim):
    y_hat, aux_loss = model(x.float())
    loss = loss_fn(y_hat, y)
    total_loss = loss + aux_loss  # 预测与label的loss + 负载均衡loss
    optim.zero_grad()
    total_loss.backward()
    optim.step()
    return model

```

## Mixtral 8*7B实现（使用switch Moe中的负载均衡损失）

> [transformers/src/transformers/models/mixtral/modular\_mixtral.py ](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modular_mixtral.py)

依旧是先定义专家模型，和上述差不多，一个FFN层，但是使用现在比较流行的基于SwiGLU激活函数的前馈神经网络，该激活函数梯度平滑，并且融合了门控机制，可以控制信息的流转。

```python
class MixtralMLP(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states   # (T,D)
```

接着定义专家组,负责将展平的数据发给对应专家进行加权计算，得到最终的结果

```python
class MixtralExperts(nn.ModuleList):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        # 添加上述定义的专家
        for _ in range(self.num_experts):
            self.append(MixtralMLP(config))

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        输入:
            hidden_states: (batch_size * sequence_length, hidden_dim)
            top_k_index: (batch_size * sequence_length, top_k)
            top_k_weights: (batch_size * sequence_length, top_k)
        输出:
            (batch_size * sequence_length, hidden_dim)
        """
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)   # (E,K,T)
        # 只遍历已经激活的专家
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            # 提取出当前专家需要处理的token(M,D)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_states.shape[-1])
            # (M,D)*(M,1)
            current_hidden_states = self[expert_idx](current_state) * top_k_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        return final_hidden_states
```

再接着定义专家模块

```python
class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.jitter_noise = config.router_jitter_noise
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = MixtralExperts(config)

    def route_tokens_to_experts(self, router_logits):
        routing_weights = torch.nn.functional.softmax(router_logits.float(), dim=-1)
        top_k_weights, top_k_index = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)
        return top_k_index, top_k_weights.to(router_logits.dtype)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # 避免专家坍缩，即冷启动
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])   # 将前两维度展平成一个维度
        router_logits = self.gate(hidden_states)
        top_k_index, top_k_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights.to(hidden_states.dtype))
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states
```

该损失值较为简单，即将专家分配的平均token数和平均门控激活值相乘，在mixtral的实现中，是在有moe层的维度进行负载均衡，上述损失值计算的是所有moe层的。这儿还涉及到padding token，尚不了解，先放一放。

Mixtral 8*7B的含义为——7B是指32层中每层一个专家的参数量，8\*7B是指每层有8个专家，这样看，实际参数并不是56B，因为7B中有很多共享的，比如嵌入、注意力等，故计算后实际参数在47B，推理参数在13B左右。
