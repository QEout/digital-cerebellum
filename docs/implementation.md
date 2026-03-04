# Digital Cerebellum — 实现指南

> 基于对现有技术栈的深入调研，本文档为架构中的每个组件提供具体的实现方案。

---

## 一、技术路线选择

```
  核心引擎: Python + PyTorch（研发阶段）
  ═══════════════════════════════════════════════════
  这是一个 ML 计算项目，不是 Web 应用。语言选择应服务于核心计算需求。

  选择 Python 的理由:
  • 所有参考实现都是 Python（EWC、RFF、小脑模型、ReservoirPy）
  • PyTorch 是在线学习/增量训练的一等公民
  • scikit-learn 的 RBFSampler 可直接用于 Random Fourier Features
  • NumPy/SciPy 做矩阵运算是基本功
  • Jupyter Notebook 方便实验、可视化学习曲线和误差变化
  • 论文复现速度最快，从想法到验证的路径最短
  • ONNX 导出成熟，验证后可无缝部署到任何运行时

  不选 TypeScript 的理由:
  • TensorFlow.js 是二等公民，API 不完整，社区小
  • 矩阵运算需要自己造轮子或绕路
  • EWC 等算法无现成实现，需要从 PyTorch 移植
  • ML 调试工具链几乎不存在
  • 选 TS 唯一的理由是"OpenClaw 兼容"，但这是集成需求，
    不应该决定核心引擎的语言

  部署阶段的语言分工:
  ──────────────────────────────────────────────────
  • 核心引擎（训练 + 实验）:        Python + PyTorch
  • 推理服务（生产部署）:            Python (FastAPI) 或 ONNX Runtime
  • OpenClaw 插件（集成层，可选）:   TypeScript（薄壳，调用推理服务）
  • 极致性能路径（未来）:            Rust + Candle（ONNX 推理）
  ──────────────────────────────────────────────────
```

---

## 二、各组件实现方案

### ① 感知层 (Perception Layer)

**目标**: 监听外部世界的事件，标准化为统一格式。

```
  技术方案:
  ──────────────────────────────────────────────────────────────
  事件源               实现方式
  ──────────────────────────────────────────────────────────────
  文件系统变更         watchdog (Python, 跨平台文件系统监听)

  消息平台             Phase 0 不需要。
                       后续可通过 HTTP API 对接 OpenClaw 等 Agent

  HTTP/Webhook         FastAPI / uvicorn（异步，高性能）

  定时/节律            自研节律引擎（见第⑨节）
                       基于 asyncio 事件循环，动态调度

  数据库变更           可选：监听 SQLite WAL
  ──────────────────────────────────────────────────────────────
```

**统一事件接口**:

```python
@dataclass
class CerebellumEvent:
    id: str                              # 唯一标识
    type: str                            # "message" | "file_change" | "schedule" | ...
    source: str                          # "wechat" | "filesystem" | "api" | ...
    payload: dict[str, Any]              # 原始数据
    timestamp: float                     # Unix seconds
    context: EventContext | None = None  # 感知层附加的上下文

@dataclass
class EventContext:
    user_id: str | None = None
    session_id: str | None = None
    related_events: list[str] | None = None
```

---

### ② 模式分离器 (Pattern Separator)

**目标**: 将事件编码为高维稀疏向量，实现模式分离（对应颗粒细胞层的维度爆炸）。

**核心算法: Random Fourier Features (随机傅里叶特征)**

生物学中，颗粒细胞层将低维输入映射到极高维的稀疏空间，本质是一个核函数机器。
在工程上，最直接的对应是 Random Fourier Features（又名 Random Kitchen Sinks），
由 Rahimi & Recht 2007 年提出，用随机投影近似核函数：

```
  数学原理:
  ──────────────────────────────────────────────────────────────
  给定输入向量 x ∈ R^d，生成高维特征 z ∈ R^D（D >> d）:

  z(x) = √(2/D) · cos(Wx + b)

  其中:
  • W ∈ R^{D×d}  随机矩阵，从 N(0, γ²I) 采样（初始化后固定）
  • b ∈ R^D      随机偏置，从 Uniform(0, 2π) 采样（初始化后固定）
  • γ            对应 RBF 核的带宽参数
  • cos 逐元素应用

  输出 z(x) 是 D 维向量，近似 RBF 核: k(x,y) ≈ z(x)ᵀz(y)
  ──────────────────────────────────────────────────────────────

  为什么选这个:
  • 极其轻量: 一次矩阵乘法 + cos，无需训练
  • 对应生物学: 颗粒细胞的随机连接 + 稀疏激活
  • 效果已验证: scikit-learn 的 RBFSampler 就是这个
  • Python 中可直接用 scikit-learn，或用 NumPy 几行实现
```

**实现步骤**:

```
  1. 事件 → 基础特征向量
     ─────────────────────────────────────────────────
     将 CerebellumEvent 编码为固定长度的数值向量:

     • 文本字段 → 本地嵌入模型 (all-MiniLM-L6-v2, 384维)
       使用 sentence-transformers:
       pip install sentence-transformers
       模型: all-MiniLM-L6-v2（英文）
       中文场景: BAAI/bge-small-zh-v1.5

     • 时间特征 → 周期编码
       hour_sin = sin(2π × hour/24)
       hour_cos = cos(2π × hour/24)
       day_sin  = sin(2π × weekday/7)
       day_cos  = cos(2π × weekday/7)

     • 类型/来源 → one-hot 编码

     • 用户状态 → 从语义记忆中检索的特征向量

     拼接后得到基础向量 x ∈ R^d（约 400~500 维）

  2. 基础向量 → 高维稀疏向量（Random Fourier Features）
     ─────────────────────────────────────────────────
     x ∈ R^d  →  z(x) ∈ R^D

     推荐 D = 4096~8192（d 的 10~20 倍）
     计算量: 一次矩阵乘法，< 1ms

  3. 稀疏化（对应 Golgi 细胞的抑制性调节）
     ─────────────────────────────────────────────────
     对 z(x) 应用 top-k 稀疏化:
     只保留绝对值最大的 k 个分量，其余置零
     推荐 k = D × 0.1（10% 稀疏度）

     这一步确保:
     • 相似但不同的输入有不同的激活模式
     • 下游预测引擎需要学习的参数更少
     • 存储和计算更高效
```

**技术依赖**:

```
  pip install sentence-transformers    # 本地嵌入模型
  pip install scikit-learn             # RBFSampler (RFF)
  # 或用 NumPy 手写 RFF（十几行代码）
```

---

### ③ 预测引擎 (Prediction Engine)

**目标**: 给定高维稀疏向量，预测应执行的动作和预期结果。对应浦肯野细胞群体的线性计算。

**关键设计决策 1: 线性模型，不是深度网络**

生物学中浦肯野细胞使用线性算法。配合颗粒细胞层的维度爆炸，
线性模型在高维空间中已经足够强大（核方法的本质）。

**关键设计决策 2: 群体编码（Population Coding）**

v1 使用单头预测 + sigmoid 置信度。但 2025 年 J.Neuroscience 研究表明，
浦肯野细胞群体的空间相关性（而非单细胞输出）才是小脑编码行为信息的方式。
因此 v2 使用 K 个并行预测头，置信度从群体一致性中涌现。

```
  模型架构: K 头线性预测器
  ──────────────────────────────────────────────────────────────

  z ∈ R^D  →  K 个独立线性头，各自预测:

  head_k:  z → W_k · z + b_k  →  output_k     (k = 1..K)

  K = 4 (Phase 0) → 8 (Phase 1+)

  每个头有独立的参数和独立的初始化，
  但共享同一个模式分离器输出 z。


  置信度 = 群体一致性（涌现，非 sigmoid）:
  ───────────────────────────
  predictions = [head_1(z), head_2(z), ..., head_K(z)]
  mean_pred   = mean(predictions)
  variance    = mean(‖pred_k - mean_pred‖² for k in 1..K)
  confidence  = exp(-variance / τ)    # τ 为温度参数

  一致 → 高置信度 / 分歧 → 低置信度
  不需要额外模型，不确定性量化是免费的。


  方案 A: 纯线性头（Phase 0，最轻量）
  ───────────────────────────
  每个头: nn.Linear(D, output_dim)
  • 参数量: K × D × output_dim
    K=4, D=4096, output=256 → ~420 万参数
  • 推理: K 次矩阵向量乘法，< 0.5ms
  • 可并行计算（torch 批处理）


  方案 B: 浅层头（Phase 1+，更强表达力）
  ───────────────────────────
  每个头: Linear(D, 512) → ReLU → Linear(512, output_dim)
  • 参数量: K × (D×512 + 512×output_dim)
  • 推理: < 2ms
  • 部署: 导出为 ONNX


  推荐: Phase 0 用方案 A (K=4)，Phase 1 切换到方案 B (K=8)
  ──────────────────────────────────────────────────────────────
```

**输出设计**:

预测引擎的输出不是"自然语言"，而是结构化的动作向量。
最终输出取 K 个头的均值预测 + 群体涌现置信度：

```
  输出结构:
  ──────────────────────────────────────────────────────────────
  {
    action_embedding: Float32[128]    # K 个头的均值动作嵌入
    outcome_embedding: Float32[128]   # K 个头的均值结果嵌入
    confidence: float                 # 群体一致性涌现（非 sigmoid）
    head_predictions: list[...]       # 各头的原始预测（用于分析）
    domain_logits: Float32[N]         # 各领域的激活度（均值）
  }

  action_embedding 通过最近邻搜索解码为具体动作:
  → 在"动作词典"（向量 → 具体 tool_call 的映射表）中查找最近的条目

  额外能力（单头模型无法做到的）:
  • 可区分"确定的部分"和"不确定的部分"
    例: 所有头同意 tool_name=send_email，但对 recipient 分歧
    → 拦截，要求用户确认收件人，而非重新推理整个请求
  ──────────────────────────────────────────────────────────────
```

**技术依赖**:

```
  pip install torch          # 训练 + 推理
  pip install onnxruntime    # 生产推理（可选，更快）
```

---

### ④ 决策路由器 / 深部核团 (Decision Router / DCN)

**目标**: 整合多头预测输出，做路由决策。对应深部核团（DCN）的主动计算功能。

v2 升级：DCN 不再是简单的 if-else 阈值，而是有自身可塑性的计算模块。

```
  生物学依据 (2025 Frontiers):
  ──────────────────────────────────────────────────────────────
  深部核团的功能远超"阈值开关":
  • 速率编码转换: 将浦肯野细胞的不规则放电转换为平滑输出
  • 多微区同步: 整合不同微区的输出，协调时序
  • 延迟补偿: 根据执行层延迟，提前发出指令
  • 独立可塑性: 通过 reward_error 更新自身权重
  ──────────────────────────────────────────────────────────────
```

```
  架构: 可学习的路由器
  ──────────────────────────────────────────────────────────────

  Phase 0 — 简单版:
  ───────────────────────────

  输入: K 个预测头的输出 + 群体一致性 confidence
  输出: routing_decision ∈ {"fast", "shadow", "slow"}

  def route(prediction: PredictionOutput) -> str:
      if prediction.confidence < self.threshold_low:
          return "slow"
      if prediction.confidence >= self.threshold_high:
          return "fast"
      return "shadow"

  阈值每个微区独立维护，通过 reward_error 自适应:
  • 初始: threshold_high = 0.95（保守）
  • RPE > 0 持续积累: 降低阈值（更信任小脑）
  • RPE < 0: 立即提高阈值（降级保护）


  Phase 1 — 主动计算版:
  ───────────────────────────

  class ActiveDCN(nn.Module):
      def __init__(self, K, pred_dim):
          self.smoother = nn.Linear(K * pred_dim, pred_dim)
          self.router   = nn.Linear(pred_dim + K, 3)  # fast/shadow/slow
          self.delay_compensator = nn.Linear(pred_dim, pred_dim)

      def forward(self, head_predictions, execution_latency):
          # 1. 平滑: 将 K 个头的不规则输出转换为一致的输出
          stacked = torch.cat(head_predictions, dim=-1)
          smoothed = self.smoother(stacked)

          # 2. 延迟补偿: 根据执行延迟调整输出
          compensated = smoothed + self.delay_compensator(smoothed) * latency

          # 3. 路由决策
          variances = torch.var(torch.stack(head_predictions), dim=0)
          route_input = torch.cat([smoothed, variances.mean(-1)])
          route_logits = self.router(route_input)

          return compensated, route_logits

  自身权重通过 reward_error 更新（多位点学习的位点 2）
  ──────────────────────────────────────────────────────────────

  参考研究:
  • STEER (ArXiv 2511.06190): 步级置信度路由，
    +20% 准确率，-48% 计算量
  • Confidence Tokens (ICML 2025): 用专门的置信度 token
    提取模型置信度，比 logit 概率更可靠
  ──────────────────────────────────────────────────────────────
```

---

### ⑤ 执行层 (Effector)

**目标**: 执行动作并返回结果。

```
  两种集成模式:
  ──────────────────────────────────────────────────────────────

  模式 A: 独立 Python 服务（Phase 0~1 推荐）
  ───────────────────────────
  • 独立 Python 进程，通过 HTTP API 或 stdin/stdout 交互
  • Phase 0: CLI 工具，输入消息 → 输出预测结果
  • Phase 1: FastAPI 服务，暴露 REST API
  • 通过 OpenAI 兼容 API 连接 LLM
  • 可对接任何 Agent 框架


  模式 B: 作为 OpenClaw 插件（后续集成方案）
  ───────────────────────────
  • Python 核心引擎作为独立进程运行
  • 薄 TypeScript 插件通过 HTTP 调用 Python 服务
  • 复用 OpenClaw 已有的 50+ 工具集成
  • 此时 TypeScript 只是胶水层，不到 100 行
  ──────────────────────────────────────────────────────────────
```

---

### ⑥ 皮层接口 (Cortex Interface)

**目标**: 与 LLM 的双向通信 + 反馈解耦 + 任务巩固。

```
  A. LLM 调用（慢路径）
  ──────────────────────────────────────────────────────────────

  当决策路由器选择慢路径时:

  1. 组装上下文（反馈解耦的核心）:
     ┌────────────────────────────────────────────────┐
     │  系统提示（常规）                               │
     │  + 小脑预测结果（即使低置信度也附上）            │
     │    "小脑预测: 用户可能想要发送文件给张三,        │
     │     置信度 0.43，仅供参考"                      │
     │  + 相关记忆（从语义记忆检索）                    │
     │  + 用户画像摘要                                 │
     └────────────────────────────────────────────────┘

     反馈解耦效果:
     • LLM 不需要从零推理，小脑的预测提供了"直觉"
     • 即使预测不完全正确，也缩小了 LLM 的搜索空间
     • 实测可减少 30~50% 的输出 token

  2. 调用 LLM:
     使用 OpenAI 兼容 API（支持 Claude/GPT/Ollama/本地模型）

     pip install openai  # 兼容所有 OpenAI API 格式的提供商

  3. 解析 LLM 输出:
     提取结构化的 tool_call + 文本回复


  B. 任务巩固流水线
  ──────────────────────────────────────────────────────────────

  每次 LLM 处理完一个任务后:

  1. 记录交互对:
     {
       input:  高维稀疏向量 (② 的输出)
       output: LLM 的 action + outcome
       domain: 领域标识
     }
     → 存入 SQLite 的 consolidation_buffer 表

  2. 后台检查（每天 / 每累积 N 条）:
     对 consolidation_buffer 按 domain 聚类
     如果某个 domain 的样本数 >= 阈值（如 10 条）
     且样本之间的输入-输出映射具有一致性
     → 触发微调

  3. 在线蒸馏:
     用 consolidation_buffer 中的数据
     对预测引擎做增量训练（几秒到几分钟）

     不是传统意义的 LLM 蒸馏（那需要白盒访问教师模型）
     而是行为克隆: 学习 f(input) → LLM_output
     本质上是监督学习，用 LLM 的输出作为标签

  4. 验证 + 上线:
     在 shadow 模式下运行新模型
     如果准确率达标 → 该领域"毕业"到快路径
     如果不达标 → 继续积累样本
  ──────────────────────────────────────────────────────────────
```

---

### ⑦ 误差比较器 (Error Comparator)

**目标**: 比较预测与实际结果，生成三种独立误差信号，分别驱动不同组件的学习。

```
  三种误差通道（v2 架构核心升级）:
  ──────────────────────────────────────────────────────────────

  依据: 2025 Nature Communications — 小脑攀爬纤维携带奖励预测误差
       2025 J.Neuroscience — 深部核团编码时序预测误差

  ┌────────────────┬─────────────────────────┬───────────────────┐
  │  误差类型       │  计算方式                │  驱动的学习目标   │
  ├────────────────┼─────────────────────────┼───────────────────┤
  │  sensory_error │  预测结果 vs 实际结果    │  预测引擎权重     │
  │  temporal_error│  预测时间 vs 实际时间    │  节律系统参数     │
  │  reward_error  │  预期效果 vs 用户反馈    │  路由器偏好       │
  └────────────────┴─────────────────────────┴───────────────────┘
```

```
  通道 1: 感觉预测误差 (Sensory Prediction Error, SPE)
  ──────────────────────────────────────────────────────────────
  predicted_action  vs  actual_action_taken    → 动作误差
  predicted_outcome vs  actual_outcome         → 结果误差

  计算方式:
  action_error  = cosine_distance(pred_action_emb, actual_action_emb)
  outcome_error = cosine_distance(pred_outcome_emb, actual_outcome_emb)

  → 更新: 预测引擎的 K 个头权重


  通道 2: 时序预测误差 (Temporal Prediction Error, TPE)
  ──────────────────────────────────────────────────────────────
  predicted_time  vs  actual_event_time       → 时间误差

  计算方式:
  temporal_error = |predicted_timestamp - actual_timestamp|

  高 TPE 意味着节律系统的预测不准（太早/太晚唤醒）

  → 更新: 节律系统的唤醒队列参数、时间模式模型


  通道 3: 奖励预测误差 (Reward Prediction Error, RPE)
  ──────────────────────────────────────────────────────────────
  预期效果 vs 用户反馈

  信号来源:
  • 用户显式反馈: "不对" / "好的" / 重试 → RPE = ±1
  • 隐式反馈: 用户是否采纳了建议 → RPE ∈ [-1, 1]
  • 结果评估: 操作是否成功完成 → RPE ∈ [0, 1]

  → 更新: 路由器的阈值和偏好
    RPE > 0: 该微区的信任度上升，阈值可以降低
    RPE < 0: 信任度下降，阈值提高（更保守）
```

```
  误差信号的来源场景（三种误差都可能产生）:
  ──────────────────────────────────────────────────────────────

  1. 即时误差（快路径执行后）
     ────────────────────────
     → SPE: 预测 vs 实际结果
     → RPE: 用户是否满意

  2. 延迟误差（用户后续反馈）
     ────────────────────────
     如果用户在 T 分钟内说"不对"/"重新来"
     → RPE = -1
     → 通过资格迹 (Eligibility Trace) 回溯到之前的预测

     资格迹实现:
     • 每次预测时记录 (event_id, model_parameters_snapshot, timestamp)
     • 延迟误差到达时，按时间衰减系数 λ^(t2-t1) 加权
     • 更新时只修改与该预测相关的参数子集

  3. 影子执行误差（中间置信度时）
     ────────────────────────
     小脑预测 vs LLM 输出 → SPE
     这是最宝贵的学习信号:
     • 不影响用户体验（以 LLM 为准）
     • 但小脑获得了高质量的训练数据

  4. 节律误差（预测唤醒时）
     ────────────────────────
     预测"用户 9:00 查邮件" vs 实际 9:12 查邮件
     → TPE = 12 分钟
     → 节律系统调整该模式的预测参数


  Phase 0: 只实现 SPE（影子执行 + 即时误差）
  Phase 1: 加入 RPE（用户反馈） + TPE（节律校正）
  ──────────────────────────────────────────────────────────────
```

---

### ⑧ 在线学习模块 (Online Learning)

**目标**: 根据三种误差信号实时更新四个可塑性位点的权重，防止灾难性遗忘。

```
  v2 核心升级: 多位点学习
  ──────────────────────────────────────────────────────────────

  依据: 2024 Nature Communications / eLife — 小脑至少 4 个可塑性位点

  ┌──────────────────┬──────────────────┬────────────────────┐
  │  可塑性位点       │  驱动的误差信号   │  学习目标          │
  ├──────────────────┼──────────────────┼────────────────────┤
  │  预测引擎权重     │  sensory_error   │  预测更准确        │
  │  路由器权重       │  reward_error    │  路由更精确        │
  │  频率滤波器参数   │  sensory_error   │  信号带宽更适配    │
  │  模式分离器增益   │  sensory_error   │  稀疏度更合适      │
  └──────────────────┴──────────────────┴────────────────────┘

  Phase 0 实现: 位点 1（预测引擎）
  Phase 1 实现: 位点 1 + 2（+ 路由器）
  Phase 2 实现: 全部 4 个位点
  ──────────────────────────────────────────────────────────────


  核心算法: SGD + EWC 正则化
  ──────────────────────────────────────────────────────────────

  标准 SGD 更新:
    θ_new = θ_old - α · ∇L(θ, x, y)

  EWC (Elastic Weight Consolidation) 修正:
    θ_new = θ_old - α · [∇L(θ, x, y) + λ · F · (θ - θ*)]

  其中:
  • F = Fisher 信息矩阵（对角近似），衡量每个参数对旧任务的重要性
  • θ* = 旧任务训练完毕时的参数快照
  • λ = 正则化强度（控制"记住旧知识" vs "学习新知识"的平衡）

  直觉:
  • 对旧任务不重要的参数 → F 小 → 自由更新
  • 对旧任务重要的参数 → F 大 → 变化被惩罚
  • 类似生物学中"已建立的突触连接更难改变"

  每个可塑性位点独立维护自己的 F 和 θ*


  各位点的更新规则:
  ──────────────────────────────────────────────────────────────

  位点 1 — 预测引擎:
  • 误差: sensory_error (SPE)
  • 损失: cosine_distance(predicted, actual) + EWC 正则
  • K 个头独立更新（各自学各自的误差）
  • 学习率: α₁ = 0.01 (新) → 0.001 (成熟)

  位点 2 — 路由器 (Phase 1):
  • 误差: reward_error (RPE)
  • 损失: -RPE × log(route_prob) (策略梯度)
  • 正反馈强化当前路由策略，负反馈回退到保守策略
  • 学习率: α₂ = 0.001（路由器变化应缓慢）

  位点 3 — 频率滤波器参数 (Phase 2):
  • 误差: sensory_error (SPE)
  • 学习高通/低通截止频率
  • 学习率: α₃ = 0.0001（极缓慢调整）

  位点 4 — 模式分离器增益 (Phase 2):
  • 误差: sensory_error (SPE)
  • 学习 Golgi 门控参数: W_golgi, b_golgi
  • 学习率: α₄ = 0.0001


  实现框架:
  ──────────────────────────────────────────────────────────────

  使用 PyTorch 实现:

  pip install torch

  关键实现点:

  1. Fisher 信息矩阵的增量计算:
     • 不是一次性计算整个 F（那需要遍历所有旧数据）
     • 而是在每次成功预测后，增量更新 F 的对角元素
     • F_ii += (∂L/∂θ_i)²（梯度平方的移动平均）

  2. 参数快照管理:
     • 每个位点独立维护 θ* 和 F
     • 定期（如每天）更新快照: θ* ← θ_current, F 重置

  3. 学习率调度:
     • 新微区: α 使用初始值（快速学习）
     • 成熟微区: α 降低（精细调整）
     • 误差突然上升: α 临时提高（环境变化，需要快速适应）

  可直接参考的现成实现:
  • tanmay1024/EWC-Implementation (PyTorch，可直接复用)
  • EsraErgun/Continual-Learning-EWC-SI (PyTorch，含 SI 对比)
  • khangbkk23/OnlineContinualLearning (2025，含 buffer 策略)
  ──────────────────────────────────────────────────────────────
```

---

### ⑨ 节律系统 (Rhythm System)

**目标**: 不是 cron job，是预测性的动态调度。

```
  实现架构:
  ──────────────────────────────────────────────────────────────

  三层触发机制:

  Layer 1: 事件驱动（立即响应）
  ──────────────────────────
  • watchdog 文件变更监听
  • HTTP/WebSocket 消息到达 (FastAPI)
  • API 回调
  → 直接触发对应微区的预测流水线
  → 延迟: < 50ms

  Layer 2: 预测性唤醒（学习用户节律）
  ──────────────────────────
  维护一个"下一次可能需要关注的时间点"优先队列:

  class RhythmEngine:
      def __init__(self):
          self.queue: list[WakeupEvent] = []  # 堆队列

      def learn_pattern(self, event: CerebellumEvent):
          """学习用户的时间模式
          例: 用户通常 9:00 查邮件 → 8:55 预唤醒"""
          ...

      async def next_wakeup(self) -> float:
          """计算下一次唤醒时间"""
          return self.queue[0].timestamp

  底层用 asyncio.sleep() 实现，但等待时间是动态预测的
  不是"每 30 分钟"，而是"预测下一个有意义的时间点"

  Layer 3: 兜底心跳（安全网）
  ──────────────────────────
  最大静默时间（如 2 小时）
  如果 Layer 1 和 Layer 2 都没有触发
  → 执行一次全局状态检查
  → 确保系统没有遗漏重要事件
  ──────────────────────────────────────────────────────────────
```

---

### ⑩ 频率滤波层 + Golgi 门控 + 状态估计器（Phase 2 新增组件）

这三个组件是 v2 架构修正中的"优化修正"，Phase 2 实现。
Phase 0-1 的接口设计需要预留插入位置。

```
  A. 频率滤波层（对应分子层中间神经元）
  ──────────────────────────────────────────────────────────────

  依据: 2025 Nature — 篮状细胞滤低频，星状细胞滤高频

  插入位置: 模式分离器 → [频率滤波] → 预测引擎

  实现:
  class FrequencyFilter(nn.Module):
      def __init__(self, dim):
          self.lowpass  = nn.Linear(dim, dim)  # 篮状细胞
          self.highpass = nn.Linear(dim, dim)  # 星状细胞
          self.cutoff_low  = nn.Parameter(torch.tensor(0.3))  # 可学习
          self.cutoff_high = nn.Parameter(torch.tensor(0.7))

      def forward(self, z, z_history):
          # 简化版: 用最近 N 帧的 z 做时域滤波
          z_smooth = exponential_moving_average(z_history, self.cutoff_low)
          z_low  = self.lowpass(z_smooth)     # 持续趋势
          z_high = self.highpass(z - z_smooth) # 瞬态事件
          return z_low, z_high

  效果:
  • z_low: 用户的持续行为模式（例: 上午写代码）
  • z_high: 突发事件（例: 突然收到紧急消息）
  • 预测引擎对两路信号做不同的预测，提高区分度


  B. Golgi 反馈门控（对应颗粒细胞层 Golgi 细胞）
  ──────────────────────────────────────────────────────────────

  依据: Golgi 细胞的抑制性反馈回路，调节颗粒细胞的稀疏度

  插入位置: RFF 输出后、top-k 稀疏化前

  实现:
  # 在 RFF 输出 z 上加可学习的增益门
  z_gated = z * sigmoid(W_golgi @ z + b_golgi)

  # W_golgi 和 b_golgi 通过 sensory_error 更新
  # （多位点学习的位点 4）

  效果:
  • 不同输入模式激活不同的门控通路
  • 系统自动学习哪些维度对当前领域更重要
  • 比固定 top-k 更灵活的稀疏化


  C. 状态估计器（平行于预测引擎）
  ──────────────────────────────────────────────────────────────

  依据: 2026 Nature Communications — 小脑某些区域编码
       ground-truth 自身运动，不只是预测

  功能: 不做预测，做"此刻到底发生了什么"的准确感知

  实现:
  class StateEstimator(nn.Module):
      def __init__(self, D, state_dim):
          self.estimator = nn.Linear(D, state_dim)

      def forward(self, z):
          return self.estimator(z)  # 当前状态估计

  与预测引擎并行运行:
  z → [预测引擎: 预测下一刻]    → predicted_state
  z → [状态估计器: 感知此刻]    → current_state

  用途:
  • 误差比较器用 current_state 而非 raw_input 做基线
  • 游戏场景: "角色当前位置/血量"的精确感知
  • LLM 场景: "当前对话状态"的结构化表示

  Phase 0-1: 可以用原始输入代替，预留接口
  Phase 2: 实现独立的状态估计模块
  ──────────────────────────────────────────────────────────────
```

---

### ⑫ 涌现认知层 (Emergent Cognition — Phase 3)

三个从小脑信号中涌现的高阶认知属性。

```
  A. 躯体标记 / 直觉 (Somatic Marker)
  ──────────────────────────────────────────────────────────────

  依据: Damasio 的躯体标记假说 — 过往经验留下"体感标记"，
       在意识推理之前就偏置了决策。
       2025 J.Neuroscience — 小脑通过 cerebello-thalamo-cortical
       通路发送预测误差信号。

  核心洞察:
  K 个预测头的分歧不只是"不确定"，其分歧的 *模式* 本身
  携带了信息。某些分歧模式在过去总是对应坏结果。

  实现:
  class SomaticMarker:
      # 1. 从 K 头输出中提取"分歧指纹"
      #    (pairwise cosine similarity 向量)
      fingerprint = extract_fingerprint(head_predictions)
      # → (K*(K-1)/2 * 2) 维向量, K=4 时为 12 维

      # 2. 与存储的效价标记做相似度匹配
      gut_feeling = feel(head_predictions, domain)
      # → GutFeeling(valence, intensity, trigger_pattern)

      # 3. 强烈负面直觉可覆盖正常路由
      if gut_feeling.should_override:  # intensity>0.6 且 valence<-0.3
          force_slow_path()

  协调机制:
  • SelfModel 的能力评估可抑制 override（专家域不容易被直觉吓退）
  • warmup 阶段（前 30 步）不记录 markers，避免噪声污染
  • 标记强度随时间衰减（sleep 周期中调用 decay）

  文件: digital_cerebellum/emergence/somatic_marker.py


  B. 好奇心驱动 (Curiosity Drive)
  ──────────────────────────────────────────────────────────────

  依据: 多巴胺系统对新奇和预测误差的响应
       Schmidhuber 1991 — 好奇心即学习进步
       CDE 2025 — 困惑度 + 值估计方差作为探索奖励
       LPM 2025 — 奖励模型改进而非预测误差本身

  核心洞察:
  大预测误差 ≠ 值得探索。关键信号是"误差在下降"（学习进步）。
  高误差 + 高学习进步 → "这很有趣，继续探索"
  高误差 + 零进步   → "这是噪声，别浪费资源"（noisy TV 问题）
  低误差 + 稳定     → "已经掌握了，去探索新的"

  实现:
  class CuriosityDrive:
      # 每域跟踪误差轨迹
      tracker.record(error)

      # 学习进步 = mean(旧窗口) - mean(新窗口)
      lp = tracker.learning_progress  # > 0 改善, < 0 退化

      # 新奇度 = 当前输入与近期输入的平均余弦距离
      novelty = compute_novelty(feature_vec)

      # 内在奖励 = 学习进步 × 新奇度加成
      intrinsic_reward = lp * (0.5 + 0.5 * novelty)

      # 分类: "explore" | "exploit" | "abandon"

  输出:
  • CuriositySignal(novelty, learning_progress, intrinsic_reward, recommendation)
  • get_exploration_ranking() → 按学习潜力排序的域列表

  文件: digital_cerebellum/emergence/curiosity_drive.py


  C. 自我模型 / 元认知 (Self-Model)
  ──────────────────────────────────────────────────────────────

  依据: 元认知判断反映了误差历史的 recency-weighted 平均
       (Metacognitive Judgments 2023)
       EGPO 2026 — 熵校准框架
       HTC 2026 — 轨迹级置信度校准

  核心洞察:
  小脑的多微区各自追踪不同功能域的误差/置信度，
  汇总后形成隐式自我模型："我擅长支付评估但不擅长代码安全"

  实现:
  class SelfModel:
      # 每域追踪: 准确率、置信度校准误差 (ECE)、快路径比例
      record(domain, correct, confidence, route)

      # 生成自我报告
      report = introspect()
      # → SelfReport(competencies, strengths, weaknesses, recommendation)
      # 能力级别: novice → learning → competent → expert

      # 建议自适应路由阈值
      thresholds = suggest_thresholds(domain)
      # expert → threshold_high=0.75  (更信任快路径)
      # novice → threshold_high=0.98  (保守，多问大脑)

  协调 — 渐进式混合:
  # 前 50 步: 使用基线配置阈值
  # 之后: alpha = min((step-50)/300, 0.4)
  # threshold = (1-alpha)*base + alpha*suggested
  # 防止自我模型在数据不足时做出过激调整

  文件: digital_cerebellum/emergence/self_model.py
  ──────────────────────────────────────────────────────────────
```

---

### ⑪ 流体记忆系统 (Fluid Memory System)

**目标**: 实现有衰减、再巩固、抽象化、睡眠周期的生物级记忆系统。

```
  核心数据结构:
  ──────────────────────────────────────────────────────────────

  每条记忆的数据模型:
  ────────────────
```

```python
  @dataclass
  class MemorySlot:
      id: str
      content: str                   # 记忆内容（文本）
      embedding: np.ndarray          # 语义向量 (384维)
      strength: float                # 记忆强度 (0.0~1.0)
      layer: str                     # "sensory" | "short_term" | "long_term"
      created_at: float              # 创建时间 (Unix)
      last_accessed: float           # 最近访问时间
      access_count: int              # 被检索次数
      source_ids: list[str]          # 由哪些原始事件/记忆抽象而来
      metadata: dict                 # 领域标签、误差关联等
```

```
  衰减函数:
  ────────────────
  短期记忆:  strength *= exp(-λ₁ · Δt)        λ₁ = 0.1   (快速衰减)
  长期记忆:  strength *= exp(-λ₂ · Δt^0.8)    λ₂ = 0.01  (亚线性慢衰减)

  参考: FadeMem (阿里+北大, 2026.01)
  不同层级使用不同 β 指数:
    短期: β=1.2 (超线性衰减，快速遗忘)
    长期: β=0.8 (亚线性衰减，经常访问的几乎不衰减)


  再巩固 (Reconsolidation):
  ────────────────
  每次 retrieve() 时:
  1. strength 重新激活:  strength = max(strength, 0.8)
  2. access_count += 1
  3. embedding 向查询方向微调:
     embedding = (1-α) · embedding + α · query_embedding
     短期: α = 0.05,  长期: α = 0.02

  生物学依据:
  记忆在被回忆时进入不稳定状态，被当前上下文修改后重新存储。
  这意味着"回忆"本身就在塑造记忆——不是 bug，是泛化能力的来源。


  睡眠周期 (Sleep Cycle):
  ────────────────
  离线批处理，低负载时段或每日定时执行:

  步骤 1: 衰减扫描
    删除 strength < 0.05 的记忆

  步骤 2: 巩固转移
    短期记忆中 access_count >= 3 或 strength > 0.7 的
    → 转移到长期记忆

  步骤 3: 模式抽象
    长期记忆中语义相似的记忆 (cosine > 0.85) 聚类
    >= 3 条的簇 → 用 LLM 或规则提取规律，生成新的抽象记忆
    原始记忆的 strength × 0.3 (加速衰减)

  步骤 4: 蒸馏检查
    成熟模式 (strength > 0.9, access_count > 10)
    → 加入任务巩固队列，尝试编译进预测引擎

  步骤 5: 冲突消解
    内容矛盾的记忆 → 保留 strength 更高的
    或合并为"以前是 A，现在是 B"的时序记忆

  参考实现:
  • Sleeping LLM (vbario/sleeping-llm) — 醒/睡周期 + LoRA 巩固
  • FadeMem (ArXiv 2601.18642) — 指数衰减 + 双层层级
  ──────────────────────────────────────────────────────────────
```

```
  存储技术选型:
  ──────────────────────────────────────────────────────────────

  SQLite（结构化存储）
  ────────────────
  Python 标准库自带 sqlite3

  Tables:
  • memory_slots — 所有记忆条目 (含 strength, layer, timestamps)
  • consolidation_buffer — 任务巩固队列
  • sleep_log — 睡眠周期执行日志
  • action_dictionary — 动作词典


  FAISS（向量检索）
  ────────────────
  pip install faiss-cpu

  用途:
  • 记忆的语义检索（retrieve 时用）
  • 模式抽象时的相似度聚类
  • 动作词典的最近邻搜索

  索引随记忆的增删动态更新


  模型权重（程序性记忆）
  ────────────────
  • 预测引擎: PyTorch .pt 文件
  • Fisher 信息矩阵: PyTorch .pt 文件
  • 每个微区独立存储
  • 部署时可导出为 ONNX


  目录结构:
  ──────────────────────────────────────────────────────────────
  ~/.digital-cerebellum/
  ├── cerebellum.db              # SQLite 主数据库
  │                              #   memory_slots, consolidation_buffer,
  │                              #   sleep_log, action_dictionary
  ├── vectors/
  │   ├── short_term.index       # FAISS 索引 (短期记忆)
  │   └── long_term.index        # FAISS 索引 (长期记忆)
  ├── models/
  │   ├── chat/                  # 对话微区
  │   │   ├── weights.pt         # 预测引擎权重
  │   │   ├── fisher.pt          # Fisher 信息矩阵
  │   │   └── meta.json          # 微区元数据
  │   ├── files/                 # 文件微区
  │   └── schedule/              # 日程微区
  ├── embeddings/
  │   └── all-MiniLM-L6-v2/     # 本地嵌入模型
  ├── sleep/
  │   └── abstractions.jsonl     # 睡眠周期产生的抽象模式日志
  └── config.yaml
  ──────────────────────────────────────────────────────────────
```

---

## 三、集成架构

```
  Phase 0: tool_call 预评估 (CLI / SDK)
  ══════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────┐
  │  LLM Agent (OpenClaw / 自建)                             │
  │      │                                                   │
  │      │ tool_call("send_email", {to: "张三", ...})        │
  │      │                                                   │
  │      ▼                                                   │
  │  ┌─ Digital Cerebellum (Python SDK) ─────────────────┐  │
  │  │                                                    │  │
  │  │  感知 → 模式分离 → K 头预测 → DCN 路由             │  │
  │  │           (RFF)    (K=4)      (群体置信度)          │  │
  │  │                                                    │  │
  │  │  高置信度 ──→ { safe: true, risk: "none" }         │  │
  │  │  低置信度 ──→ { safe: false, risk: "wrong_param" } │  │
  │  │                                                    │  │
  │  └────────────────────────────────────────────────────┘  │
  │      │                                                   │
  │      ▼                                                   │
  │  safe=true  → 执行 tool_call                             │
  │  safe=false → 要求 LLM 重新考虑                          │
  │      │                                                   │
  │      ▼                                                   │
  │  执行结果 → 三通道误差信号 → 在线学习                     │
  └─────────────────────────────────────────────────────────┘


  Phase 1: FastAPI 服务 + 多微区
  ══════════════════════════════════════════════════════════

  ┌───────────────────┐       ┌──────────────────────────┐
  │  任意 Agent 框架   │       │  Digital Cerebellum      │
  │  (OpenClaw /      │       │  (Python FastAPI)         │
  │   LangChain /     │──────▶│                          │
  │   自建)           │ HTTP  │  POST /evaluate          │
  │                   │       │  { tool_name, params }   │
  │  tool_call ──┐    │◀──────│                          │
  │              │    │       │  → { safe, risk, conf }  │
  │              ▼    │       │                          │
  │  小脑预评估:      │       │  POST /feedback          │
  │  安全→执行        │       │  { result, user_ok }     │
  │  风险→重新考虑    │       │                          │
  └───────────────────┘       └──────────────────────────┘


  Phase 2+: 游戏 AI 集成
  ══════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────┐
  │  LLM (运营层)                     延迟: ~1s          │
  │  ─────────────                                      │
  │  目标优先级 / 出装 / 打不打团                         │
  │         │ 战略指令（每 5-10s）                       │
  │         ▼                                            │
  │  Digital Cerebellum (微操层)      延迟: <10ms        │
  │  ─────────────                                      │
  │  补刀 / 走位 / combo 时序                             │
  │         │ 键鼠输出（20Hz+）                          │
  │         ▼                                            │
  │  游戏环境 → 视觉感知 → 模式分离 → 预测 → 执行        │
  └──────────────────────────────────────────────────────┘

  核心流水线与 Phase 0 完全相同，只是换了 I/O 适配器。
  验证了 UCT（通用小脑变换）的设计。
```

---

## 四、核心依赖清单

```
  依赖                              版本         用途
  ──────────────────────────────────────────────────────────────
  核心依赖:
  torch                             >=2.2       模型训练 + 推理
  numpy                             >=1.26      数值计算
  scikit-learn                      >=1.4       RBFSampler (RFF)
  sentence-transformers             >=3.0       本地嵌入模型
  faiss-cpu                         >=1.8       向量检索
  openai                            >=1.30      LLM API 调用

  工具依赖:
  fastapi + uvicorn                 >=0.110     API 服务（Phase 1+）
  watchdog                          >=4.0       文件系统监听
  pyyaml                            >=6.0       配置解析

  开发/实验依赖:
  jupyter                           >=1.0       实验笔记本
  matplotlib                        >=3.8       可视化学习曲线
  pytest                            >=8.0       测试框架
  ──────────────────────────────────────────────────────────────

  总安装大小估算:
  • torch (CPU):            ~800MB
  • sentence-transformers:  ~200MB (含模型)
  • faiss-cpu:              ~30MB
  • 其他:                   ~50MB
  • 合计:                   ~1.1GB（开发环境）
  • 生产部署（ONNX only）:  ~200MB

  运行时内存估算:
  • 嵌入模型常驻:           ~100MB
  • 预测引擎 (所有微区):    ~50MB
  • FAISS 向量索引:         ~50MB
  • 事件缓冲区:             ~10MB
  • 合计:                   ~210MB
  ──────────────────────────────────────────────────────────────
```

---

## 五、Phase 0 最小验证方案

Phase 0 的目标是用最少的代码验证核心假设：
**"基于群体编码的轻量预测器，能否从 LLM 的 tool_call 中学习，并预评估操作合理性？"**

```
  首要微区: LLM tool_call 预评估
  ──────────────────────────────────────────────────────────────

  场景:
  LLM Agent 准备调用工具 → 数字小脑评估是否安全 → 放行或拦截

  输入: { tool_name, params_embedding, context_embedding }
  输出: { safe_prob, risk_type, confidence (群体涌现) }

  需要验证:
  1. 模式分离器能否有效区分不同 tool_call 模式？
  2. K=4 多头预测是否产生有意义的置信度分布？
  3. 预测引擎能否从执行结果中学习？
  4. 经过 N 次学习后，预评估准确率能到多少？
  5. EWC 是否有效防止了灾难性遗忘？
  6. 推理延迟是否 < 10ms？


  代码量估算:
  ──────────────────────────────────
  pattern_separator.py     ~120 行   (RFF + Golgi 门控接口预留)
  prediction_engine.py     ~200 行   (K=4 多头线性模型)
  error_comparator.py      ~120 行   (三种误差接口，Phase 0 只实现 SPE)
  online_learner.py        ~220 行   (SGD + EWC，多位点接口预留)
  decision_router.py       ~80 行    (阈值路由 + RPE 自适应接口)
  cortex_interface.py      ~120 行   (OpenAI API 调用)
  fluid_memory.py          ~200 行   (流体记忆 v0: strength + 衰减，无睡眠)
  main.py                  ~100 行   (流水线串联)
  ──────────────────────────────────
  合计:                    ~1160 行

  + experiment.ipynb       Jupyter 实验笔记本（可视化 + 分析）


  验证实验设计:
  ──────────────────────────────────
  1. 准备 200+ 条 tool_call 记录
     （可从 OpenClaw / 自建 Agent 运行中收集）
     包含: send_email, read_file, search_web, create_task 等
  2. 前 50 条全部走 LLM（冷启动阶段，小脑只观察）
  3. 第 51~100 条开始影子执行（小脑预评估 vs 实际结果）
  4. 第 101~200 条按群体置信度路由
  5. 记录: 每条的预测准确率、群体置信度、各头一致性、延迟

  成功标准:
  • 第 100 条之后，>50% 的 tool_call 走快路径
  • 快路径准确率 > 90%
  • 高风险操作拦截率 > 95%
  • 快路径延迟 < 10ms
  • 群体置信度与实际准确率的相关系数 > 0.7
  ──────────────────────────────────────────────────────────────
```

---

## 六、项目文件结构

```
  digital-cerebellum/
  ├── docs/
  │   ├── architecture.md              # 架构设计 + 神经科学映射
  │   └── implementation.md            # 实现指南（本文档）
  ├── digital_cerebellum/              # pip install digital-cerebellum
  │   ├── __init__.py                  # 包入口，导出核心 API
  │   ├── main.py                      # DigitalCerebellum 主管线
  │   ├── brain.py                     # DigitalBrain（完整认知架构）
  │   ├── core/                        # 核心计算原语
  │   │   ├── types.py                 # 核心数据类型
  │   │   ├── feature_encoder.py       # 特征编码器（苔状纤维）
  │   │   ├── pattern_separator.py     # ② 模式分离器 (RFF + top-k)
  │   │   ├── prediction_engine.py     # ③ K头预测引擎 + 树突掩码 + StateConditioner
  │   │   ├── error_comparator.py      # ⑦ 三通道误差比较器 (SPE/TPE/RPE)
  │   │   ├── online_learner.py        # ⑧ 在线学习 (SGD + EWC + replay)
  │   │   ├── microzone.py             # 微区基类 + 学习信号
  │   │   ├── frequency_filter.py      # ⑩A 频率滤波层 (Phase 2)
  │   │   ├── golgi_gate.py            # ⑩B Golgi 反馈门控 (Phase 2)
  │   │   └── state_estimator.py       # ⑩C 状态估计器 (Phase 2)
  │   ├── routing/
  │   │   └── decision_router.py       # ④ 决策路由器 (DCN)
  │   ├── cortex/
  │   │   ├── cortex_interface.py      # ⑥ 皮层接口 (LLM)
  │   │   └── consolidation.py         # 任务巩固流水线
  │   ├── memory/
  │   │   ├── fluid_memory.py          # 流体记忆 (衰减/再巩固/检索)
  │   │   └── sleep_cycle.py           # 睡眠周期 (巩固/抽象/蒸馏)
  │   ├── microzones/                  # 可插拔微区（通用小脑变换）
  │   │   ├── tool_call.py             # tool_call 安全评估微区
  │   │   └── payment.py               # 支付风险微区
  │   └── emergence/                   # Phase 3: 涌现认知
  │       ├── __init__.py
  │       ├── somatic_marker.py        # ⑫A 躯体标记 / 直觉
  │       ├── curiosity_drive.py       # ⑫B 好奇心驱动
  │       └── self_model.py            # ⑫C 自我模型 / 元认知
  ├── memory/
  │   ├── fluid_memory.py              # 流体记忆 (感知→短期→长期)
  │   ├── skill_store.py               # ⑬ Phase 4: 程序性记忆 (SkillStore)
  │   └── sleep_cycle.py               # 睡眠巩固
  ├── microzones/
  │   ├── __init__.py                  # ALL_MICROZONES 导出
  │   ├── tool_call.py                 # ToolCallMicrozone
  │   ├── payment.py                   # PaymentMicrozone
  │   ├── shell_command.py             # ⑭ Phase 5: ShellCommandMicrozone
  │   ├── file_operation.py            # ⑭ Phase 5: FileOperationMicrozone
  │   ├── api_call.py                  # ⑭ Phase 5: APICallMicrozone
  │   └── response_prediction.py       # ⑭ Phase 5: ResponsePredictionMicrozone
  ├── micro_ops/
  │   ├── __init__.py                  # MicroOpEngine 导出
  │   ├── engine.py                    # ⑮ Phase 6: MicroOpEngine (连续控制循环)
  │   └── environments.py              # ⑮ Phase 6: TargetTracker, BalanceBeam
  ├── core/
  │   ├── state_encoder.py             # ⑯ Phase 6: StateEncoder (数值状态编码)
  │   ├── forward_model.py             # ⑯ Phase 6: ForwardModel (前向模型)
  │   └── action_encoder.py            # ⑯ Phase 6: ActionEncoder (动作空间编码)
  ├── benchmarks/
  │   ├── dataset.py                   # 统一数据集格式
  │   ├── sequential_dataset.py        # 时序场景数据集
  │   ├── runner.py                    # Benchmark 运行器 + 消融配置
  │   ├── run_all.py                   # 全量 benchmark 入口
  │   └── results/                     # 保存的 benchmark 结果
  ├── experiments/
  │   └── closed_loop.py               # 闭环实验（真实 LLM 蒸馏）
  ├── examples/
  │   ├── openai_agent.py              # OpenAI agent 集成示例
  │   ├── langchain_guard.py           # LangChain 工具守卫示例
  │   ├── multi_microzone.py           # 多微区示例
  │   └── brain_demo.py                # DigitalBrain 完整演示
  ├── examples/
  │   ├── skill_acquisition_demo.py    # Phase 4 技能习得演示
  │   └── micro_ops_demo.py            # Phase 6 微操引擎演示
  ├── tests/
  │   ├── test_core.py                 # 核心组件测试 (23)
  │   ├── test_microzones.py           # 微区测试 (15)
  │   ├── test_new_microzones.py       # Phase 5 微区测试 (33)
  │   ├── test_phase1.py               # Phase 1 测试 (17)
  │   ├── test_phase2.py               # Phase 2 测试 (18)
  │   ├── test_phase3.py               # Phase 3 测试 (33)
  │   ├── test_skill_store.py          # Phase 4 技能库测试 (26)
  │   ├── test_temporal_detector.py    # 时序检测器测试 (11)
  │   ├── test_mcp_server.py           # MCP Server 测试 (12)
  │   └── test_micro_ops.py            # Phase 6 微操测试 (34)
  ├── paper/
  │   ├── main.tex                     # 论文 LaTeX 源码
  │   └── references.bib               # 参考文献
  ├── pyproject.toml                   # 包配置
  ├── LICENSE                          # MIT
  └── README.md
```
