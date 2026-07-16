# 智能座舱Agent - 认知记忆模块

负责处理、存储和检索用户相关信息，支持个性化交互和上下文感知功能。

## 项目结构

```
cognitive_memory/
├── __init__.py                        # 模块入口
├── models/
│   ├── memory.py                      # 7个核心数据模型 (MemoryItem, UserProfile, InteractionRecord, SceneContext, BehaviorPattern等)
│   └── __init__.py
├── storage/
│   ├── base.py                        # 4个存储抽象接口 (BaseMemoryStore, BaseProfileStore, BaseInteractionStore, BasePatternStore)
│   ├── short_term.py                  # 短期记忆存储 (LRU内存字典, O(1)查询, 可配置容量/过期时间)
│   ├── long_term.py                   # 长期记忆存储 (SQLite持久化, 4张表, WAL模式, 复合索引)
│   └── __init__.py
├── learner/
│   ├── memory_decay.py                # 记忆衰减与强化引擎 (艾宾浩斯遗忘曲线, 间隔重复, 重要性加权, 情感增强)
│   ├── pattern_learner.py             # 5类行为模式学习器 (温度/路线/媒体/交互/时间模式)
│   └── __init__.py
├── core/
│   ├── memory_manager.py              # 中央编排器 (14个公共接口, 完整记忆生命周期管理)
│   ├── user_profile_builder.py        # 用户画像构建器 (渐进式特征提取, 置信度评估)
│   ├── context_engine.py              # 上下文感知引擎 (场景相似度匹配, 需求预测, 场景变化检测)
│   ├── privacy.py                     # 数据安全 (AES-256-GCM加密, 脱敏, 审计日志)
│   └── __init__.py
├── api/
│   ├── routes.py                      # 8个RESTful API接口 (统一ApiResponse格式)
│   └── __init__.py
├── tests/
│   ├── test_memory.py                 # 15个单元测试
│   ├── test_integration.py            # 8个集成测试
│   └── __init__.py
├── pyproject.toml                     # 项目配置
└── requirements.txt                   # 依赖清单
```

## 快速使用

### 安装

```bash
pip install -r requirements.txt
```

### 基本用法

```python
import asyncio
from cognitive_memory.core import MemoryManager
from cognitive_memory.models.memory import (
    InteractionRecord,
    SceneContext,
    MemoryQuery,
    MemoryType,
)

async def main():
    # 初始化记忆管理器
    manager = MemoryManager(db_path="cockpit_memory.db")

    # 创建场景上下文
    scene = SceneContext(
        time_of_day="morning",
        weather="sunny",
        traffic_condition="smooth",
        road_type="highway",
        location_type="home",
        cabin_temperature=22.0,
    )

    # 记录用户交互
    interaction = InteractionRecord(
        user_id="user_001",
        interaction_type="voice_command",
        intent="navigate",
        raw_input="导航到公司",
        processed_input={"destination": "公司", "trip_purpose": "commute"},
        scene_context=scene,
        system_response={"action": "start_navigation", "destination": "公司"},
        was_successful=True,
        user_satisfaction=0.95,
        session_id="session_001",
    )
    memory_id = await manager.record_interaction(interaction)

    # 查询记忆
    result = await manager.query_memories(
        user_id="user_001",
        keywords=["导航"],
        max_results=10,
        sort_by="relevance",
    )
    for item in result.items:
        print(f"记忆: {item.content}, 强度: {item.strength:.2f}")

    # 构建用户画像
    profile = await manager.build_user_profile("user_001")
    print(f"温度偏好: {profile.temperature_preference}°C")
    print(f"画像置信度: {profile.confidence_score:.2f}")

    # 预测用户需求
    current_scene = SceneContext(
        time_of_day="morning",
        weather="sunny",
        traffic_condition="moderate",
        road_type="highway",
        location_type="home",
    )
    predictions = await manager.predict_user_needs("user_001", current_scene)
    for p in predictions:
        print(f"预测: {p['pattern_name']}, 置信度: {p['confidence']:.2f}")

    # 获取行为模式
    patterns = await manager.get_behavior_patterns("user_001")
    for p in patterns:
        print(f"模式: {p.pattern_name} ({p.pattern_type}), 置信度: {p.confidence:.2f}")

    # 删除用户数据 (GDPR合规)
    deleted = await manager.forget_user_data("user_001")
    print(f"已删除 {deleted} 条记录")

asyncio.run(main())
```

### 运行测试

```bash
python -m pytest cognitive_memory/tests/ -v
```

## 模块关系

```
InteractionRecord → MemoryEncoder → MemoryItem
       ↓                                 ↓
UserProfileBuilder ← BehaviorPatterns ← PatternLearner
       ↓                                 ↓
 ProfileStore                       MemoryStore (Short/Long)
       ↓                                 ↓
ContextEngine ←──────────────────────────┘
```

## 质量评估指标体系

评估认知记忆模块质量需从以下六个维度进行全面衡量，每个维度包含量化指标和定性标准。

### 一、记忆准确性 (Memory Accuracy)

衡量记忆存储和检索的精确程度。

| 指标 | 计算方式 | 目标值 | 说明 |
|------|----------|--------|------|
| **Precision@K** | `TP@K / K` | ≥ 0.85 | 检索结果中相关记忆的占比，评估检索精确度 |
| **Recall@K** | `TP@K / 相关记忆总数` | ≥ 0.80 | 所有相关记忆中被成功检索的比例，评估覆盖率 |
| **MRR** (Mean Reciprocal Rank) | `1/N * Σ(1/rank_i)` | ≥ 0.90 | 第一个相关结果排名的倒数均值，评估排序质量 |
| **NDCG@K** | `DCG@K / IDCG@K` | ≥ 0.85 | 归一化折损累计增益，综合考虑排序位置和相关性等级 |
| **上下文匹配准确率** | `正确匹配数 / 总查询数` | ≥ 0.80 | 场景上下文与记忆检索的匹配正确率 |
| **画像准确率** | `正确属性数 / 画像属性总数` | ≥ 0.75 | 用户画像中各项属性与实际用户特征的一致程度 |

**定性评估标准：**
- 检索结果与用户意图的语义相关性
- 相似场景下记忆召回的一致性
- 边缘场景（如首次交互、异常输入）的鲁棒性
- 历史记忆与当前上下文的关联合理性

### 二、信息保留率 (Information Retention Rate)

衡量记忆随时间衰减和持久化的效果。

| 指标 | 计算方式 | 目标值 | 说明 |
|------|----------|--------|------|
| **短期记忆保留率** | `t时刻可检索数 / 初始存储数` | 1h: ≥ 80%, 24h: ≥ 50% | 短期记忆随时间衰减后的存活比例 |
| **长期记忆持久率** | `N天后可检索数 / 初始存储数` | 7天: ≥ 90%, 30天: ≥ 70% | 长期记忆在持久化存储中的存活比例 |
| **巩固成功率** | `成功转化数 / 候选转化数` | ≥ 0.60 | 短期记忆成功转化为长期记忆的比例 |
| **衰减曲线拟合度** | `1 - MSE(实际衰减, 理论衰减)` | ≥ 0.85 | 实际衰减曲线与艾宾浩斯理论曲线的拟合优度 |
| **关键记忆保护率** | `CRITICAL记忆保留数 / CRITICAL记忆总数` | ≥ 0.98 | 高重要性记忆的长期保留比例 |
| **遗忘合理性** | `合理遗忘数 / 总遗忘数` | ≥ 0.90 | 被遗忘的记忆中确实无价值的比例 |

**定性评估标准：**
- 衰减速度是否与记忆重要性正相关
- 是否在遗忘前经历了合理的衰减过渡期
- 高频访问的记忆是否得到有效强化
- 长时间未访问的低价值记忆是否被正确清理

### 三、响应速度 (Response Speed)

衡量系统在不同负载下的性能表现。

| 指标 | 计算方式 | 目标值 | 说明 |
|------|----------|--------|------|
| **存储延迟 P50/P95/P99** | 百分位延迟统计 | P50: < 10ms, P95: < 50ms, P99: < 100ms | 记忆存储操作的响应时间分布 |
| **检索延迟 P50/P95/P99** | 百分位延迟统计 | P50: < 20ms, P95: < 100ms, P99: < 200ms | 记忆检索操作的响应时间分布 |
| **存储吞吐量** | `操作数 / 秒` | ≥ 500 ops/s | 单节点每秒可处理的存储操作数 |
| **查询吞吐量** | `查询数 / 秒` | ≥ 1000 qps | 单节点每秒可处理的查询请求数 |
| **并发性能** | `吞吐量 @ N并发` | 10并发: 线性扩展 ≥ 80% | 并发场景下的吞吐量衰减比例 |
| **画像构建耗时** | 端到端耗时 | < 500ms (200条交互) | 用户画像全量构建的时间 |
| **维护周期耗时** | 单次维护耗时 | < 1s (1000条记忆) | 后台衰减检查和清理的耗时 |

**定性评估标准：**
- 冷启动（首次请求）到热稳定（缓存命中）的性能提升幅度
- 数据量增长时的性能退化曲线（是否接近线性）
- 存储和查询操作是否产生阻塞等待
- 系统资源（CPU/内存）占用是否合理

### 四、泛化能力 (Generalization)

衡量模型对未见过的用户、场景和数据的适应能力。

| 指标 | 计算方式 | 目标值 | 说明 |
|------|----------|--------|------|
| **跨场景迁移准确率** | `新场景正确率 / 已知场景正确率` | ≥ 0.70 | 已学习模式在相似但不同场景下的预测准确率衰减 |
| **新用户冷启动效果** | `第N次交互的预测命中率` | 第5次: ≥ 0.40, 第20次: ≥ 0.65 | 新用户随着交互次数增加的预测命中率提升曲线 |
| **模式发现覆盖率** | `自动发现模式数 / 人工标注模式数` | ≥ 0.80 | 自动学习到的模式对已知模式的覆盖程度 |
| **跨用户迁移能力** | `同类用户模式共享的准确率` | ≥ 0.60 | 从相似用户群体迁移行为模式的预测准确率 |
| **模式泛化率** | `泛化成功数 / 新场景模式匹配数` | ≥ 0.70 | 已有模式在新场景中依然有效的比例 |
| **学习速度** | `达到稳定准确率所需交互次数` | ≤ 30次 | 从初始状态到画像稳定所需的最小交互量 |

**定性评估标准：**
- 是否能够识别用户行为的变化趋势并自适应调整
- 不同用户画像之间的区分度是否足够
- 对异常行为模式的过滤能力
- 季节性/周期性变化的适应能力

### 五、抗干扰性 (Anti-interference)

衡量系统在噪声、冲突和异常条件下的稳定性。

| 指标 | 计算方式 | 目标值 | 说明 |
|------|----------|--------|------|
| **噪声鲁棒性** | `含噪准确率 / 纯净准确率` | ≥ 0.85 (20%噪声) | 在输入数据含噪声时的准确率保持比例 |
| **冲突处理成功率** | `正确处理数 / 冲突场景数` | ≥ 0.80 | 多条矛盾信息同时存在时的正确处理比例 |
| **数据一致性评分** | `一致操作数 / 总操作数` | ≥ 0.95 | 并发操作下数据不产生脏读/脏写的比例 |
| **异常输入拒绝率** | `正确拒绝数 / 异常输入数` | ≥ 0.95 | 对格式错误、超出范围等异常输入的正确拒绝 |
| **记忆污染防护率** | `未受影响记忆数 / 总记忆数` | ≥ 0.98 | 错误数据注入后未被污染的记忆比例 |
| **恢复能力** | `恢复后准确率 / 故障前准确率` | ≥ 0.95 | 系统从异常状态恢复后的性能保持比例 |

**定性评估标准：**
- 对用户偶然误操作（如说错指令）的识别和纠正能力
- 传感器数据异常（如GPS漂移）时的场景推断稳定性
- 多用户共用车辆时的用户识别和记忆隔离
- 系统升级或数据迁移后的记忆一致性

### 六、用户体验 (User Experience)

衡量记忆模块对最终用户交互体验的实际改善效果。

| 指标 | 计算方式 | 目标值 | 说明 |
|------|----------|--------|------|
| **需求预测命中率** | `预测命中数 / 总预测数` | ≥ 0.70 | 主动推荐被用户接受的比例 |
| **推荐接受率** | `用户采纳数 / 推荐总数` | ≥ 0.65 | 系统推荐被用户实际采纳的比例 |
| **交互减少率** | `1 - (有记忆交互步数 / 无记忆交互步数)` | ≥ 0.30 | 引入记忆模块后完成相同任务所需交互步数的减少比例 |
| **用户满意度相关性** | `满意度与记忆质量的相关性系数` | ≥ 0.50 | 记忆模块表现与用户满意度评分的相关性 |
| **首次交互成功率** | `首次交互成功数 / 总交互数` | ≥ 0.85 | 用户首次发出指令就被正确理解和执行的比率 |
| **个性化感知度** | `用户感知到个性化的场景比例` | ≥ 0.60 | 用户可感知到系统"记住"了其偏好的场景占比 |

**定性评估标准：**
- 用户是否感受到系统"越来越懂我"的渐进式体验改善
- 主动推荐是否自然、不突兀、时机恰当
- 系统是否避免了过度个性化造成的"信息茧房"效应
- 用户对数据隐私和安全性的信任度
- 记忆相关的系统行为是否可解释、可追溯

### 综合评价框架

以上六个维度构成完整的质量评估体系，建议采用加权评分：

| 维度 | 权重 | 评估重点 |
|------|------|----------|
| 记忆准确性 | 25% | 核心检索质量 |
| 信息保留率 | 20% | 长期记忆持久性 |
| 响应速度 | 20% | 实时交互体验 |
| 用户体验 | 15% | 最终价值体现 |
| 泛化能力 | 10% | 扩展适应能力 |
| 抗干扰性 | 10% | 系统稳定性 |

**综合得分** = Σ(维度得分 × 维度权重)，各维度得分 = 该维度各项指标达标率的加权平均。

### 基准测试

```bash
# 运行完整基准测试
python -m pytest cognitive_memory/tests/test_benchmark.py -v -s

# 仅运行性能测试
python -m pytest cognitive_memory/tests/test_benchmark.py -v -k "perf"

# 仅运行准确率测试
python -m pytest cognitive_memory/tests/test_benchmark.py -v -k "accuracy"
```