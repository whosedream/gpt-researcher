"""Benchmark dataset: 20 Chinese research questions for E2E testing.

Each question exercises different research capabilities:
- Factual lookup, comparison, analysis, synthesis
- Multi-hop reasoning, recent events, historical topics
- Technical, scientific, economic, cultural domains
"""

BENCHMARK_QUESTIONS = [
    # --- Factual / Direct ---
    {
        "id": 1,
        "question": "量子计算的基本原理是什么？目前主要的量子比特实现方式有哪些？",
        "category": "technical",
        "expected_keywords": ["量子比特", "叠加", "纠缠", "超导", "离子阱"],
    },
    {
        "id": 2,
        "question": "CRISPR-Cas9基因编辑技术的最新临床应用进展如何？",
        "category": "science",
        "expected_keywords": ["CRISPR", "临床试验", "基因治疗"],
    },
    {
        "id": 3,
        "question": "2024年全球半导体产业链发生了哪些重大变化？",
        "category": "economics",
        "expected_keywords": ["半导体", "芯片", "供应链", "出口管制"],
    },
    # --- Comparison ---
    {
        "id": 4,
        "question": "比较中国和美国在人工智能领域的发展策略和政策差异",
        "category": "comparison",
        "expected_keywords": ["人工智能", "政策", "监管", "投资"],
    },
    {
        "id": 5,
        "question": "锂电池和固态电池在能量密度、安全性和成本方面有何差异？",
        "category": "technical",
        "expected_keywords": ["锂电池", "固态电池", "能量密度", "安全性"],
    },
    # --- Analysis / Synthesis ---
    {
        "id": 6,
        "question": "分析俄乌冲突对全球粮食安全的影响及各国应对措施",
        "category": "analysis",
        "expected_keywords": ["俄乌冲突", "粮食安全", "小麦", "供应链"],
    },
    {
        "id": 7,
        "question": "全球碳中和目标下，氢能产业的发展前景和主要挑战是什么？",
        "category": "analysis",
        "expected_keywords": ["碳中和", "氢能", "绿氢", "燃料电池"],
    },
    {
        "id": 8,
        "question": "mRNA疫苗技术平台除了新冠疫苗外还有哪些应用方向？",
        "category": "science",
        "expected_keywords": ["mRNA", "疫苗", "肿瘤", "个性化医疗"],
    },
    # --- Multi-hop / Complex ---
    {
        "id": 9,
        "question": "大语言模型的训练数据质量如何影响模型性能？有哪些数据清洗方法？",
        "category": "technical",
        "expected_keywords": ["大语言模型", "训练数据", "数据清洗", "去重"],
    },
    {
        "id": 10,
        "question": "量子纠缠现象如何应用于量子通信和量子密钥分发？",
        "category": "technical",
        "expected_keywords": ["量子纠缠", "量子通信", "QKD", "BB84"],
    },
    # --- Recent Events ---
    {
        "id": 11,
        "question": "2024年巴黎奥运会中国代表团取得了哪些突破性成绩？",
        "category": "recent",
        "expected_keywords": ["巴黎奥运会", "中国", "金牌", "成绩"],
    },
    {
        "id": 12,
        "question": "OpenAI GPT-5和Google Gemini 2.0在能力上有哪些新突破？",
        "category": "recent",
        "expected_keywords": ["GPT-5", "Gemini", "多模态", "推理"],
    },
    # --- Historical / Cultural ---
    {
        "id": 13,
        "question": "丝绸之路对东西方文化交流产生了哪些深远影响？",
        "category": "cultural",
        "expected_keywords": ["丝绸之路", "贸易", "文化", "佛教"],
    },
    {
        "id": 14,
        "question": "量子力学的发展历程中，哪些关键实验改变了物理学的认知？",
        "category": "science",
        "expected_keywords": ["量子力学", "双缝实验", "光电效应", "薛定谔"],
    },
    # --- Technical Deep Dive ---
    {
        "id": 15,
        "question": "RISC-V指令集架构为什么受到越来越多公司的关注？",
        "category": "technical",
        "expected_keywords": ["RISC-V", "开源", "指令集", "ARM"],
    },
    {
        "id": 16,
        "question": "可重复使用火箭技术SpaceX是如何实现的？对航天产业有什么影响？",
        "category": "technical",
        "expected_keywords": ["SpaceX", "可重复使用", "火箭", "着陆"],
    },
    # --- Economic / Policy ---
    {
        "id": 17,
        "question": "全球数字货币（CBDC）的发展现状和各国央行的推进策略是什么？",
        "category": "economics",
        "expected_keywords": ["CBDC", "数字货币", "央行", "数字人民币"],
    },
    {
        "id": 18,
        "question": "人工智能在医疗诊断中的应用现状和面临的主要挑战是什么？",
        "category": "analysis",
        "expected_keywords": ["AI", "医疗诊断", "医学影像", "数据隐私"],
    },
    # --- Cross-domain ---
    {
        "id": 19,
        "question": "脑机接口技术的最新进展和潜在应用领域有哪些？",
        "category": "technical",
        "expected_keywords": ["脑机接口", "Neuralink", "神经信号", "康复"],
    },
    {
        "id": 20,
        "question": "全球稀土资源的分布格局和供应链安全面临哪些挑战？",
        "category": "economics",
        "expected_keywords": ["稀土", "供应链", "中国", "战略资源"],
    },
]

# Category distribution for balanced testing
CATEGORY_COUNTS = {}
for q in BENCHMARK_QUESTIONS:
    cat = q["category"]
    CATEGORY_COUNTS[cat] = CATEGORY_COUNTS.get(cat, 0) + 1
