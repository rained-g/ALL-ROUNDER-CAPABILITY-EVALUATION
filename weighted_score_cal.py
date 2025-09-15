import pandas as pd
import numpy as np

# 读入结果文件
df = pd.read_csv('./results/Metric.csv', encoding='utf-8')

# 定义各亚组在原始测试集中的占比（真实先验）
prevalence = {
    'cf': 0.067,
    'cfpf': 0.479,
    'cfcl': 0.079,
    'all': 0.374
}

# 计算不同加权策略
def get_weights(strategy='prevalence'):
    if strategy == 'equal':
        # 每个亚组权重相同
        w = {k: 1.0 for k in prevalence.keys()}
    elif strategy == 'prevalence':
        # 使用真实分布
        w = prevalence
    elif strategy == 'inv':
        # 逆频率
        w = {k: 1.0/v for k, v in prevalence.items()}
    elif strategy == 'inv_sqrt':
        # 平方根倒数
        w = {k: 1.0/np.sqrt(v) for k, v in prevalence.items()}
    else:
        raise ValueError("Unknown strategy")
    
    # 归一化
    total = sum(w.values())
    return {k: v/total for k, v in w.items()}

# 计算加权分数
def compute_weighted_scores(df, strategy='prevalence'):
    weights = get_weights(strategy)
    results = []
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        weighted_scores = {'model': model, 'strategy': strategy}
        
        for metric in ['BLEU-1', 'ROUGEL', 'neg-macro F1 14', 'micro F1 14']:
            score = 0.0
            for _, row in model_df.iterrows():
                subgroup = row['subgroup']
                score += weights[subgroup] * row[metric]
            weighted_scores[metric] = score
        results.append(weighted_scores)
    
    return pd.DataFrame(results)

# 合并所有策略结果
all_results = []
for strat in ['equal', 'prevalence', 'inv', 'inv_sqrt']:
    all_results.append(compute_weighted_scores(df, strat))

final_df = pd.concat(all_results, ignore_index=True)

# 按策略和模型展示
print(final_df)

# 也可以存储到文件
final_df.to_csv('./results/Weighted_Metrics.csv', index=False)
