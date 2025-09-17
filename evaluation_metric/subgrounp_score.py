from evaluation import dataframe_data,cal_pos_F1,cal_neg_F1,cal_BLEU,calculate_rouge_l
import numpy as np

#原始报告的路径-new-try1
origin_all_test_new = r'./origin_report/all_test.csv'
all_test_labeled_new = r'./origin_report/test_data_labeled.csv'


total_bleu = {'BLEU-1': 0, 'BLEU-2': 0, 'BLEU-3': 0, 'BLEU-4': 0}
total_pos = {"pos-Macro-F1-14":0.0,"pos-Micro-F1-14":0.0,"pos-Macro-F1-5":0.0,"pos-Micro-F1-5":0.0}
total_neg = {"neg-Macro-F1-14":0.0,"neg-Micro-F1-14":0.0,"neg-Macro-F1-5":0.0,"neg-Micro-F1-5":0.0}
total_rouge = 0

CF_bleu={'BLEU-1': 0, 'BLEU-2': 0, 'BLEU-3': 0, 'BLEU-4': 0}
CFCL_bleu={'BLEU-1': 0, 'BLEU-2': 0, 'BLEU-3': 0, 'BLEU-4': 0}
CFPF_bleu={'BLEU-1': 0, 'BLEU-2': 0, 'BLEU-3': 0, 'BLEU-4': 0}
all_bleu = {'BLEU-1': 0, 'BLEU-2': 0, 'BLEU-3': 0, 'BLEU-4': 0}
CF_pos={"pos-Macro-F1-14":0.0,"pos-Micro-F1-14":0.0,"pos-Macro-F1-5":0.0,"pos-Micro-F1-5":0.0}
CFCL_pos={"pos-Macro-F1-14":0.0,"pos-Micro-F1-14":0.0,"pos-Macro-F1-5":0.0,"pos-Micro-F1-5":0.0}
CFPF_pos={"pos-Macro-F1-14":0.0,"pos-Micro-F1-14":0.0,"pos-Macro-F1-5":0.0,"pos-Micro-F1-5":0.0}
all_pos = {"pos-Macro-F1-14":0.0,"pos-Micro-F1-14":0.0,"pos-Macro-F1-5":0.0,"pos-Micro-F1-5":0.0}
CF_neg={"neg-Macro-F1-14":0.0,"neg-Micro-F1-14":0.0,"neg-Macro-F1-5":0.0,"neg-Micro-F1-5":0.0}
CFCL_neg={"neg-Macro-F1-14":0.0,"neg-Micro-F1-14":0.0,"neg-Macro-F1-5":0.0,"neg-Micro-F1-5":0.0}
CFPF_neg={"neg-Macro-F1-14":0.0,"neg-Micro-F1-14":0.0,"neg-Macro-F1-5":0.0,"neg-Micro-F1-5":0.0}
all_neg = {"neg-Macro-F1-14":0.0,"neg-Micro-F1-14":0.0,"neg-Macro-F1-5":0.0,"neg-Micro-F1-5":0.0}
CF_rouge=0
CFPF_rouge=0
CFCL_rouge=0
all_rouge = 0
length = 1
for j in range(length):
    file_name = f"./results/model1_standard{j+1}.csv"
    print(file_name)
    df, df_G, references, candidates= dataframe_data(origin_all_test_new,all_test_labeled_new,file_name)
    bleu = cal_BLEU(references, candidates)
    total_bleu["BLEU-1"] += bleu["BLEU-1"]
    total_bleu["BLEU-2"] += bleu["BLEU-2"]
    total_bleu["BLEU-4"] += bleu["BLEU-4"]
    precision, recall, f1 = calculate_rouge_l(references, candidates)
    total_rouge += f1*100
    pos = cal_pos_F1(df,df_G)
    neg = cal_neg_F1(df,df_G)
    total_pos["pos-Macro-F1-14"] += pos["pos-Macro-F1-14"]
    total_pos["pos-Micro-F1-14"] += pos["pos-Micro-F1-14"]
    total_pos["pos-Macro-F1-5"] += pos["pos-Macro-F1-5"]
    total_pos["pos-Micro-F1-5"] += pos["pos-Micro-F1-5"]
    
    total_neg["neg-Macro-F1-14"] += neg["neg-Macro-F1-14"]
    total_neg["neg-Micro-F1-14"] += neg["neg-Micro-F1-14"]
    total_neg["neg-Macro-F1-5"] += neg["neg-Macro-F1-5"]
    total_neg["neg-Micro-F1-5"] += neg["neg-Micro-F1-5"]

    types = df['type'].unique()
    print(types)
    print(types)
    for t in types:
        print(f"\n=== Type: {t} ===")
        # 筛选出当前类型的数据（choose type or Subgroup              "type" is the Views goup,"Subgoup" is the Rare subgroup and in "Subgoup" CF is Rare CFCL is Common）
        # df_t = df[df['Subgroup'] == t].reset_index(drop=True)
        # df_G_t = df_G[df_G['Subgroup'] == t].reset_index(drop=True)
        df_t = df[df['type'] == t].reset_index(drop=True)
        df_G_t = df_G[df_G['type'] == t].reset_index(drop=True)
        
        # 提取报告文本
        references_t = df_t['report'].tolist()
        candidates_t = df_G_t['report'].tolist()
        
        # 计算 BLEU 分数
        bleu = cal_BLEU(references_t, candidates_t)
        
        # 计算 ROUGE-L 分数
        precision, recall, f1 = calculate_rouge_l(references_t, candidates_t)
        # print(f"ROUGE-L Precision: {precision*100:.2f}, Recall: {recall*100:.2f}, F1: {f1*100:.2f}")
        
        # 计算正类 F1 分数
        pos = cal_pos_F1(df_t, df_G_t)
        
        # 计算负类 F1 分数
        neg = cal_neg_F1(df_t, df_G_t)
        if t=='CF':
            CF_bleu["BLEU-1"] += bleu["BLEU-1"]
            CF_bleu["BLEU-2"] += bleu["BLEU-2"]
            CF_bleu["BLEU-4"] += bleu["BLEU-4"]

            CF_pos["pos-Macro-F1-14"] += pos["pos-Macro-F1-14"]
            CF_pos["pos-Micro-F1-14"] += pos["pos-Micro-F1-14"]
            CF_pos["pos-Macro-F1-5"] += pos["pos-Macro-F1-5"]
            CF_pos["pos-Micro-F1-5"] += pos["pos-Micro-F1-5"]

            CF_neg["neg-Macro-F1-14"] += neg["neg-Macro-F1-14"]
            CF_neg["neg-Micro-F1-14"] += neg["neg-Micro-F1-14"]
            CF_neg["neg-Macro-F1-5"] += neg["neg-Macro-F1-5"]
            CF_neg["neg-Micro-F1-5"] += neg["neg-Micro-F1-5"]

            CF_rouge += f1*100
        elif t=='CFCL':
            CFCL_bleu["BLEU-1"] += bleu["BLEU-1"]
            CFCL_bleu["BLEU-2"] += bleu["BLEU-2"]
            CFCL_bleu["BLEU-4"] += bleu["BLEU-4"]

            CFCL_pos["pos-Macro-F1-14"] += pos["pos-Macro-F1-14"]
            CFCL_pos["pos-Micro-F1-14"] += pos["pos-Micro-F1-14"]
            CFCL_pos["pos-Macro-F1-5"] += pos["pos-Macro-F1-5"]
            CFCL_pos["pos-Micro-F1-5"] += pos["pos-Micro-F1-5"]


            CFCL_neg["neg-Macro-F1-14"] += neg["neg-Macro-F1-14"]
            CFCL_neg["neg-Micro-F1-14"] += neg["neg-Micro-F1-14"]
            CFCL_neg["neg-Macro-F1-5"] += neg["neg-Macro-F1-5"]
            CFCL_neg["neg-Micro-F1-5"] += neg["neg-Micro-F1-5"]

            CFCL_rouge += f1*100
        elif t=='CFPF':
            CFPF_bleu["BLEU-1"] += bleu["BLEU-1"]
            CFPF_bleu["BLEU-2"] += bleu["BLEU-2"]
            CFPF_bleu["BLEU-4"] += bleu["BLEU-4"]

            CFPF_pos["pos-Macro-F1-14"] += pos["pos-Macro-F1-14"]
            CFPF_pos["pos-Micro-F1-14"] += pos["pos-Micro-F1-14"]
            CFPF_pos["pos-Macro-F1-5"] += pos["pos-Macro-F1-5"]
            CFPF_pos["pos-Micro-F1-5"] += pos["pos-Micro-F1-5"]


            CFPF_neg["neg-Macro-F1-14"] += neg["neg-Macro-F1-14"]
            CFPF_neg["neg-Micro-F1-14"] += neg["neg-Micro-F1-14"]
            CFPF_neg["neg-Macro-F1-5"] += neg["neg-Macro-F1-5"]
            CFPF_neg["neg-Micro-F1-5"] += neg["neg-Micro-F1-5"]

            CFPF_rouge += f1*100
        
        else:
            all_bleu["BLEU-1"] += bleu["BLEU-1"]
            all_bleu["BLEU-2"] += bleu["BLEU-2"]
            all_bleu["BLEU-4"] += bleu["BLEU-4"]

            all_pos["pos-Macro-F1-14"] += pos["pos-Macro-F1-14"]
            all_pos["pos-Micro-F1-14"] += pos["pos-Micro-F1-14"]
            all_pos["pos-Macro-F1-5"] += pos["pos-Macro-F1-5"]
            all_pos["pos-Micro-F1-5"] += pos["pos-Micro-F1-5"]


            all_neg["neg-Macro-F1-14"] += neg["neg-Macro-F1-14"]
            all_neg["neg-Micro-F1-14"] += neg["neg-Micro-F1-14"]
            all_neg["neg-Macro-F1-5"] += neg["neg-Macro-F1-5"]
            all_neg["neg-Micro-F1-5"] += neg["neg-Micro-F1-5"]

            all_rouge += f1*100

print("total")
print(f'BLEU-1:{total_bleu["BLEU-1"]/length:.2f}, BLEU-2: {total_bleu["BLEU-2"]/length:.2f}, BLEU-4: {total_bleu["BLEU-4"]/length:.2f}')
print(f'rouge_L:{total_rouge/length:.2f}')
print(f'pos-Macro-F1-14:{total_pos["pos-Macro-F1-14"]/length:.2f},pos-Micro-F1-14:{total_pos["pos-Micro-F1-14"]/length:.2f},pos-Macro-F1-5:{total_pos["pos-Macro-F1-5"]/length:.2f},pos-Micro-F1-5:{total_pos["pos-Micro-F1-5"]/length:.2f}')
print(f'neg-Macro-F1-14:{total_neg["neg-Macro-F1-14"]/length:.2f},neg-Micro-F1-14:{total_neg["neg-Micro-F1-14"]/length:.2f},neg-Macro-F1-5:{total_neg["neg-Macro-F1-5"]/length:.2f},neg-Micro-F1-5:{total_neg["neg-Micro-F1-5"]/length:.2f}')



print("CF")
print(f'BLEU-1:{CF_bleu["BLEU-1"]/length:.2f}, BLEU-2: {CF_bleu["BLEU-2"]/length:.2f}, BLEU-4: {CF_bleu["BLEU-4"]/length:.2f}')
print(f'rouge_L:{CF_rouge/length:.2f}')
print(f'pos-Macro-F1-14:{CF_pos["pos-Macro-F1-14"]/length:.2f},pos-Micro-F1-14:{CF_pos["pos-Micro-F1-14"]/length:.2f},pos-Macro-F1-5:{CF_pos["pos-Macro-F1-5"]/length:.2f},pos-Micro-F1-5:{CF_pos["pos-Micro-F1-5"]/length:.2f}')
print(f'neg-Macro-F1-14:{CF_neg["neg-Macro-F1-14"]/length:.2f},neg-Micro-F1-14:{CF_neg["neg-Micro-F1-14"]/length:.2f},neg-Macro-F1-5:{CF_neg["neg-Macro-F1-5"]/length:.2f},neg-Micro-F1-5:{CF_neg["neg-Micro-F1-5"]/length:.2f}')
print(f'pos-Macro-F1-14:{CF_pos["pos-Macro-F1-14"]/total_pos["pos-Macro-F1-14"]:.2f},pos-Micro-F1-14:{CF_pos["pos-Micro-F1-14"]/total_pos["pos-Micro-F1-14"]:.2f},pos-Macro-F1-5:{CF_pos["pos-Macro-F1-5"]/total_pos["pos-Macro-F1-5"]:.2f},pos-Micro-F1-5:{CF_pos["pos-Micro-F1-5"]/total_pos["pos-Micro-F1-5"]:.2f}')
print(f'neg-Macro-F1-14:{CF_neg["neg-Macro-F1-14"]/total_neg["neg-Macro-F1-14"]:.2f},neg-Micro-F1-14:{CF_neg["neg-Micro-F1-14"]/total_neg["neg-Micro-F1-14"]:.2f},neg-Macro-F1-5:{CF_neg["neg-Macro-F1-5"]/total_neg["neg-Macro-F1-5"]:.2f},neg-Micro-F1-5:{CF_neg["neg-Micro-F1-5"]/total_neg["neg-Micro-F1-5"]:.2f}')

print("CFCL")
print(f'BLEU-1:{CFCL_bleu["BLEU-1"]/length:.2f}, BLEU-2: {CFCL_bleu["BLEU-2"]/length:.2f}, BLEU-4: {CFCL_bleu["BLEU-4"]/length:.2f}')
print(f'rouge_L:{CFCL_rouge/length:.2f}')
print(f'pos-Macro-F1-14:{CFCL_pos["pos-Macro-F1-14"]/length:.2f},pos-Micro-F1-14:{CFCL_pos["pos-Micro-F1-14"]/length:.2f},pos-Macro-F1-5:{CFCL_pos["pos-Macro-F1-5"]/length:.2f},pos-Micro-F1-5:{CFCL_pos["pos-Micro-F1-5"]/length:.2f}')
print(f'neg-Macro-F1-14:{CFCL_neg["neg-Macro-F1-14"]/length:.2f},neg-Micro-F1-14:{CFCL_neg["neg-Micro-F1-14"]/length:.2f},neg-Macro-F1-5:{CFCL_neg["neg-Macro-F1-5"]/length:.2f},neg-Micro-F1-5:{CFCL_neg["neg-Micro-F1-5"]/length:.2f}')
print(f'pos-Macro-F1-14:{CFCL_pos["pos-Macro-F1-14"]/total_pos["pos-Macro-F1-14"]:.2f},pos-Micro-F1-14:{CFCL_pos["pos-Micro-F1-14"]/total_pos["pos-Micro-F1-14"]:.2f},pos-Macro-F1-5:{CFCL_pos["pos-Macro-F1-5"]/total_pos["pos-Macro-F1-5"]:.2f},pos-Micro-F1-5:{CFCL_pos["pos-Micro-F1-5"]/total_pos["pos-Micro-F1-5"]:.2f}')
print(f'neg-Macro-F1-14:{CFCL_neg["neg-Macro-F1-14"]/total_neg["neg-Macro-F1-14"]:.2f},neg-Micro-F1-14:{CFCL_neg["neg-Micro-F1-14"]/total_neg["neg-Micro-F1-14"]:.2f},neg-Macro-F1-5:{CFCL_neg["neg-Macro-F1-5"]/total_neg["neg-Macro-F1-5"]:.2f},neg-Micro-F1-5:{CFCL_neg["neg-Micro-F1-5"]/total_neg["neg-Micro-F1-5"]:.2f}')


print("CFPF")
print(f'BLEU-1:{CFPF_bleu["BLEU-1"]/length:.2f}, BLEU-2: {CFPF_bleu["BLEU-2"]/length:.2f}, BLEU-4: {CFPF_bleu["BLEU-4"]/length:.2f}')
print(f'rouge_L:{CFPF_rouge/length:.2f}')
print(f'pos-Macro-F1-14:{CFPF_pos["pos-Macro-F1-14"]/length:.2f},pos-Micro-F1-14:{CFPF_pos["pos-Micro-F1-14"]/length:.2f},pos-Macro-F1-5:{CFPF_pos["pos-Macro-F1-5"]/length:.2f},pos-Micro-F1-5:{CFPF_pos["pos-Micro-F1-5"]/length:.2f}')
print(f'neg-Macro-F1-14:{CFPF_neg["neg-Macro-F1-14"]/length:.2f},neg-Micro-F1-14:{CFPF_neg["neg-Micro-F1-14"]/length:.2f},neg-Macro-F1-5:{CFPF_neg["neg-Macro-F1-5"]/length:.2f},neg-Micro-F1-5:{CFPF_neg["neg-Micro-F1-5"]/length:.2f}')
print(f'pos-Macro-F1-14:{CFPF_pos["pos-Macro-F1-14"]/total_pos["pos-Macro-F1-14"]:.2f},pos-Micro-F1-14:{CFPF_pos["pos-Micro-F1-14"]/total_pos["pos-Micro-F1-14"]:.2f},pos-Macro-F1-5:{CFPF_pos["pos-Macro-F1-5"]/total_pos["pos-Macro-F1-5"]:.2f},pos-Micro-F1-5:{CFPF_pos["pos-Micro-F1-5"]/total_pos["pos-Micro-F1-5"]:.2f}')
print(f'neg-Macro-F1-14:{CFPF_neg["neg-Macro-F1-14"]/total_neg["neg-Macro-F1-14"]:.2f},neg-Micro-F1-14:{CFPF_neg["neg-Micro-F1-14"]/total_neg["neg-Micro-F1-14"]:.2f},neg-Macro-F1-5:{CFPF_neg["neg-Macro-F1-5"]/total_neg["neg-Macro-F1-5"]:.2f},neg-Micro-F1-5:{CFPF_neg["neg-Micro-F1-5"]/total_neg["neg-Micro-F1-5"]:.2f}')

print("all")
print(f'BLEU-1:{all_bleu["BLEU-1"]/length:.2f}, BLEU-2: {all_bleu["BLEU-2"]/length:.2f}, BLEU-4: {all_bleu["BLEU-4"]/length:.2f}')
print(f'rouge_L:{all_rouge/length:.2f}')
print(f'pos-Macro-F1-14:{all_pos["pos-Macro-F1-14"]/length:.2f},pos-Micro-F1-14:{all_pos["pos-Micro-F1-14"]/length:.2f},pos-Macro-F1-5:{all_pos["pos-Macro-F1-5"]/length:.2f},pos-Micro-F1-5:{all_pos["pos-Micro-F1-5"]/length:.2f}')
print(f'neg-Macro-F1-14:{all_neg["neg-Macro-F1-14"]/length:.2f},neg-Micro-F1-14:{all_neg["neg-Micro-F1-14"]/length:.2f},neg-Macro-F1-5:{all_neg["neg-Macro-F1-5"]/length:.2f},neg-Micro-F1-5:{all_neg["neg-Micro-F1-5"]/length:.2f}')
print(f'pos-Macro-F1-14:{all_pos["pos-Macro-F1-14"]/total_pos["pos-Macro-F1-14"]:.2f},pos-Micro-F1-14:{all_pos["pos-Micro-F1-14"]/total_pos["pos-Micro-F1-14"]:.2f},pos-Macro-F1-5:{all_pos["pos-Macro-F1-5"]/total_pos["pos-Macro-F1-5"]:.2f},pos-Micro-F1-5:{all_pos["pos-Micro-F1-5"]/total_pos["pos-Micro-F1-5"]:.2f}')
print(f'neg-Macro-F1-14:{all_neg["neg-Macro-F1-14"]/total_neg["neg-Macro-F1-14"]:.2f},neg-Micro-F1-14:{all_neg["neg-Micro-F1-14"]/total_neg["neg-Micro-F1-14"]:.2f},neg-Macro-F1-5:{all_neg["neg-Macro-F1-5"]/total_neg["neg-Macro-F1-5"]:.2f},neg-Micro-F1-5:{all_neg["neg-Micro-F1-5"]/total_neg["neg-Micro-F1-5"]:.2f}')


### 1. 计算标准差和基尼系数的函数
def calc_gini(array):
    """计算基尼系数"""
    array = np.array(array)
    if np.amin(array) < 0:
        array -= np.amin(array)  # 保证非负
    array += 1e-8  # 防止为0
    array = np.sort(array)
    n = array.shape[0]
    index = np.arange(1, n+1)
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))

def print_metric_stats(metric_name, scores, k=1.0):
    SMA = np.mean(scores)
    std = np.std(scores)
    gini = calc_gini(scores)
    final_score_std = SMA - k * std
    final_score_gini = SMA * (1 - gini)
    # print(f"{metric_name} 四子集分数: {scores}")
    # print(f"{metric_name} 宏平均: {SMA:.4f}, 标准差: {std:.4f}, 基尼系数: {gini:.4f}")
    # print(f"{metric_name} 标准差综合指标: {final_score_std:.2f}")
    print(f"{metric_name} 基尼系数综合指标: {final_score_gini:.2f}")
    # print(f"{metric_name} 标准差: {std:.4f}")
    print(f"{metric_name} 基尼系数: {gini:.4f}")

### 2. 在输出各子集分数后，添加如下代码（以F1为例，其他指标同理）


# 假设你已经有四个子集的分数
'''
When use Rare subgroup ,only need two like:
f1_14_pos_Macro_scores = [
    CF_pos["pos-Macro-F1-14"]/length,
    CFCL_pos["pos-Macro-F1-14"]/length,
]
'''
# 例如
f1_14_pos_Macro_scores = [
    CF_pos["pos-Macro-F1-14"]/length,
    CFCL_pos["pos-Macro-F1-14"]/length,
    CFPF_pos["pos-Macro-F1-14"]/length,
    all_pos["pos-Macro-F1-14"]/length
]

f1_14_pos_Micro_scores = [
    CF_pos["pos-Micro-F1-14"]/length,
    CFCL_pos["pos-Micro-F1-14"]/length,
    CFPF_pos["pos-Micro-F1-14"]/length,
    all_pos["pos-Micro-F1-14"]/length
]

f1_5_pos_Macro_scores = [
    CF_pos["pos-Macro-F1-5"]/length,
    CFCL_pos["pos-Macro-F1-5"]/length,
    CFPF_pos["pos-Macro-F1-5"]/length,
    all_pos["pos-Macro-F1-5"]/length
]

f1_5_pos_Micro_scores = [
    CF_pos["pos-Micro-F1-5"]/length,
    CFCL_pos["pos-Micro-F1-5"]/length,
    CFPF_pos["pos-Micro-F1-5"]/length,
    all_pos["pos-Micro-F1-5"]/length
]

f1_14_neg_Macro_scores = [
    CF_neg["neg-Macro-F1-14"]/length,
    CFCL_neg["neg-Macro-F1-14"]/length,
    CFPF_neg["neg-Macro-F1-14"]/length,
    all_neg["neg-Macro-F1-14"]/length
]

f1_14_neg_Micro_scores = [
    CF_neg["neg-Micro-F1-14"]/length,
    CFCL_neg["neg-Micro-F1-14"]/length,
    CFPF_neg["neg-Micro-F1-14"]/length,
    all_neg["neg-Micro-F1-14"]/length
]

f1_5_neg_Macro_scores = [
    CF_neg["neg-Macro-F1-5"]/length,
    CFCL_neg["neg-Macro-F1-5"]/length,
    CFPF_neg["neg-Macro-F1-5"]/length,
    all_neg["neg-Macro-F1-5"]/length
]

f1_5_neg_Micro_scores = [
    CF_neg["neg-Micro-F1-5"]/length,
    CFCL_neg["neg-Micro-F1-5"]/length,
    CFPF_neg["neg-Micro-F1-5"]/length,
    all_neg["neg-Micro-F1-5"]/length
]

BLEU_1_scores = [
    CF_bleu["BLEU-1"]/length,
    CFCL_bleu["BLEU-1"]/length,
    CFPF_bleu["BLEU-1"]/length,
    all_bleu["BLEU-1"]/length
]

BLEU_2_scores = [
    CF_bleu["BLEU-2"]/length,
    CFCL_bleu["BLEU-2"]/length,
    CFPF_bleu["BLEU-2"]/length,
    all_bleu["BLEU-2"]/length
]

BLEU_4_scores = [
    CF_bleu["BLEU-4"]/length,
    CFCL_bleu["BLEU-4"]/length,
    CFPF_bleu["BLEU-4"]/length,
    all_bleu["BLEU-4"]/length
]

ROUGE_L_scores = [
    CF_rouge/length,
    CFCL_rouge/length,
    CFPF_rouge/length,
    all_rouge/length
]

all_info = {
    "F1-14-pos-Macro": f1_14_pos_Macro_scores,
    "F1-14-pos-Micro": f1_14_pos_Micro_scores,
    "F1-5-pos-Macro": f1_5_pos_Macro_scores,
    "F1-5-pos-Micro": f1_5_pos_Micro_scores,
    "F1-14-neg-Macro": f1_14_neg_Macro_scores,
    "F1-14-neg-Micro": f1_14_neg_Micro_scores,
    "F1-5-neg-Macro": f1_5_neg_Macro_scores,
    "F1-5-neg-Micro": f1_5_neg_Micro_scores,
    "BLEU-1": BLEU_1_scores,
    "BLEU-2": BLEU_2_scores,
    "BLEU-4": BLEU_4_scores,
    "ROUGE-L": ROUGE_L_scores,
}

for k,v in all_info.items():
    print_metric_stats(k, v)




