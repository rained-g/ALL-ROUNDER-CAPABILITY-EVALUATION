from operator import gt
from BLEU4.bleu4_nltk import bleu_score, simple_tokenizer
from ROUGEL.rougel import rouge_l
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk

# 下载必要的 NLTK 资源
def download_nltk_resources():
    """下载必要的 NLTK 资源"""
    resources = ['punkt_tab', 'punkt', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            if 'punkt' in resource:
                nltk.data.find(f'tokenizers/{resource}')
            else:
                nltk.data.find(resource)
            print(f"NLTK {resource} 资源已存在")
        except LookupError:
            print(f"正在下载 NLTK {resource} 资源...")
            try:
                nltk.download(resource)
                print(f"NLTK {resource} 资源下载完成！")
            except Exception as e:
                print(f"NLTK {resource} 资源下载失败: {e}")
                if resource == 'wordnet':
                    print("警告: wordnet 下载失败，METEOR 评分可能无法正常工作")

# 在导入后立即下载资源
# download_nltk_resources()

#原始报告的路径-mimic
origin_mimic_test = r"./origin_report/mimic/Test.csv"
origin_openi_test = r"./origin_report/openi/Test.csv"
mimic_labeled_test = r"./results/CheXbert/labeled_mimic_test_reports.csv"
openi_labeled_test = r"./results/CheXbert/origin/openi/labeled_openi_test_reports.csv"

#原始报告的路径-complete
origin_complete_maria1_test = r'./origin_report/complete/maria1/test_data.csv'
origin_complete_maria2_test = r'./origin_report/complete/maria2/test_data.csv'
origin_complete_maria2_test_test = r'../datasets/MIMIC-complete/processed_data_MARIA2/test_data_new.csv'
maria1_labeled_test = r'./results/CheXbert/origin/mimic/maria1/labeled_reports.csv'
maria2_labeled_test = r'./results/CheXbert/origin/mimic/maria2/labeled_reports.csv'
maria2_labeled_test_new = r'./results/CheXbert/mimic/complete/newtry/origin.csv'

#原始报告的路径-new-try1
origin_all_test_new = r'./origin_report/complete/all_test_new/test_data_new.csv'
all_test_labeled_new = r'./results/CheXbert/newtry1/original/all_test_new.csv'

origin_all_test = r'./origin_report/complete/Maira2_2461/test_data.csv'
all_test_labeled = r'./results/CheXbert/newtry1/original/test_data.csv'
#生成报告的路径
# mimic_test_G = r"./results/CheXbert/labeled_mimic_test_img_reports.csv"
# openi_test_G = r"./results/CheXbert/labeled_openi_test_img_reports.csv"
# openi_test_5000_G = r'./results/CheXbert/labeled_5000_openi_test_img_reports.csv'
# mimic_test_5000_G = r'./results/CheXbert/labeled_5000_mimic_test_img_reports.csv'
# openi_MARIA_test_G = r'./results/CheXbert/labeled_MIRIA-2openi_test_img_reports.csv'
# mimic_MARIA_test_G = r'./results/CheXbert/labeled_MIRIA2mimic_test_img_reports.csv'
# complete_temp1 = r'./results/CheXbert/labeled_complete_temp1.csv'
# complete_openi_test = r'./results/CheXbert/labeled_complete_openi_test_reports.csv'
img5k_maria1_test_G = r'./results/CheXbert/mimic/complete/original/mimic_maira1_5000_test_img_reports.csv'
img5k_maria2_test_G = r'./results/CheXbert/mimic/complete/original/mimic_maria2_5000_test_img_reports.csv'
img5k_openi_test_G = r'./results/CheXbert/openi/img/labeled_5000_openi_test_img_reports.csv'

img6w_maria1_test_G = r'./results/CheXbert/mimic/complete/original/mimic_maria1_6w_test_img_reports.csv'
img6w_maria2_test_G = r'./results/CheXbert/mimic/complete/original/mimic_maria2_6w_test_img_reports.csv'
img6w_openi_test_G = r'./results/CheXbert/openi/img/labeled_openi_test_img_reports.csv'

ind_maria1_test_G = r'./results/CheXbert/mimic/complete/maria1/mimic_maria1_test_report.csv'
ind_maria2_test_G = r'./results/CheXbert/mimic/complete/maria1/mimic_maria2_test_report.csv'
ind_openi_test_G = r'./results/CheXbert/openi/maria1/labeled_complete_openi_test_reports.csv'

all_maria2_test_G = r'./results/CheXbert/mimic/complete/maria2/mimic_maria2_test_report.csv'
all_maria2_test_G2 = r'./results/CheXbert/mimic/complete/maria2/mimic_maria2_test_report2.csv'
all_maria1_test_G = r'./results/CheXbert/mimic/complete/maria2/mimic_maria1_test_report.csv'
all_openi_test_G = r'./results/CheXbert/openi/maria2/openi_test_report.csv'

MARIA2_maria2_test_G = r'./results/CheXbert/mimic/MARIA2/complete/maria2/MARIA-2-complete_mimic-report.csv'
MARIA2_maria1_test_G = r'./results/CheXbert/mimic/MARIA2/complete/maria1/MARIA-2-complete_mimic-report.csv'
MARIA2_openi_test_G = r'./results/CheXbert/openi/MARIA2/labeled_MIRIA-2openi_test_img_reports.csv'
#new-try
'''
newtry1_maria2_test_G = r'./results/CheXbert/mimic/complete/newtry/fuse1.csv'
newtry2_maria2_test_G = r'./results/CheXbert/mimic/complete/newtry/fuse2.csv'
newtry3_maria2_test_G = r'./results/CheXbert/mimic/complete/newtry/fuse3.csv'
newtry2_4096_maria2_test_G = r'./results/CheXbert/mimic/complete/newtry/fuse2_4096.csv'
newtry3_1_maria2_test_G = r'./results/CheXbert/mimic/complete/newtry/fuse3_1.csv'
newtry3_openi_test_G = r'./results/CheXbert/mimic/complete/newtry/fuse3_openi.csv'
newtry4_1_maria2_test_new_G = r'./results/CheXbert/mimic/complete/newtry/fuse4_1.csv'    #比3差
newtryDL4_2_maria2_test_new_G = r'./results/CheXbert/mimic/complete/newtry/fuse4_2.csv'
newtry1_test_maria2_test_new_G = r'./results/CheXbert/mimic/complete/newtry/fuse1_test.csv'
newtry3_2_maria2_test_G = r'./results/CheXbert/mimic/complete/newtry/fuse3_2.csv' 
newtry3_3_maria2_test_G = r'./results/CheXbert/mimic/complete/newtry/fuse3_3.csv'    #比3差
newtry3_4_maria2_test_G = r'./results/CheXbert/mimic/complete/newtry/fuse3_4.csv'     #3中最优
newtryVicuna1_2maria2_test_G = r'./results/CheXbert/mimic/complete/newtry/fuseVicuna1_2.csv'
newtry4_2_maria2_test_new_G = r'./results/CheXbert/mimic/complete/newtry/fuseVicuna4_2.csv'    #比3差
newtryV1_1_maria2_test_new_G = r'./results/CheXbert/mimic/complete/newtry/fuseVicuna1_1.csv'    #比3差
newtryV4_2s_maria2_test_new_G = r'./results/CheXbert/mimic/complete/newtry/fuseVicuna4_2small.csv'
fuse1_Vicuna_small_slow_noatt_nostandard = r'./results/CheXbert/mimic/complete/newtry/fuse1_Vicuna_small_slow_noatt_nostandard.csv'
fuse1_vicuna_small_slow_att_standard = r'./results/CheXbert/mimic/complete/newtry/fuse1_vicuna_small_slow_att_standard.csv'
fuse1_Vicuna_small_slow_att_nostandard = r'./results/CheXbert/mimic/complete/newtry/fuse1_Vicuna_small_slow_att_nostandard.csv.csv'
fuse4_2_vicuna_small_high_noatt_nostandard = r'./results/CheXbert/mimic/complete/newtry/fuse4-2_vicuna_small_high_noatt_nostandard.csv'
'''
#new-try1-old
'''
new_fuse1_vicuna_small_high_att_standard = r'./results/CheXbert/newtry1/Vicuna/new_fuse1_vicuna_small_high_att_standard.csv'
new_fuse1_vicuna_small_high_noatt_standard = r'./results/CheXbert/newtry1/Vicuna/new_fuse1_vicuna_small_high_noatt_standard.csv'
new_fuse3_vicuna_small_high_att_standard = r'./results/CheXbert/newtry1/Vicuna/new_fuse3_vicuna_small_high_att_standard.csv'
new_fuse3_0_vicuna_small_high_att_standard = r'./results/CheXbert/newtry1/Vicuna/new_fuse3_0_vicuna_small_high_att_standard.csv'
new_fuse3_vicuna_small_high_noatt_standard = r'./results/CheXbert/newtry1/Vicuna/new_fuse3_vicuna_small_high_noatt_standard.csv'
new_fuse3_1_vicuna_small_high_att_standard = r'./results/CheXbert/newtry1/Vicuna/new_fuse3_1_vicuna_small_high_att_standard.csv'
new_fuse3_1_vicuna_small_high_noatt_standard = r'./results/CheXbert/newtry1/Vicuna/new_fuse3_1_vicuna_small_high_noatt_standard.csv'
new_fuse3_01_vicuna_small_high_att_standard = r'./results/CheXbert/newtry1/Vicuna/new_fuse3_01_vicuna_small_high_att_standard.csv'
new_fuse3_DL_small_high_att_standard = r'./results/CheXbert/newtry1/Vicuna/new_fuse3_DL_small_high_att_standard.csv'
new_fuse3_DL_small_high_noatt_standard = r'./results/CheXbert/newtry1/Vicuna/new_fuse3_DL_small_high_noatt_standard.csv'
new_fuse1_maira1_vicuna_small_high_att_standard = r'./results/CheXbert/newtry1/Vicuna/new_fuse1_maria1_4096_vicuna_small_high_att_standard.csv'
new_fuse1_maira1_vicuna_small_high_noatt_standard = r'./results/CheXbert/newtry1/Vicuna/new_fuse1_maria1_4096_vicuna_small_high_noatt_standard.csv'
MARIA2_all_mimic_report = r'./results/CheXbert/newtry1/MAIRA2/MARIA2_all_mimic_report.csv'
'''
#new-try1
new_fuse3_Vicuna_small_high_att_standard1 = r'./results/CheXbert/newtry1/Vicuna/new_fuse3_Vicuna_small_high_att_standard1.csv'
new_fuse3_Vicuna_small_high_att_standard2 = r'./results/CheXbert/newtry1/Vicuna/new_fuse3_Vicuna_small_high_att_standard2.csv'
new_fuse3_Vicuna_small_high_att_standard3 = r'./results/CheXbert/newtry1/Vicuna/new_fuse3_Vicuna_small_high_att_standard3.csv'
new_fuse3_Vicuna_small_high_noatt_standard1 = r'./results/CheXbert/newtry1/Vicuna/new_fuse3_Vicuna_small_high_noatt_standard1.csv'
new_fuse3_Vicuna_small_high_noatt_standard2 = r'./results/CheXbert/newtry1/Vicuna/new_fuse3_Vicuna_small_high_noatt_standard2.csv'
new_fuse3_Vicuna_small_high_noatt_standard3 = r'./results/CheXbert/newtry1/Vicuna/new_fuse3_Vicuna_small_high_noatt_standard3.csv'

new_fuse3_Qwen_small_high_att_standard_1 = r'./results/CheXbert/newtry1/Qwen/new_fuse3_Qwen_small_high_att_standard_1.csv'
new_fuse3_Qwen_small_high_att_standard_2 = r'./results/CheXbert/newtry1/Qwen/new_fuse3_Qwen_small_high_att_standard_2.csv'
new_fuse3_Qwen_small_high_att_standard_3 = r'./results/CheXbert/newtry1/Qwen/new_fuse3_Qwen_small_high_att_standard_3.csv'
new_fuse3_Qwen_small_high_noatt_standardi = r'./results/CheXbert/newtry1/Qwen/new_fuse3_Qwen_small_high_noatt_standard{i}.csv'
new_fuse3_Qwen_small_high_noatt_standard1 = r'./results/CheXbert/newtry1/Qwen/new_fuse3_Qwen_small_high_noatt_standard1.csv'
new_fuse3_Qwen_small_high_noatt_standard2 = r'./results/CheXbert/newtry1/Qwen/new_fuse3_Qwen_small_high_noatt_standard2.csv'
new_fuse3_Qwen_small_high_noatt_standard3 = r'./results/CheXbert/newtry1/Qwen/new_fuse3_Qwen_small_high_noatt_standard3.csv'

new_fuse1_Vicuna_small_high_att_standard1 = r'./results/CheXbert/newtry1/Vicuna/new_fuse1_Vicuna_small_high_att_standard1.csv'
new_fuse1_Vicuna_small_high_att_standard2 = r'./results/CheXbert/newtry1/Vicuna/new_fuse1_Vicuna_small_high_att_standard2.csv'
new_fuse1_Vicuna_small_high_att_standard3 = r'./results/CheXbert/newtry1/Vicuna/new_fuse1_Vicuna_small_high_att_standard3.csv'
new_fuse1_Vicuna_small_high_noatt_standard1 = r'./results/CheXbert/newtry1/Vicuna/new_fuse1_Vicuna_small_high_noatt_standard1.csv'
new_fuse1_Vicuna_small_high_noatt_standard2 = r'./results/CheXbert/newtry1/Vicuna/new_fuse1_Vicuna_small_high_noatt_standard2.csv'
new_fuse1_Vicuna_small_high_noatt_standard3 = r'./results/CheXbert/newtry1/Vicuna/new_fuse1_Vicuna_small_high_noatt_standard3.csv'

new_fuse1_Qwen_small_high_att_standard_1 = r'./results/CheXbert/newtry1/Qwen/new_fuse1_Qwen_small_high_att_standard_1.csv'
new_fuse1_Qwen_small_high_att_standard_2 = r'./results/CheXbert/newtry1/Qwen/new_fuse1_Qwen_small_high_att_standard_2.csv'
new_fuse1_Qwen_small_high_att_standard_3 = r'./results/CheXbert/newtry1/Qwen/new_fuse1_Qwen_small_high_att_standard_3.csv'
new_fuse1_Qwen_small_high_noatt_standard1 = r'./results/CheXbert/newtry1/Qwen/new_fuse1_Qwen_small_high_noatt_standard1.csv'
new_fuse1_Qwen_small_high_noatt_standard2 = r'./results/CheXbert/newtry1/Qwen/new_fuse1_Qwen_small_high_noatt_standard2.csv'
new_fuse1_Qwen_small_high_noatt_standard3 = r'./results/CheXbert/newtry1/Qwen/new_fuse1_Qwen_small_high_noatt_standard3.csv'

new_fuse3_DL_small_high_att_standard_1 = r'./results/CheXbert/newtry1/DL/new_fuse3_DL_small_high_att_standard_1.csv'
new_fuse3_DL_small_high_att_standard_4 = r'./results/CheXbert/newtry1/DL/new_fuse3_DL_small_high_att_standard_4.csv'
new_fuse3_DL_small_high_att_standard_3 = r'./results/CheXbert/newtry1/DL/new_fuse3_DL_small_high_att_standard_3.csv'

new_fuse3_avg_Vicuna_small_high_att_standard1 = r'./results/CheXbert/newtry1/Vicuna/new_fuse3_avg_Vicuna_small_high_att_standard1.csv'

new_fuse3_DL_small_high_att_standard_old_avg = r'./results/CheXbert/newtry1/DL/new_fuse3_DL_small_high_att_standard_old.csv'
new_fuse3_DL_small_high_noatt_standard_old_avg = r'./results/CheXbert/newtry1/DL/new_fuse3_DL_small_high_noatt_standard_old.csv'
new_fuse3_DL_small_high_noatt_standard_old_long1 = r'./results/CheXbert/newtry1/DL/new_fuse3_DL_small_high_noatt_standard_old_long1.csv'
new_fuse3_DL_small_high_noatt_standard_old_long2 = r'./results/CheXbert/newtry1/DL/new_fuse3_DL_small_high_noatt_standard_old_long2.csv'

old_1712_att = r'./results/CheXbert/newtry1/Vicuna/old_1712_att_standard1.csv'
old_1712_noatt = r'./results/CheXbert/newtry1/Vicuna/old_1712_noatt_standard1.csv'
old_4096_att = r'./results/CheXbert/newtry1/Vicuna/old_4096_att_standard1.csv'


new_fuse3_1_vicuna_small_high_att_standard = r'./results/CheXbert/newtry1/Vicuna/past/new_fuse3_1_vicuna_small_high_att_standard.csv'
new_fuse3_1_vicuna_small_high_noatt_standard = r'./results/CheXbert/newtry1/Vicuna/past/new_fuse3_1_vicuna_small_high_noatt_standard.csv'
new_fuse1_maira1_vicuna_small_high_att_standard = r'./results/CheXbert/newtry1/Vicuna/past/new_fuse1_maria1_4096_vicuna_small_high_att_standard.csv'
new_fuse1_maira1_vicuna_small_high_noatt_standard = r'./results/CheXbert/newtry1/Vicuna/past/new_fuse1_maria1_4096_vicuna_small_high_noatt_standard.csv'


small_maira2_att_1 = r'./results/CheXbert/newtry1/small_maira2/small_maira2_high_att_standard1.csv'
small_maira2_att_2 = r'./results/CheXbert/newtry1/small_maira2/small_maira2_high_att_standard2.csv'
small_maira2_noatt_1 = r'./results/CheXbert/newtry1/small_maira2/small_maira2_high_noatt_standard1.csv'

stage3_fuse3_3w_att_1 = r'./results/CheXbert/newtry1/stage/Vicuna_Fuse3_stage3_3w_att_standard1.csv'
stage3_fuse3_3w_noatt_1 = r'./results/CheXbert/newtry1/stage/Vicuna_Fuse3_stage3_3w_noatt_standard1.csv'

stage3_Newfuse3_3w_att_1 = r'./results/CheXbert/newtry1/stage/Vicuna_NewFuse3_stage3_3w_att_standard1.csv'

stage3_fuse3noself_3w_att_1 = r'./results/CheXbert/newtry1/stage/Vicuna_Fuse3noself_stage3_3w_att_standard1.csv'
stage3_fuse3noself_3w_att_2 = r'./results/CheXbert/newtry1/stage/Vicuna_Fuse3noself_stage3_3w_att_standard2.csv'

stage3_fuse3noself_3w_att_new1 = r'./results/CheXbert/newtry1/stage/Vicuna_Fuse3noself_stage3_3w_att_standard_new1.csv'

stage3_Fuse3old_3w_att_1 = r'./results/CheXbert/newtry1/stage/Vicuna_Fuse3old_stage3_3w_att_standard1.csv'
stage3_Fuse3old_3w_att_2 = r'./results/CheXbert/newtry1/stage/Vicuna_Fuse3old_stage3_3w_att_standard2.csv'

stage3_single_6w_att_1 = r'./results/CheXbert/newtry1/stage/Vicuna_single_stage3_6w_att_standard1.csv'

new_fuselast_att_2 = r'./results/CheXbert/newtry1/Vicuna/Vicuna_Fuse3last_35k_att_standard2.csv'

Fuse3_new1 = r'./results/CheXbert/newtry1/Vicuna/Vicuna_Fuse3_new1_3w_att_standard1.csv'

data_5k = r'./results/CheXbert/newtry1/Vicuna/Vicuna_Fuse3_new1_5k_att_standard1.csv'
data_grad = r'./results/CheXbert/newtry1/Vicuna/Vicuna_Fuse3_new1_5k_grad_att_standard1.csv'
data_5k1 = r'./results/CheXbert/newtry1/Vicuna/Vicuna_Fuse3_new1_5k1_att_standard2.csv'
data_5k1_noatt = r'./results/CheXbert/newtry1/Vicuna/Vicuna_Fuse3_new1_5k1_noatt_standard2.csv'
data_15k4 = r'./results/CheXbert/newtry1/Fuse3_update/ver2/Fuse3_update_15K4_att_standard2.csv'

data_5k_maira2 = r'./results/CheXbert/newtry1/Vicuna/small_maira2_5K_high_att_standard1.csv'
data_15k_maira2 = r'./results/CheXbert/newtry1/Vicuna/small_maira2_15K_high_att_standard1.csv'


# df = pd.read_csv(fuse1_Vicuna_small_slow_att_nostandard)
# print(len(df))
#数据处理函数定义
def dataframe_data(origin_path,origin_labeled_path,G_labeled_path):
    def determine_type(row):
        has_cl = pd.notna(row.get(columns_name3))
        has_pf = pd.notna(row.get(columns_name4))
        if has_cl and has_pf:
            return 'all'
        elif has_cl:
            return 'CFCL'
        elif has_pf:
            return 'CFPF'
        else:
            return 'CF'
    f = 0
    df = pd.read_csv(origin_path).copy()
    df_labeled = pd.read_csv(origin_labeled_path)
    df1 = pd.read_csv(G_labeled_path)

    columns_name1 = 'study_id'
    columns_name2 = 'Report Impression'
    columns_name3 = 'Current_lateral_dicom_id'
    columns_name4 = 'Prior_frontal_dicom_id'
    columns_name5 = 'type'

    if (columns_name1 in df1.columns) and (columns_name1 in df_labeled.columns) :
        pass
    else:
        df1['study_id'] = df['study_id']
        df_labeled['study_id'] = df['study_id']
        f = 1
    
    if (columns_name2 in df1.columns) or (columns_name2 in df_labeled.columns):
        df1.rename(columns={columns_name2: 'report'}, inplace=True)
        df_labeled.rename(columns={columns_name2: 'report'}, inplace=True)
        f = 1
    else:
        pass
    if (columns_name5 in df1.columns) and (columns_name5 in df_labeled.columns):
        pass
    else:
        df['type'] = df.apply(determine_type, axis=1)
        # 将 type 列添加到 df_labeled 和 df1 中
        df_labeled['type'] = df['type']
        df1['type'] = df['type']
        f=1
    if f : 
        df1.to_csv(G_labeled_path, index=False)
        df_labeled.to_csv(origin_labeled_path, index=False)
        

    text = df_labeled['report'].tolist()
    G_text = df1['report'].tolist()

    if len(G_text)!=len(text):
        limited = min(len(G_text),len(text))
        return df_labeled[:limited],df1[:limited],text[:limited],G_text[:limited]
    return df_labeled,df1,text,G_text
#计算F1分数
def cal_pos_F1(df1,df2):
    # print(df1.head())
    df_o = df1.replace(-1,1,inplace=False)
    # print(df_o.head())
    df_o.fillna(0, inplace=True)
    df_G = df2.replace(-1,1,inplace=False)
    df_G.fillna(0, inplace=True)

    columns_of_interest = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

    gt_labels_14 = df_o.iloc[:, 1:15].values  # 取第2列到最后一列
    pred_labels_14 = df_G.iloc[:, 1:15].values  # 取第2列到最后一列
    # print(gt_labels_14[0])
    gt_labels_5 = df_o[columns_of_interest].values  # ground truth labels
    pred_labels_5 = df_G[columns_of_interest].values

    # print(f"gt_labels 长度: {gt_labels_14.shape[0]}, pred_labels 长度: {pred_labels_14.shape[0]}")
    # print(type(gt_labels_14))
    # print(len(gt_labels_14.flatten()))
    # micro_f1_14 = f1_score(gt_labels_14.flatten(), pred_labels_14.flatten(), average='micro')
    # macro_f1_14 = f1_score(gt_labels_14.flatten(), pred_labels_14.flatten(), average='macro')

    # micro_f1_5 = f1_score(gt_labels_5.flatten(), pred_labels_5.flatten(), average='micro')
    # macro_f1_5 = f1_score(gt_labels_5.flatten(), pred_labels_5.flatten(), average='macro')
    micro_f1_14 = f1_score(list(gt_labels_14), list(pred_labels_14), average='micro')
    macro_f1_14 = f1_score(list(gt_labels_14), list(pred_labels_14), average='macro')

    micro_f1_5 = f1_score(list(gt_labels_5), list(pred_labels_5), average='micro')
    macro_f1_5 = f1_score(list(gt_labels_5), list(pred_labels_5), average='macro')

    print(f"pos-Macro-F1-14: {macro_f1_14*100:.2f}, pos-Micro-F1-14: {micro_f1_14*100:.2f}")
    print(f"pos-Macro-F1-5: {macro_f1_5*100:.2f}, pos-Micro-F1-5: {micro_f1_5*100:.2f}")

    result = {"pos-Macro-F1-14":macro_f1_14*100,"pos-Micro-F1-14":micro_f1_14*100,"pos-Macro-F1-5":macro_f1_5*100,"pos-Micro-F1-5":micro_f1_5*100}

    return result

def cal_neg_F1(df1,df2):

    df_o = df1.replace(-1,0,inplace=False)
    # print(df_o.head())
    df_o.fillna(0, inplace=True)
    # print(df_o.head())
    df_G = df2.replace(-1,0,inplace=False)
    df_G.fillna(0, inplace=True)

    columns_of_interest = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

    gt_labels_14 = df_o.iloc[:, 1:15].values  # 取第2列到最后一列
    pred_labels_14 = df_G.iloc[:, 1:15].values  # 取第2列到最后一列

    gt_labels_5 = df_o[columns_of_interest].values  # ground truth labels
    pred_labels_5 = df_G[columns_of_interest].values

    # print(f"gt_labels 长度: {gt_labels_14.shape[0]}, pred_labels 长度: {pred_labels_14.shape[0]}")

    # micro_f1_14 = f1_score(gt_labels_14.flatten(), pred_labels_14.flatten(), average='micro')
    # macro_f1_14 = f1_score(gt_labels_14.flatten(), pred_labels_14.flatten(), average='macro')

    # micro_f1_5 = f1_score(gt_labels_5.flatten(), pred_labels_5.flatten(), average='micro')
    # macro_f1_5 = f1_score(gt_labels_5.flatten(), pred_labels_5.flatten(), average='macro')

    micro_f1_14 = f1_score(list(gt_labels_14), list(pred_labels_14), average='micro')
    macro_f1_14 = f1_score(list(gt_labels_14), list(pred_labels_14), average='macro')

    micro_f1_5 = f1_score(list(gt_labels_5), list(pred_labels_5), average='micro')
    macro_f1_5 = f1_score(list(gt_labels_5), list(pred_labels_5), average='macro')

    print(f"neg-Macro-F1-14: {macro_f1_14*100:.2f}, neg-Micro-F1-14: {micro_f1_14*100:.2f}")
    print(f"neg-Macro-F1-5: {macro_f1_5*100:.2f}, neg-Micro-F1-5: {micro_f1_5*100:.2f}")

    result = {"neg-Macro-F1-14":macro_f1_14*100,"neg-Micro-F1-14":micro_f1_14*100,"neg-Macro-F1-5":macro_f1_5*100,"neg-Micro-F1-5":micro_f1_5*100}

    return result
#计算BLEU分数
def cal_BLEU(references, candidates):
    bleu_scores_all = []

    for reference, candidate in zip(references, candidates):
        # 计算 BLEU 分数
        scores = bleu_score(reference, candidate)
        bleu_scores_all.append(scores)

    average_bleu = {'BLEU-1': 0, 'BLEU-2': 0, 'BLEU-3': 0, 'BLEU-4': 0}

    # 遍历每个分数字典
    for scores in bleu_scores_all:
        for key in average_bleu.keys():
            average_bleu[key] += scores[key]

    # 对每个分数取平均
    for key in average_bleu.keys():
        average_bleu[key] /= len(bleu_scores_all)
        average_bleu[key] *=100
    # 打印最终平均分数
    print("Average BLEU Scores:", average_bleu)
    return average_bleu

def cal_meteor(references, candidates):
    """计算 METEOR 评分，如果 wordnet 不可用则跳过"""
    # 首先检查 wordnet 是否可用
    try:
        from nltk.corpus import wordnet
        wordnet.synsets('test')  # 测试 wordnet 是否工作
    except Exception as e:
        print(f"WordNet 不可用: {e}")
        print("跳过 METEOR 评分计算")
        return 0.0
    
    meteor_scores_all = []
    for reference, candidate in zip(references, candidates):
        try:
            # 计算 BLEU 分数
            ref_tokens = word_tokenize(reference)
            hyp_tokens = word_tokenize(candidate)

            # Meteor 要求 references 是 list of list
            meteor = meteor_score([ref_tokens], hyp_tokens)
            meteor_scores_all.append(meteor)
        except Exception as e:
            print(f"METEOR 计算错误: {e}")
            print("跳过当前样本的 METEOR 评分")
            continue

    if meteor_scores_all:
        avg_meteor = np.mean(meteor_scores_all)
        print(f"Average METEOR Score: {avg_meteor*100:.2f}")
        return avg_meteor
    else:
        print("所有样本的 METEOR 评分都失败了")
        return 0.0
'''
#原始报告数据的读取
df_origin_mimic_test = pd.read_csv(origin_mimic_test)
df_origin_openi_test = pd.read_csv(origin_openi_test)
df_mimic_test = pd.read_csv(mimic_test)
df_openi_test = pd.read_csv(openi_test)
#生成报告数据的读取
df_mimic_test_G = pd.read_csv(mimic_test_G)
df_openi_test_G = pd.read_csv(openi_test_G)
df_openi_test_5000_G = pd.read_csv(openi_test_5000_G)
df_mimic_test_5000_G = pd.read_csv(mimic_test_5000_G)
#添加唯一标识，在此之前记得将列名改为report
df_mimic_test_G['study_id'] = df_origin_mimic_test['study_id']
df_openi_test_G['study_id'] = df_origin_openi_test['study_id']
df_openi_test_5000_G['study_id'] = df_origin_openi_test['study_id']
df_mimic_test_5000_G['study_id'] = df_origin_mimic_test['study_id']
df_mimic_test['study_id'] = df_origin_mimic_test['study_id']
df_openi_test['study_id'] = df_origin_openi_test['study_id']
#保存修改
# df_mimic_test_G.to_csv(mimic_test_G)
# df_openi_test_G.to_csv(openi_test_G)
# df_openi_test_5000_G.to_csv(oprni_test_5000_G)
# df_mimic_test_5000_G.to_csv(mimic_test_5000_G)
# df_mimic_test.to_csv(mimic_test)
# df_openi_test.to_csv(openi_test)
#读取文本数据
mimic_test_text = df_mimic_test["report"]
openi_test_text = df_openi_test["report"]

mimic_test_G_text = df_mimic_test_G["report"]
openi_test_G_text = df_openi_test_G["report"]
openi_5000_test_G_text = df_openi_test_5000_G["report"]
mimic_5000_test_G_text = df_mimic_test_5000_G["report"]

openi_test_text = openi_test_text.tolist()
openi_test_G_text = openi_test_G_text.tolist()
openi_5000_test_G_text = openi_5000_test_G_text.tolist()

mimic_test_text = mimic_test_text.tolist()
mimic_test_G_text = mimic_test_G_text.tolist()
mimic_5000_test_G_text = mimic_5000_test_G_text.tolist()
'''
# df, df_G, references, candidates= dataframe_data(origin_complete_maria1_test,maria1_labeled_test,MARIA2_maria1_test_G)
df, df_G, references, candidates= dataframe_data('/mnt/sda/ymji/Report-Generation/evaluation_metric/origin_report/complete/all_test_new/test_data_new.csv','/mnt/sda/ymji/Report-Generation/evaluation_metric/results/CheXbert/newtry1/original/all_test_new.csv','/mnt/sda/ymji/Report-Generation/evaluation_metric/results/CheXbert/newtry1/MAIRA2/MARIA2_all_mimic_report.csv')
# references = openi_test_text
# candidates = openi_test_G_text
# evaluation-metric/evaluation.py
'''
for reference, candidate in zip(references, candidates):
    # 计算 BLEU 分数
    scores = bleu_score(reference, candidate)
    bleu_scores_all.append(scores)
# print(bleu_scores_all[:5])

average_bleu = {'BLEU-1': 0, 'BLEU-2': 0, 'BLEU-3': 0, 'BLEU-4': 0}

# 遍历每个分数字典
for scores in bleu_scores_all:
    for key in average_bleu.keys():
        average_bleu[key] += scores[key]

# 对每个分数取平均
for key in average_bleu.keys():
    average_bleu[key] /= len(bleu_scores_all)
    average_bleu[key] *=100
# 打印最终平均分数
print("Average BLEU Scores:", average_bleu)
'''
cal_BLEU(references, candidates)
# cal_meteor(references, candidates)
def calculate_rouge_l(reference_texts, candidate_texts):
    """
    对一组参考文本和生成文本计算 ROUGE-L 分数。

    :param reference_texts: 参考文本列表
    :param candidate_texts: 生成文本列表
    :return: Precision、Recall、F1 的平均值
    """
    precision_list, recall_list, f1_list = [], [], []
    for ref, cand in zip(reference_texts, candidate_texts):
        scores = rouge_l(ref, cand)
        precision_list.append(scores.precision)
        recall_list.append(scores.recall)
        f1_list.append(scores.fmeasure)
    
    avg_precision = sum(precision_list) / len(precision_list)
    avg_recall = sum(recall_list) / len(recall_list)
    avg_f1 = sum(f1_list) / len(f1_list)
    
    return avg_precision, avg_recall, avg_f1

precision, recall, f1 = calculate_rouge_l(references, candidates)

# 输出结果
print(f"ROUGE-L Precision: {precision*100:.2f}, ROUGE-L Recall: {recall*100:.2f}, ROUGE-L F1 Score: {f1*100:.2f}")

cal_pos_F1(df,df_G)
cal_neg_F1(df,df_G)
'''
df_mimic_test.replace(-1,0,inplace=False)
df_mimic_test.fillna(-2, inplace=True)
df_openi_test.replace(-1,0,inplace=False)
df_openi_test.fillna(-2, inplace=True)

df_mimic_test_G.replace(-1,0,inplace=False)
df_mimic_test_G.fillna(-2, inplace=True)
df_openi_test_G.replace(-1,0,inplace=False)
df_openi_test_G.fillna(-2, inplace=True)


df_mimic_test_5000_G.replace(-1,0,inplace=False)
df_mimic_test_5000_G.fillna(-2, inplace=True)
df_openi_test_5000_G.replace(-1,0,inplace=False)
df_openi_test_5000_G.fillna(-2, inplace=True)

df.replace(-1,0,inplace=False)
df.fillna(-2, inplace=True)
df_G.replace(-1,0,inplace=False)
df_G.fillna(-2, inplace=True)

columns_of_interest = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

gt_labels = df.iloc[:, 1:-1].values  # 取第2列到最后一列
pred_labels = df_G.iloc[:, 1:-1].values  # 取第2列到最后一列

# gt_labels = df_mimic_test[columns_of_interest].values  # ground truth labels
# pred_labels = df_G[columns_of_interest].values

print(df_mimic_test.head())
print(df_mimic_test_G.head())
print(gt_labels[0])
print(pred_labels[0])

print(f"gt_labels 长度: {gt_labels.shape[0]}")
print(f"pred_labels 长度: {pred_labels.shape[0]}")

micro_f1 = f1_score(gt_labels.flatten(), pred_labels.flatten(), average='micro')
macro_f1 = f1_score(gt_labels.flatten(), pred_labels.flatten(), average='macro')

print(f"Micro-F1: {micro_f1*100:.1f}")
print(f"Macro-F1: {macro_f1*100:.1f}")
'''

'''
types = df['type'].unique()
print(types)
for t in types:
    print(f"\n=== Type: {t} ===")
    
    # 筛选出当前类型的数据
    df_t = df[df['type'] == t].reset_index(drop=True)
    df_G_t = df_G[df_G['type'] == t].reset_index(drop=True)
    
    # 提取报告文本
    references_t = df_t['report'].tolist()
    candidates_t = df_G_t['report'].tolist()
    
    # 计算 BLEU 分数
    cal_BLEU(references_t, candidates_t)
    
    # 计算 ROUGE-L 分数
    precision, recall, f1 = calculate_rouge_l(references_t, candidates_t)
    print(f"ROUGE-L Precision: {precision*100:.2f}, Recall: {recall*100:.2f}, F1: {f1*100:.2f}")
    
    # 计算正类 F1 分数
    cal_pos_F1(df_t, df_G_t)
    
    # 计算负类 F1 分数
    cal_neg_F1(df_t, df_G_t)
'''