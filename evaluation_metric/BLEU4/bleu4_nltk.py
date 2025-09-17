from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re
# from bleu import Bleu

#是否去掉标点符号
def simple_tokenizer(text):
    # 使用正则表达式匹配所有单词
    # text = re.sub(r'[^\w\s]', '', text.lower())
    return re.findall(r'\b\w+\b', text.lower())

def bleu_score(reference, candidate):
    """
    计算BLEU-4分数，参考翻译和生成翻译传入作为输入。

    :param reference: 参考翻译，应该是一个由句子组成的列表，每个句子是一个词的列表
    :param candidate: 生成的翻译，应该是一个词的列表
    :return: BLEU-4 分数
    """

    # 使用 smoothing_function 对 BLEU 分数进行平滑，避免零分
    smoothie = SmoothingFunction().method4
    reference = [simple_tokenizer(reference)]  # 参考文本
    candidate = simple_tokenizer(candidate)  # 生成文本
    bleu_scores = {}
    bleu_scores['BLEU-1'] = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu_scores['BLEU-2'] = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu_scores['BLEU-3'] = sentence_bleu(reference, candidate, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothie)
    bleu_scores['BLEU-4'] = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    return bleu_scores


# 示例使用
# reference1 = "The heart size is within normal limits. Prominent right paratracheal soft tissues XXXX representing adenopathy. No focal airspace consolidation, pleural effusions or pneumothorax. No acute bony abnormalities."
# candidate1 ="Heart size is normal. The mediastinal and hilar contours are unremarkable. Lungs are clear without focal consolidation, pleural effusion or pneumothorax. No acute osseous abnormalities identified."
# # def clean_text(text):
# #     # 使用正则表达式替换所有非字母数字字符为空格
# #     cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
# #     # 按空格分割文本为单词列表
# #     return cleaned_text.split()
# # reference_words = clean_text(reference)
# # candidate_words = clean_text(candidate)
# # 计算 BLEU-4 分数
# results = bleu_score(reference1, candidate1)
# for bleu, score in results.items():
#     print(f'{bleu}: {score:.4f}')

# gts = {'img1': [reference1]}
# res = {'img1': [candidate1]}

# bleu = Bleu(n=4)
# score, scores = bleu.compute_score(gts, res)
# print("BLEU score:", score)
# print("Individual n-gram BLEU scores:", scores)