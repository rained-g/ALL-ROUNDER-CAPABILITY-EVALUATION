from rouge_score import rouge_scorer

def rouge_l(reference, candidate):
    """
    计算 ROUGE-L 分数，参考文本和生成文本作为输入。

    :param reference: 参考文本（字符串类型）
    :param candidate: 生成的文本（字符串类型）
    :return: ROUGE-L Precision, Recall, F1 Score
    """
    # 创建 ROUGE Scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # 计算 ROUGE-L
    scores = scorer.score(reference, candidate)

    return scores['rougeL']

# # 示例
# reference_text = "PA and lateral views were obtained. Lungs are clear. There is no pneumothorax or pleural effusion. The heart and mediastinum are within normal limits. Bony structures are intact. A 5 mm stable right apical nodule."
# candidate_text = "As compared to the previous radiograph, no relevant change is seen. The lung volumes are normal and there is no evidence of pneumonia or pulmonary edema. No pleural effusions. Normal size of the cardiac silhouette without parenchymal opacities concerning for pneumothorax."

# # 计算 ROUGE-L 分数
# rouge_l_scores = rouge_l(reference_text, candidate_text)
# for i in rouge_l_scores:
#     print(i)
# print("ROUGE-L Scores:", rouge_l_scores[2])



"""rouge库
from rouge import Rouge

reference_text = "The heart size is within normal limits. Prominent right paratracheal soft tissues XXXX representing adenopathy. No focal airspace consolidation, pleural effusions or pneumothorax. No acute bony abnormalities."
candidate_text = "Heart size is normal. The mediastinal and hilar contours are unremarkable. Lungs are clear without focal consolidation, pleural effusion or pneumothorax. No acute osseous abnormalities identified."

rouge = Rouge()
scores = rouge.get_scores(candidate_text, reference_text)

# 打印结果
print("ROUGE-1 precision:", scores[0]["rouge-1"]["p"])
print("ROUGE-1 recall:", scores[0]["rouge-1"]["r"])
print("ROUGE-1 F1 score:", scores[0]["rouge-1"]["f"])


"""




"""coco库
import numpy as np
import pdb

def my_lcs(string, sub):

    if(len(string)< len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0,len(sub)+1)] for j in range(0,len(string)+1)]

    for j in range(1,len(sub)+1):
        for i in range(1,len(string)+1):
            if(string[i-1] == sub[j-1]):
                lengths[i][j] = lengths[i-1][j-1] + 1
            else:
                lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])

    return lengths[len(string)][len(sub)]

class Rouge():
    '''
    Class for computing ROUGE-L score for a set of candidate sentences for the MS COCO test set

    '''
    def __init__(self):
        # vrama91: updated the value below based on discussion with Hovey
        self.beta = 1.2

    def calc_score(self, candidate, refs):

        assert(len(candidate)==1)	
        assert(len(refs)>0)         
        prec = []
        rec = []

        # split into tokens
        token_c = candidate[0].split(" ")
    	
        for reference in refs:
            # split into tokens
            token_r = reference.split(" ")
            # compute the longest common subsequence
            lcs = my_lcs(token_r, token_c)
            prec.append(lcs/float(len(token_c)))
            rec.append(lcs/float(len(token_r)))

        prec_max = max(prec)
        rec_max = max(rec)

        if(prec_max!=0 and rec_max !=0):
            score = ((1 + self.beta**2)*prec_max*rec_max)/float(rec_max + self.beta**2*prec_max)
        else:
            score = 0.0
        return score

    def compute_score(self, gts, res):

        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        score = []
        for id in imgIds:
            hypo = res[id]
            ref  = gts[id]

            score.append(self.calc_score(hypo, ref))

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

        average_score = np.mean(np.array(score))
        return average_score, np.array(score)

    def method(self):
        return "Rouge"
# 创建 Rouge 类的实例
rouge = Rouge()

# 示例参考文本和候选文本
candidate = ["The lungs are hyperinflated with a large hiatal hernia. No new opacities have developed since the prior study. There is no pleural effusion or pneumothorax. Aortic calcifications are noted in addition to calcification of mediastinal fat on the left side."]
refs = [
    "Hyperinflated lungs as before compatible with emphysema. Left apical chronic inflammatory and fibrotic changes with apical hilar retraction, unchanged since prior XXXX. XXXX opacities and chronic inflammatory change right midlung as before. Stable mediastinal contour without overt evidence of adenopathy. No acute airspace disease or CHF. No XXXX acute abnormalities since the previous chest radiograph.",
]

# 计算 ROUGE-L 分数
rouge_score = rouge.calc_score(candidate, refs)
print("ROUGE-L Score:", rouge_score)



"""