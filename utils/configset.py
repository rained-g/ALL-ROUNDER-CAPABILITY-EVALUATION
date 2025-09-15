# config.py
from enum import Enum
from dataclasses import dataclass
from sched import scheduler

from sklearn.model_selection import learning_curve


class FusionType(Enum):
    LINEAR = 1
    SELF_ATTN = 2
    CROSS_ATTN = 3
    FUSE4 = 4
    CROSS_ATTN_NEW = 5

@dataclass
class TrainingConfig:
    fusion_type: FusionType = FusionType.CROSS_ATTN
    lr_settings: dict = None
    save_dir: str = "./results/LoRA"      #这是训练过程中保存训练过程中模型的根路径
    Generate_files: str = "./results/Generate_files"
    wandb_name: str = "LoRA_with_RadDINO_complete_part4_1"
    checkpoint_path: str = ''
    checkpoint_status: bool = False
    load_status: bool = False
    stage : str = ''
    # 训练参数
    scheduler_norm : bool = False
    Difference : str = "None"
    Prompt_type :str = "old",
    optimizer :str = 'old',
    LLMS :str = "Vicuna"
    MLP_initial :str = "random",
    epoches: int = 1
    batch_size: int = 1
    warmup_ratio: float = 0.05
    input_length: int = 1712
    CF_num: int = 0
    FF_num: int  = 0
    FL_num: int  = 0
    FFL_num: int  = 0
    CF_loss: int = 0
    FF_loss: int = 0
    FL_loss: int = 0
    FFL_loss: int = 0
    path_name: str = ''
    grad_accum_steps: int = 1  # 模拟的 batch size
    learning_ratio: float = 1.0
    gradient_checkpointing: bool = True  # 禁用缓存，适用于梯度检查点
    prompt = {}
    prompt_embed = {}
    attention_mask = {}
    def __post_init__(self):
        self.lr_settings = self.lr_settings or {
            'projector': 5e-5,
            'lora': 2e-5,
            'fusion': 3e-5
        }
    def reset_num(self):
        self.CF_num = 0
        self.FF_num = 0
        self.FL_num = 0
        self.FFL_num = 0
        self.CF_loss = 0
        self.FF_loss = 0
        self.FL_loss = 0
        self.FFL_loss = 0

    def original_prompt_vicuna(self):  #比不过下面的那个新的
        self.prompt = {
            "usual1" : "}</s>",
            "system_prompt" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.</s><s>Given the current frontal image: {",
            "C_lateral" : "The current lateral image: {",
            "P_frontal" : "The prior frontal image: {",
            "P_Re" : "PRIOR_findings: {",
            "user_prompt" : "Provide a description of the findings in the radiology study.</s> <s>INDICATION: {",
            "Tech" : "TECHNIQUE: {",
            "Comp" : "COMPARISON: {",
            "system_prompt_fused1" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.</s><s>Given the fusion image information of the current frontal image, the current lateral image and the prior frontal image: {",
            "system_prompt_fused2" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.</s><s>Given the fusion image information of the current frontal image and the prior frontal image: {",
            "system_prompt_fused3" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.</s><>Given the fusion image information of the current frontal image and the current lateral image: {",
        }
    
    def old_prompt_vicuna(self):
            self.prompt = {
                "usual1" : "}\n",
                "system_prompt" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.</s><s>Given the current frontal image: {",
                "C_lateral" : "The current lateral image: {",
                "P_frontal" : "The prior frontal image: {",
                "P_Re" : "PRIOR_findings: {",
                "user_prompt" : "Provide a description of the findings in the radiology study.\n INDICATION: {",
                "Tech" : "TECHNIQUE: {",
                "Comp" : "COMPARISON: {",
                "system_prompt_fused1" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.</s><s>Given the fusion image information of the current frontal image, the current lateral image and the prior frontal image: {",
                "system_prompt_fused2" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.</s><s>Given the fusion image information of the current frontal image and the prior frontal image: {",
                "system_prompt_fused3" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.</s><s>Given the fusion image information of the current frontal image and the current lateral image: {",
            }

    def new_prompt_Qwen(self):
            self.prompt = {
                "system_prompt": "<|im_start|>system\nYou are an expert radiology assistant tasked with interpreting a chest X-ray study.<|im_end|>\n",

                "user_prompt_multi_image": (
                    "<|im_start|>user\n"
                    "<|vision_start|>Given the following radiology images:\n"
                    "{current_frontal_block}"
                    "{current_lateral_block}"
                    "{prior_frontal_block}"
                    "{CFCL_info}"
                    "{CFPF_info}"
                    "{CFPFCL_info}<|vision_end|>\n"
                    "{prior_report_block}"
                    "{indication_block}"
                    "{technique_block}"
                    "{comparison_block}"
                    "Please provide a comprehensive description of the findings in the radiology study."
                    "<|im_end|>\n"
                    "<|im_start|>assistant\n"
                ),
            }
    def new_prompt_vicuna(self):
            self.prompt = {
                "usual1" : "}\n",
                "system_prompt" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.</s>\nGiven the full current frontal image:{",
                "C_lateral" : "The current lateral image:{",
                "P_frontal" : "The prior frontal image:{",
                "delta_image": "Comparison information between current frontal image and prior frontal image:{",
                "patches_image" : "Pacth images of current frontal image, current lateral image and prior frontal image:{",
                # "P_Re" : "PRIOR_Report:{",      #MAIRA2
                "P_Re" : "PRIOR-Study's Findings:{",      #Fuse3
                "user_prompt" : "Provide a description of the findings in the radiology study.\nINDICATION:{",
                "Tech" : "TECHNIQUE:{",
                "Comp" : "COMPARISON:{",
                "analysis": "Review of prior prediction errors and correction suggestions:{",
                "system_prompt_fused1" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.</s>\nGiven the fusion image information of the current frontal image, the current lateral image and the prior frontal image:{",
                "system_prompt_fused2" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.</s>\nGiven the fusion image information of the current frontal image and the prior frontal image:{",
                "system_prompt_fused3" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.</s>\nGiven the fusion image information of the current frontal image and the current lateral image:{",
                "system_prompt_fused" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.</s><s>Given the fusion information of input images:{",
                }
    
    def new_prompt_vicuna_maira2(self):
            self.prompt = {
                "usual1" : "}\n",
                "system_prompt" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.</s>\nGiven the full current frontal image:{",
                "C_lateral" : "The current lateral image:{",
                "P_frontal" : "The prior frontal image:{",
                "delta_image": "Comparison information between current frontal image and prior frontal image:{",
                "patches_image" : "Pacth images of current frontal image, current lateral image and prior frontal image:{",
                "P_Re" : "PRIOR_Report:{",      #MAIRA2
                # "P_Re" : "PRIOR-Study's Findings:{",      #Fuse3
                "user_prompt" : "Provide a description of the findings in the radiology study.\nINDICATION:{",
                "Tech" : "TECHNIQUE:{",
                "Comp" : "COMPARISON:{",
                "analysis": "Review of prior prediction errors and correction suggestions:{",
                "system_prompt_fused1" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.</s>\nGiven the fusion image information of the current frontal image, the current lateral image and the prior frontal image:{",
                "system_prompt_fused2" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.</s>\nGiven the fusion image information of the current frontal image and the prior frontal image:{",
                "system_prompt_fused3" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.</s>\nGiven the fusion image information of the current frontal image and the current lateral image:{",
                "system_prompt_fused" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.</s><s>Given the fusion information of input images:{",
                }

    def vicuna_prompt_thought(self):
         self.prompt = {
              "usual1" : "}\n",
            #   "system_prompt_fused" : "[AI Assistant Guiding Principles]\n You are a world-class AI radiology assistant. Your task is to generate clinically accurate, logically consistent, and professionally formatted radiology reports. Before generating any content, you must strictly adhere to the following three core principles:\nPrinciple 1: Principle of Objectivity\nYour report 【must be based solely】on the visual findings observed in the provided imaging, as well as comparisons with any available prior studies.\n You are strictly prohibited from restating or paraphrasing the 【INDICATION,TECHNIQUE or COMPARISON】fields in your findings section. Your role is to interpret the imaging—not to repeat the clinical request.\n Principle 2: Principle of Logical Consistency\n All of your conclusions 【must be internally coherent.】 Before finalizing the report, perform a “self-check” to ensure that you have not made any contradictory statements.\n If a finding is uncertain, clearly label it as 【'uncertain' or 'possible'】, but never contradict a definitive statement with a conflicting one.\n Principle 3: Principle of Diagnostic Clarity\n Your output must provide maximum clinical value to referring physicians. Please follow the structure below:\n 1. Begin with an objective and detailed description of all significant imaging findings(especially any positive findings.)\n 2. After describing all your findings, you 【must provide one or more clear, concise diagnostic impressions】 related to that observation.\n 3. Diagnostic impressions should be phrased using 【standardized medical terminology】 (e.g., Cardiomegaly, Atelectasis, Lung Opacity, Edema) to ensure clarity.\n 4.Your diagnostic conclusions must be logically consistent with the preceding imaging descriptions.</s>\nGiven the fusion information of input images:{",
            "system_prompt_fused" : "[AI Assistant Guiding Principles]\n You are a world-class AI radiology assistant. Your task is to generate clinically accurate, logically consistent, and professionally formatted radiology Findings. Before generating any content, you must strictly adhere to the following three core principles:\nPrinciple 1: Principle of Objectivity\nYour findings 【must be based solely】on the visual findings observed in the provided imaging, as well as comparisons with any available prior studies.\n You are strictly prohibited from restating or paraphrasing the 【INDICATION,TECHNIQUE or COMPARISON】fields in your findings section. Your role is to interpret the imaging,not to repeat the clinical request.\n Principle 2: Principle of Logical Consistency\n All of your conclusions 【must be internally coherent.】 Before finalizing the findings, perform a “self-check” to ensure that you have not made any contradictory statements.\n If a finding is uncertain, clearly label it as 【'uncertain' or 'possible'】, but never contradict a definitive statement with a conflicting one.\n Principle 3: Principle of Diagnostic Clarity\n Your output must provide maximum clinical value to referring physicians. Please follow the structure below:\n 1. Begin with an objective and detailed description of all significant imaging findings(especially any positive findings.)\n 2. After describing all your findings, you 【must provide one or more clear, concise diagnostic impressions】 related to that observation.\n 3. Diagnostic impressions should be phrased using 【standardized medical terminology】 (e.g., Cardiomegaly, Atelectasis, Lung Opacity, Edema) to ensure clarity.\n 4.Your diagnostic conclusions must be logically consistent with the preceding imaging descriptions.</s>\nGiven the fusion information of input images:{",
            #   "system_prompt_fused" : "[AI Assistant Guiding Principles]\n You are a world-class radiology assistant. Your task is to interpret a chest X-ray study and provide a description of the findings in the radiology study. Before generating any content, you must strictly adhere to the following three core principles:\nPrinciple 1: Principle of Objectivity\nYour findings 【must be based solely】on the visual findings observed in the provided imaging, as well as comparisons with any available prior studies.\n You are strictly prohibited from restating or paraphrasing the 【INDICATION,TECHNIQUE or COMPARISON】fields in your findings section. Your role is to interpret the imaging, not to repeat the clinical request.\n Principle 2: Principle of Logical Consistency\n All of your conclusions 【must be internally coherent.】 Before finalizing the findings, perform a “self-check” to ensure that you have not made any contradictory statements.\n If a finding is uncertain, clearly label it as 【'uncertain' or 'possible'】, but never contradict a definitive statement with a conflicting one.\n Principle 3: Principle of Integrated Diagnosis\n Your primary goal is to write a single, coherent, and conclusive Findings section. To do this, you must:\n 1. Describe and Diagnose Together: As you describe your objective observations (e.g., 'opacity', 'enlargement'), you 【must】 integrate your clinical interpretations and diagnostic conclusions where appropriate within the same narrative.\n 2. Use Diagnostic Language: These conclusions should use standard diagnostic phrasing (e.g., '...concerning for pneumonia,' '...suggestive of cardiomegaly,' '...which may represent a small pleural effusion,' 'No acute findings'). This ensures clinical clarity.\n 3. Maintain Logical Support: Ensure that all your diagnostic statements are directly supported by the objective observations you have just described.</s>\nGiven the fusion information of input images:{",
            #   "P_Re" : "PRIOR_Report:{",
              "P_Re" : "PRIOR-Study's Findings:{",
              "user_prompt" : "INDICATION:{",
              "Tech" : "TECHNIQUE:{",
              "Comp" : "COMPARISON:{",
              "analysis": "Review of prior prediction errors and correction suggestions:{",


              "system_prompt_fused1" : "[AI Assistant Guiding Principles]\n You are a world-class AI radiology assistant. Your task is to generate clinically accurate, logically consistent, and professionally formatted radiology Findings. Before generating any content, you must strictly adhere to the following three core principles:\nPrinciple 1: Principle of Objectivity\nYour findings 【must be based solely】on the visual findings observed in the provided imaging, as well as comparisons with any available prior studies.\n You are strictly prohibited from restating or paraphrasing the 【INDICATION,TECHNIQUE or COMPARISON】fields in your findings section. Your role is to interpret the imaging,not to repeat the clinical request.\n Principle 2: Principle of Logical Consistency\n All of your conclusions 【must be internally coherent.】 Before finalizing the findings, perform a “self-check” to ensure that you have not made any contradictory statements.\n If a finding is uncertain, clearly label it as 【'uncertain' or 'possible'】, but never contradict a definitive statement with a conflicting one.\n Principle 3: Principle of Diagnostic Clarity\n Your output must provide maximum clinical value to referring physicians. Please follow the structure below:\n 1. Begin with an objective and detailed description of all significant imaging findings(especially any positive findings.)\n 2. After describing all your findings, you 【must provide one or more clear, concise diagnostic impressions】 related to that observation.\n 3. Diagnostic impressions should be phrased using 【standardized medical terminology】 (e.g., Cardiomegaly, Atelectasis, Lung Opacity, Edema) to ensure clarity.\n 4.Your diagnostic conclusions must be logically consistent with the preceding imaging descriptions.\n",
            #   "image": "Given the fusion information of input images:{",
            #   "user_prompt" : "Provide a description of the findings in the radiology study.\nINDICATION:{",
         }
    def new_prompt_DL(self):
            self.prompt = {
                "Part1" : "[RADIOLOGY REPORT GENERATION TASK]\n ### Input Data:\n" ,
                "CF" : "[CURRENT FRONTAL IMAGE]:",
                "CFCL" : "[FUSED INFORMATION OF CURRENT LATERAL IMAGE AND CURRENT LATERAL IMAGE]: ",
                "CFPF" : "[FUSED INFORMATION OF CURRENT LATERAL IMAGE AND PRIOR FRONTAL IMAGE]: ",
                "ALL" : "[FUSED INFORMATION OF CURRENT LATERAL IMAGE, CURRENT LATERAL IMAGE AND PRIOR FRONTAL IMAGE]: ",
                "IND" : "<｜end▁of▁sentence｜><｜begin▁of▁sentence｜>--INDICATION--:",
                "Tech" : "--TECHNIQUE--:",
                "Comp" : "--COMPARISON--:",
                "P_Re" : "--PRIOR FINDINGS--:",
                "user_prompt" : "### Structured Findings Generation Requirements\n<think>\n1. Anatomical Structure Analysis:\n- Identify the opacity status of [Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices]\n   a. Identify anatomical structures\n   b. Detect pathological signs\n   c. Compare with prior studies2. Apply three-phase reasoning:\n   Phase 1: Primary observation, localize the coordinates of abnormal regions\n   Phase 2: Quantify features (size/density/boundary)\n   Phase 3: Perform cross-modal comparative analysis\n3. Priority Ordering of Findings:\n[Critical] > [Significant] > [Incidental]\n\n### Output Guidelines\nBEGIN FINDINGS\n<｜end▁of▁sentence｜>",
            }
        
    def new_prompt_DL_change(self):
            self.prompt = {
                "Part1" : "[RADIOLOGY REPORT GENERATION TASK]\n ### Input Data:\n" ,
                "CF" : "[CURRENT FRONTAL IMAGE]:",
                "CFCL" : "[FUSED INFORMATION OF CURRENT LATERAL IMAGE AND CURRENT LATERAL IMAGE]: ",
                "CFPF" : "[FUSED INFORMATION OF CURRENT LATERAL IMAGE AND PRIOR FRONTAL IMAGE]: ",
                "ALL" : "[FUSED INFORMATION OF CURRENT LATERAL IMAGE, CURRENT LATERAL IMAGE AND PRIOR FRONTAL IMAGE]: ",
                "IND" : "\n\n--INDICATION--:",
                "Tech" : "--TECHNIQUE--:",
                "Comp" : "--COMPARISON--:",
                "P_Re" : "--PRIOR FINDINGS--:",
                "user_prompt" : "### Structured Findings Generation Requirements\n<think>\n1. Anatomical Structure Analysis:\n- Identify the opacity status of [Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices]\n   a. Identify anatomical structures\n   b. Detect pathological signs\n   c. Compare with prior studies2. Apply three-phase reasoning:\n   Phase 1: Primary observation, localize the coordinates of abnormal regions\n   Phase 2: Quantify features (size/density/boundary)\n   Phase 3: Perform cross-modal comparative analysis\n3. Priority Ordering of Findings:\n[Critical] > [Significant] > [Incidental]\n\n### Output Guidelines\nBEGIN FINDINGS\n<｜end▁of▁sentence｜>",
            }

    
    def optimized_medical_prompt(self):
        self.prompt = {
            "fusion_mapping": {
                "CF only": "Fusion of Current Frontal View (CF)",
                "CF+PF": "Fusion of Current Frontal View (CF) and Prior Frontal View (PF)",
                "CF+CL": "Fusion of Current Frontal View (CF) and Current Lateral View (CL)",
                "CF+CL+PF": "Fusion of Current Frontal/Lateral Views (CF+CL) and Prior Frontal View (PF)"
            },
            "full_prompt": (
                "[RADIOLOGY REPORT GENERATION TASK]\n"
                "### Input Data:\n"
                "1. [CURRENT FRONTAL IMAGE]: {frontal_features}\n"
                "2. [FUSED INFORMATION OF CURRENT LATERAL IMAGE AND CURRENT LATERAL IMAGE]: {lateral_features}\n"
                "3. [FUSED INFORMATION OF CURRENT LATERAL IMAGE AND PRIOR FRONTAL IMAGE]: {prior_frontal}\n"
                "4. [FUSED INFORMATION OF CURRENT LATERAL IMAGE, CURRENT LATERAL IMAGE AND PRIOR FRONTAL IMAGE]: {prior_frontal}\n"
                "--INDICATION--: {indication_text}\n"
                "--TECHNIQUE--: {technique_details}\n"
                "--COMPARISON--: {comparison_text}\n"
                "--PRIOR FINDINGS--: {prior_findings}\n\n"
                
                "### Structured Findings Generation Requirements\n"
                "<think>\n"  # 强制推理起始标记
                "1. Anatomical Structure Analysis:\n"
                "- Identify the opacity status of {anatomy_list}\n"
                "   a. Identify anatomical structures\n"
                "   b. Detect pathological signs\n"
                "   c. Compare with prior studies\n"
                "2. Apply three-phase reasoning:\n"
                "   Phase 1: Primary observation\n"
                "   Phase 2: Differential diagnosis\n"
                "   Phase 3: Clinical correlation\n"
                "3. Generate report with:\n"
                "   - Finding prioritization\n\n"
                
                "### Output Format Requirements:\n"
                "BEGIN REPORT\n"
                "**Impression:**\n"
                "1. {finding1}\n"
                "2. {finding2}\n"
                "**Recommendations:**\n"
                "- {recommendation}\n"
                "END REPORT"
            )
        }

    def old_prompt_Qwen(self):
            self.prompt = {
                "usual1_img" : "<|vision_end|>",
                "system_prompt" : "<|im_start|>You are an expert radiology assistant tasked with interpreting a chest X-ray study.<|im_end|> Given the current frontal image: <|vision_start|>",
                "C_lateral" : "The current lateral image: <|vision_start|>",
                "P_frontal" : "The prior frontal image: <|vision_start|>",
                "P_Re" : "PRIOR_Report: ",
                "user_prompt" : "Provide a description of the findings in the radiology study.<|im_end|> INDICATION: ",
                "Tech" : "TECHNIQUE: ",
                "Comp" : "COMPARISON: ",
                "system_prompt_fused1" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.<|im_end|> Given the fusion image information of the current frontal image, the current lateral image and the prior frontal image: <|vision_start|>",
                "system_prompt_fused2" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.<|im_end|> Given the fusion image information of the current frontal image and the prior frontal image: <|vision_start|>",
                "system_prompt_fused3" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.<|im_end|> Given the fusion image information of the current frontal image and the current lateral image: <|vision_start|>",
            }

    def new_prompt_Qwen(self):
            self.prompt = {
                "system_prompt": "<|im_start|>system\nYou are an expert radiology assistant tasked with interpreting a chest X-ray study.<|im_end|>\n",

                "user_prompt_multi_image": (
                    "<|im_start|>user\n"
                    "<|vision_start|>Given the following radiology images:\n"
                    "{current_frontal_block}"
                    "{current_lateral_block}"
                    "{prior_frontal_block}"
                    "{CFCL_info}"
                    "{CFPF_info}"
                    "{CFPFCL_info}<|vision_end|>\n"
                    "{prior_report_block}"
                    "{indication_block}"
                    "{technique_block}"
                    "{comparison_block}"
                    "Please provide a comprehensive description of the findings in the radiology study."
                    "<|im_end|>\n"
                    "<|im_start|>assistant\n"
                ),
            }

        
    def new_prompt(self):
        self.prompt = {
            "system_prompt" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.</s> Given the the following information: [Current Frontal Image]: {",
            "C_lateral" : "[Current Lateral Image]: {",
            "P_frontal" : "[Prior Frontal Image]: {",
            "system_prompt_fused1" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.</s> Given the following information: [Fusion Image Information of the current frontal image, the current lateral image and the prior frontal image]: {",
            "system_prompt_fused2" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.</s> Given the following information: [Fusion Image Information of the current frontal image and the prior frontal image]: {",
            "system_prompt_fused3" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.</s> Given the following information: [Fusion Image Information of the current frontal image and the current lateral image]: {",
            "P_Re" : "[PRIOR_findings]: {",
            "Ind" : "[INDICATION]: {",
            "Tech" : "[TECHNIQUE]: {",
            "Comp" : "[COMPARISON]: {",
            "user_prompt" : 'Instructions: 1. Analyze all the provided information carefully. 2. Summarize key abnormalities and notable findings from the current images. 3. If there is a comparison or prior image or PRIOR_findings, clearly describe any changes or differences. 4. The generated "findings" report should be clear, professional, and succinct. Provide a description of the findings in the radiology study.Please reason step by step, and give your final answer: <think></s>',
            
        }

    def get_DL_prompt(self):
        self.prompt = {
            "user_prompt": (
                "[RADIOLOGY REPORT GENERATION TASK]\n"
                "### Input Data:\n"
                "{cf_image_description}"
                "{cl_image_description}"
                "{pf_image_description}"
                "{cfpf_info}"
                "{cfcl_info}"
                "{cfclpf_info}"
                "{indication}"
                "{technique}"
                "{comparison}"
                "{prior_report}"
                "### Structured Findings Generation Instructions:\n"
                "You are tasked with generating the FINDINGS section of a radiology report based on the data above.\n"
                "Please analyze the anatomical structures and report findings for the following categories:\n"
                "- Enlarged Cardiomediastinum\n"
                "- Cardiomegaly\n"
                "- Lung Opacity\n"
                "- Lung Lesion\n"
                "- Edema\n"
                "- Consolidation\n"
                "- Pneumonia\n"
                "- Atelectasis\n"
                "- Pneumothorax\n"
                "- Pleural Effusion\n"
                "- Pleural Other\n"
                "- Fracture\n"
                "- Support Devices\n"
                "Use three-phase reasoning:\n"
                "1. Localize abnormalities (coordinates/regions)\n"
                "2. Quantify features (e.g., size, density, boundary)\n"
                "3. Compare current and prior images\n"
                "Order your findings by clinical importance: [Critical] > [Significant] > [Incidental].\n"
                "Respond with the FINDINGS section only.\n"
                "Begin your response with `<think>` followed by your reasoning.<｜end▁of▁sentence｜>\n\n"
            )
        }

@dataclass 
class ModelConfig:
    # 模型路径
    vicuna_path: str = "./model/lmsys/vicuna-7b-v1.5"
    dino_path: str = "./model/rad-dino-maira-2"
    Qwen_path: str = "./model/Qwen/qwen2.5-7b"
    DS_distill: str = "./model/DS/distill-llama-8B"
    model_save_path: str  = "./results/LoRA"
    model_load_path: str  = "./results/LoRA"
    # 投影器参数
    projector_type: str = "mlp4x_gelu"
    target_dim: int = 4096
    hidden_size: int = 1024
    mm_hidden_size: int = 768
    target_dim: int = 4096
    projetc_save_path: str = './results/MLP'
    projetc_load_path: str = './results/MLP'
    # LoRA配置
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    def __post_init__(self):
        self.projector_type = self.projector_type or "mlp4x_gelu"

@dataclass 
class MultiviewConfig:
    # fuse2参数
    target_length: int = 1650
    image_dim: int = 768
    nhead: int = 8
    TARGET_PATCH_GROUPS: int = 5
    SUMMARY_OUTPUT_SEQ_LEN: int = 128

    #fuse3参数
    view_types: list = None
    multia_save_path: str  = './results/multiviewsatt'
    summary_save_path: str = './results/summary_save_path'
    view_encoder_save_path: str = './results/view'

    multia_load_path: str  = './results/multiviewsatt'
    def __post_init__(self):
        self.view_types = ['cf_cl', 'cf_pf', 'cf_cl_pf', 'self']


# 主程序中使用
# from config import TrainingConfig, FusionType
# from optimizer_factory import create_optimizer

# cfg = TrainingConfig(
#     fusion_type=FusionType.CROSS_ATTN,
#     base_lr=2e-5,
#     save_dir="./results/cross_attn_v2"
# )
# optimizer = create_optimizer(cfg, model, peft_vicuna)