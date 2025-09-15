# data_processor.py
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd



class DataProcessor:
    def __init__(self, tokenizer: AutoTokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def process_function_vicuna(self,examples):
    #最终的结果是discrete
        processed_datasets={ }
        #在目标文本部分添加终止符号
        object_e = [i+"</s>" for i in examples["findings"]]
        object_i = [i+"}"+ "</s>" if i else "}</s>" for i in examples['indication']]     
        object_pf = [i+"}"+ "</s>" if i else "}</s>" for i in examples['prior_report']]
        object_t = [i+"}"+ "</s>" if i else "}</s>" for i in examples['technique']]
        object_c = [i+"}"+ "</s>" if i else "}</s>" for i in examples['comparison']]
        print(f"indications为{object_i[0]}")
        print(f"prior_findings为{object_pf[0]}")
        print(f"technology为{object_t[0]}")
        print(f"comparisons为{object_c[0]}")
        inputs_prior_text = self.tokenizer(object_pf, padding=True, max_length= 100, truncation=True, return_tensors="pt", add_special_tokens=False)
        inputs_text = self.tokenizer(object_e, padding=True, max_length=140, truncation=True, return_tensors="pt")
        inputs_indication = self.tokenizer(object_i, padding=True, max_length= 30, truncation=True, return_tensors="pt", add_special_tokens=False)
        inputs_technology = self.tokenizer(object_t, padding=True, max_length= 15, truncation=True, return_tensors="pt", add_special_tokens=False)
        inputs_comparison = self.tokenizer(object_c, padding=True, max_length= 18, truncation=True, return_tensors="pt", add_special_tokens=False)
        #labels    目标生成文本的input_ids
        processed_datasets['labels'] = inputs_text["input_ids"]
        processed_datasets['attention_mask'] = inputs_text['attention_mask']
        processed_datasets['prior'] = inputs_prior_text["input_ids"]
        processed_datasets['prior_att'] = inputs_prior_text['attention_mask']
        processed_datasets['indication'] = inputs_indication["input_ids"]
        processed_datasets['indication_att'] = inputs_indication["attention_mask"]
        processed_datasets['technology'] = inputs_technology["input_ids"]
        processed_datasets['technology_att'] = inputs_technology["attention_mask"]
        processed_datasets['comparison'] = inputs_comparison["input_ids"]
        processed_datasets['comparison_att'] = inputs_comparison["attention_mask"]
        print(f"prior_text的维度为{processed_datasets['prior'].shape},类型为{processed_datasets['prior'].dtype}")
        print(f"label的维度为{processed_datasets['labels'].shape},类型为{processed_datasets['labels'].dtype}")
        print(f"att的维度为{processed_datasets['attention_mask'].shape},类型为{processed_datasets['attention_mask'].dtype}")
        print(f"indication的维度为{processed_datasets['indication'].shape},类型为{processed_datasets['technology'].dtype}")
        print(f"comparison的维度为{processed_datasets['comparison'].shape},类型为{processed_datasets['comparison'].dtype}")
        print(f"technique的维度为{processed_datasets['technology'].shape},类型为{processed_datasets['technology'].dtype}")
        return processed_datasets




    def process_function_Qwen(self,examples):
        #最终的结果是discrete
        processed_datasets={ }
        #在目标文本部分添加终止符号
        object_e = [i+"<|im_end|>" for i in examples["findings"]]
        object_i = [i+"}"+ "<|im_end|>" if i else "}<|im_end|>" for i in examples['indication']]     
        object_pf = [i+"}"+ "<|im_end|>" if i else "}<|im_end|>" for i in examples['prior_findings']]
        object_t = [i+"}"+ "<|im_end|>" if i else "}<|im_end|>" for i in examples['technique']]
        object_c = [i+"}"+ "<|im_end|>" if i else "}<|im_end|>" for i in examples['comparison']]
        print(f"indications为{object_i[0]}")
        print(f"prior_findings为{object_pf[0]}")
        print(f"technology为{object_t[0]}")
        print(f"comparisons为{object_c[0]}")
        inputs_prior_text = self.tokenizer(object_pf, padding=True, max_length= 80, truncation=True, return_tensors="pt")
        inputs_text = self.tokenizer(object_e, padding=True, max_length=140, truncation=True, return_tensors="pt")
        inputs_indication = self.tokenizer(object_i, padding=True, max_length= 30, truncation=True, return_tensors="pt")
        inputs_technology = self.tokenizer(object_t, padding=True, max_length= 15, truncation=True, return_tensors="pt")
        inputs_comparison = self.tokenizer(object_c, padding=True, max_length= 18, truncation=True, return_tensors="pt")
        #labels    目标生成文本的input_ids
        processed_datasets['labels'] = inputs_text["input_ids"]
        processed_datasets['attention_mask'] = inputs_text['attention_mask']
        processed_datasets['prior'] = inputs_prior_text["input_ids"]
        processed_datasets['prior_att'] = inputs_prior_text['attention_mask']
        processed_datasets['indication'] = inputs_indication["input_ids"]
        processed_datasets['indication_att'] = inputs_indication["attention_mask"]
        processed_datasets['technology'] = inputs_technology["input_ids"]
        processed_datasets['technology_att'] = inputs_technology["attention_mask"]
        processed_datasets['comparison'] = inputs_comparison["input_ids"]
        processed_datasets['comparison_att'] = inputs_comparison["attention_mask"]
        print(f"prior_text的维度为{processed_datasets['prior'].shape},类型为{processed_datasets['prior'].dtype}")
        print(f"label的维度为{processed_datasets['labels'].shape},类型为{processed_datasets['labels'].dtype}")
        print(f"att的维度为{processed_datasets['attention_mask'].shape},类型为{processed_datasets['attention_mask'].dtype}")
        print(f"indication的维度为{processed_datasets['indication'].shape},类型为{processed_datasets['technology'].dtype}")
        print(f"comparison的维度为{processed_datasets['comparison'].shape},类型为{processed_datasets['comparison'].dtype}")
        print(f"technique的维度为{processed_datasets['technology'].shape},类型为{processed_datasets['technology'].dtype}")
        return processed_datasets
    

    def process_function_DL(self,examples):
        #最终的结果是discrete
        processed_datasets={ }
        #在目标文本部分添加终止符号
        object_e = [i+"<｜end▁of▁sentence｜>" for i in examples["findings"]]
        object_i = [i+ "<｜end▁of▁sentence｜>" if i else "<｜end▁of▁sentence｜>" for i in examples['indication']]     
        object_pf = [i+ "<｜end▁of▁sentence｜>" if i else "<｜end▁of▁sentence｜>" for i in examples['prior_report']]
        object_t = [i+ "<｜end▁of▁sentence｜>" if i else "<｜end▁of▁sentence｜>" for i in examples['technique']]
        object_c = [i+ "<｜end▁of▁sentence｜>" if i else "<｜end▁of▁sentence｜>" for i in examples['comparison']]
        print(f"indications为{object_i[0]}")
        print(f"prior_findings为{object_pf[0]}")
        print(f"technology为{object_t[0]}")
        print(f"comparisons为{object_c[0]}")
        inputs_prior_text = self.tokenizer(object_pf, padding=True, max_length= 100, truncation=True, return_tensors="pt", add_special_tokens=False)
        inputs_text = self.tokenizer(object_e, padding=True, max_length=140, truncation=True, return_tensors="pt")
        inputs_indication = self.tokenizer(object_i, padding=True, max_length= 30, truncation=True, return_tensors="pt", add_special_tokens=False)
        inputs_technology = self.tokenizer(object_t, padding=True, max_length= 15, truncation=True, return_tensors="pt", add_special_tokens=False)
        inputs_comparison = self.tokenizer(object_c, padding=True, max_length= 18, truncation=True, return_tensors="pt", add_special_tokens=False)
        #labels    目标生成文本的input_ids
        processed_datasets['labels'] = inputs_text["input_ids"]
        processed_datasets['attention_mask'] = inputs_text['attention_mask']
        processed_datasets['prior'] = inputs_prior_text["input_ids"]
        processed_datasets['prior_att'] = inputs_prior_text['attention_mask']
        processed_datasets['indication'] = inputs_indication["input_ids"]
        processed_datasets['indication_att'] = inputs_indication["attention_mask"]
        processed_datasets['technology'] = inputs_technology["input_ids"]
        processed_datasets['technology_att'] = inputs_technology["attention_mask"]
        processed_datasets['comparison'] = inputs_comparison["input_ids"]
        processed_datasets['comparison_att'] = inputs_comparison["attention_mask"]
        print(f"prior_text的维度为{processed_datasets['prior'].shape},类型为{processed_datasets['prior'].dtype}")
        print(f"label的维度为{processed_datasets['labels'].shape},类型为{processed_datasets['labels'].dtype}")
        print(f"att的维度为{processed_datasets['attention_mask'].shape},类型为{processed_datasets['attention_mask'].dtype}")
        print(f"indication的维度为{processed_datasets['indication'].shape},类型为{processed_datasets['technology'].dtype}")
        print(f"comparison的维度为{processed_datasets['comparison'].shape},类型为{processed_datasets['comparison'].dtype}")
        print(f"technique的维度为{processed_datasets['technology'].shape},类型为{processed_datasets['technology'].dtype}")
        return processed_datasets

    def get_train_processed_dataset_all(self, data_path,cache_path,processed_cache_path):
        huggingface_data = load_dataset("csv", data_files=data_path, split="train",cache_dir = cache_path)
        return huggingface_data.map(self.process_function_train_all, batched=True, batch_size=1000, remove_columns=huggingface_data.column_names, cache_file_name=processed_cache_path)
    
    def get_all_path(self,data_path,cache_path):
        results = {}
        huggingface_data = load_dataset("csv", data_files=data_path, split="train",cache_dir = cache_path)
        real_paths_train_C_F = ['./datasets/MIMIC-complete/mimic-cxr-images/files/' + i for i in huggingface_data['Current_frontal_dicom_id']]
        real_paths_train_P_F = ['./datasets/MIMIC-complete/mimic-cxr-images/files/' + i if pd.notna(i) else None for i in huggingface_data['Prior_frontal_dicom_id']]
        real_paths_train_C_L = ['./datasets/MIMIC-complete/mimic-cxr-images/files/' + i if pd.notna(i) else None for i in huggingface_data['Current_lateral_dicom_id']]
        results['CF'] = real_paths_train_C_F
        results['PF'] = real_paths_train_P_F
        results['CL'] = real_paths_train_C_L
        return results
    
    def process_function_test_all(self,examples):
        #最终的结果是discrete
        processed_datasets={ }
        #在目标文本部分添加终止符号
        object_i = [i+"}"+ "</s>" if i else "}</s>" for i in examples['indication']]     
        object_pf = [i+"}"+ "</s>" if i else "}</s>" for i in examples['prior_report']]
        object_t = [i+"}"+ "</s>" if i else "}</s>" for i in examples['technique']]
        object_c = [i+"}"+ "</s>" if i else "}</s>" for i in examples['comparison']]
        print(f"indications为{object_i[0]}")
        print(f"prior_findings为{object_pf[0]}")
        print(f"technology为{object_t[0]}")
        print(f"comparisons为{object_c[0]}")
        inputs_prior_text = self.tokenizer(object_pf, padding=True, max_length= 100, truncation=True, return_tensors="pt", add_special_tokens=False)
        inputs_indication = self.tokenizer(object_i, padding=True, max_length= 30, truncation=True, return_tensors="pt", add_special_tokens=False)
        inputs_technology = self.tokenizer(object_t, padding=True, max_length= 15, truncation=True, return_tensors="pt", add_special_tokens=False)
        inputs_comparison = self.tokenizer(object_c, padding=True, max_length= 18, truncation=True, return_tensors="pt", add_special_tokens=False)
        #labels    目标生成文本的input_ids
        processed_datasets['prior'] = inputs_prior_text["input_ids"]
        processed_datasets['prior_att'] = inputs_prior_text['attention_mask']
        processed_datasets['indication'] = inputs_indication["input_ids"]
        processed_datasets['indication_att'] = inputs_indication["attention_mask"]
        processed_datasets['technology'] = inputs_technology["input_ids"]
        processed_datasets['technology_att'] = inputs_technology["attention_mask"]
        processed_datasets['comparison'] = inputs_comparison["input_ids"]
        processed_datasets['comparison_att'] = inputs_comparison["attention_mask"]
        print(f"prior_text的维度为{processed_datasets['prior'].shape},类型为{processed_datasets['prior'].dtype}")
        print(f"pri_att的维度为{processed_datasets['prior_att'].shape},类型为{processed_datasets['prior_att'].dtype}")
        print(f"indication的维度为{processed_datasets['indication'].shape},类型为{processed_datasets['technology'].dtype}")
        print(f"comparison的维度为{processed_datasets['comparison'].shape},类型为{processed_datasets['comparison'].dtype}")
        print(f"technique的维度为{processed_datasets['technology'].shape},类型为{processed_datasets['technology'].dtype}")
        return processed_datasets
    
    def get_test_processed_dataset_all(self, data_path,cache_path,processed_cache_path):
        huggingface_data = load_dataset("csv", data_files=data_path, split="train",cache_dir = cache_path)
        return huggingface_data.map(self.process_function_test_all, batched=True, batch_size=1000, remove_columns=huggingface_data.column_names, cache_file_name=processed_cache_path)






# 主程序中使用

# from data_processor import DataProcessor
# processor = DataProcessor(
#     tokenizer=self.tokenizer, 
#     image_processor=pipe_image
# )
# processed_data = processor.load_dataset(huggingface_data)


