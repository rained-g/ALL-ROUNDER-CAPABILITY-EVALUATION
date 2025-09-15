import re
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
import torch

def load_datasets_path(str):
    all_paths = {}
    all_paths["data_complete_35k"] = r'./datasets/MIMIC-complete/processed_data_MARIA2/train_data_35000.csv'
    all_paths["data_complete_train"] = r'./datasets/MIMIC-complete/processed_data_MARIA2/train_data.csv'
    all_paths["data_complete_valid"] = r'./datasets/MIMIC-complete/processed_data_MARIA2/val_data.csv'
    all_paths["data_complete_test"] = r'./datasets/MIMIC-complete/processed_data_MARIA2/test_data.csv'
    return all_paths[str]
def load_cachesets(str):
    all_paths = {}
    all_paths["data_complete_35k"] = r"./datasets_cache/load/data_complete_MARIA2_train_35k"
    all_paths["data_complete_train"] = r"./datasets_cache/load/data_complete_MARIA2_train"
    all_paths["data_complete_valid"] = r"./datasets_cache/load/data_complete_MARIA2_valid"
    all_paths["data_complete_test"] = r"./datasets_cache/load/data_complete_MARIA2_test"
    all_paths["data_complete_all_test_new"] = r"./datasets_cache/load/data_complete_all_test_new"
    return all_paths[str]
def load_processed_cache_path(str):
    all_paths = {}
    all_paths["Vicuna_35k"] = r'./datasets_cache/map/Vicuna/data_complete_train_35K.arrow'
    all_paths['Vicuna_all_test_new'] = r'./datasets_cache/map/Vicuna/all_test_new.arrow'
    all_paths["DL_35k"] = r'./datasets_cache/map/Vicuna/data_complete_train_35K.arrow'
    all_paths["DL_all_test_new"] = r'./datasets_cache/map/DL/all_test_new.arrow'
    return all_paths[str]
def get_path(str1,str2):
    results = {}
    results['data'] = load_datasets_path(str1)
    results['cache'] = load_cachesets(str1)
    results["processed_cache"] = load_processed_cache_path(str2)
    return results

def process_text(text_list):
    return [re.sub(r'\s+', ' ', text).strip() for text in text_list]

def get_prompt_img_templates():
    return {
        "system_prompt" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study. </s>",
        "user_prompt" : "Provide a description of the findings in the radiology image. </s>"
    }
def get_prompt_ind_templates():
    return{
        "system_prompt" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.</s> Given the current frontal image {",
        "user_prompt" : "} Provide a description of the findings in the radiology study.</s> INDICATION: {"
    }
def get_prompt_all_templates():
    return {
        "usual1" : "}</s>",
        "system_prompt" : "You are an expert radiology assistant tasked with interpreting a chest X-ray study.</s> Given the current frontal image: {",
        "C_lateral" : "The current lateral image: {",
        "P_frontal" : "The prior frontal image: {",
        "P_Re" : "PRIOR_findings: {",
        "user_prompt" : "Provide a description of the findings in the radiology study.</s> INDICATION: {",
        "Tech" : "TECHNIQUE: {",
        "Comp" : "COMPARISON: {",
    }