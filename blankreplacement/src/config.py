import torch

class Config:
    # 경로 설정
    DICT_PATH = r"data\ksl_gloss_dictionary_for_mapping.csv"
    DATA_PATH = r"data\train7000.csv"
    OUTPUT_PATH = r"data\br_augmented_final7000.csv"

    # 모델 설정
    MODEL_NAME = "klue/roberta-base"
    TOP_K = 10
    DEVICE = 0 if torch.cuda.is_available() else -1 

    # 증강 파라미터
    MAX_TARGETS_PER_SENT = 5
    MAX_AUG_PER_SENT = 3
    ALLOWED_OKT_POS = {"Noun", "Verb"}
    
    # 데이터셋 컬럼명
    WKL_COL = "Text"
    GKSL_COL = "Gloss"

    # 사전 컬럼명
    DICT_KEY_COL = "word"
    DICT_BASE_GLOSS_COL = "word"
    DICT_COMB_COL = "combination"