from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from config import Config

class MaskedWordPredictor:
    def __init__(self):
        # 모델과 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.model = AutoModelForMaskedLM.from_pretrained(Config.MODEL_NAME)
        
        # 빈칸 채우기 파이프라인 설정
        self.fill_mask = pipeline(
            "fill-mask",
            model=self.model,
            tokenizer=self.tokenizer,
            top_k=Config.TOP_K,
            device=Config.DEVICE
        )
        self.mask_token = self.tokenizer.mask_token

    def mask_by_span(self, sentence: str, start: int, end: int) -> str:
        """특정 위치의 단어를 [MASK]로 교체"""
        return sentence[:start] + self.mask_token + sentence[end:]

    def predict(self, masked_sentence: str) -> List[Dict]:
        """[MASK]에 들어갈 후보군 예측"""
        results = self.fill_mask(masked_sentence)
        
        candidates = []
        for r in results:
            candidates.append({
                "candidate_surface": str(r["token_str"]).strip(),
                "score": float(r["score"]),
                "sequence": r["sequence"]
            })
        return candidates