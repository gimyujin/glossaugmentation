import pandas as pd
from collections import defaultdict
from typing import Dict, List
from config import Config

class GlossDictionary:
    def __init__(self, dict_path: str = Config.DICT_PATH):
        self.dict_path = dict_path
        self.gloss_map = self._build_dict()

    def _clean_text(self, x) -> str:
        if pd.isna(x): return ""
        return str(x).strip()

    def _is_nan_like(self, x) -> bool:
        if pd.isna(x): return True
        s = str(x).strip().lower()
        return s in {"", "nan", "none", "null"}

    def _build_dict(self) -> Dict[str, List[str]]:
        df = pd.read_csv(self.dict_path)
        temp_dict = defaultdict(list)

        for _, row in df.iterrows():
            key = self._clean_text(row[Config.DICT_KEY_COL])
            if not key: continue

            comb = row.get(Config.DICT_COMB_COL, "")
            base = self._clean_text(row[Config.DICT_BASE_GLOSS_COL])

            # 조합형이 있으면 우선순위, 없으면 기본형
            gloss_form = self._clean_text(comb) if not self._is_nan_like(comb) else base
            
            if gloss_form and gloss_form not in temp_dict[key]:
                temp_dict[key].append(gloss_form)
        
        return dict(temp_dict)

    def get_glosses(self, key: str) -> List[str]:
        """단어에 해당하는 수어 리스트를 긴 순서대로 반환"""
        candidates = self.gloss_map.get(key, [])
        return sorted(candidates, key=len, reverse=True)

    def __contains__(self, key: str):
        return key in self.gloss_map