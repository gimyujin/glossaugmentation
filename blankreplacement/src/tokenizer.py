from dataclasses import dataclass
from typing import List, Optional
from konlpy.tag import Okt
from config import Config

@dataclass
class TargetToken:
    surface: str
    dict_key: str
    pos: str
    start: int
    end: int

class KoreanAnalyzer:
    def __init__(self):
        self.okt = Okt()

    def normalize_key(self, surface: str, pos: str) -> Optional[str]:
        if not surface.strip(): return None
        # 현재 로직상 명사와 동사만 허용
        return surface.strip() if pos in Config.ALLOWED_OKT_POS else None

    def extract_targets(self, sentence: str, gloss_dict) -> List[TargetToken]:
        targets = []
        try:
            pos_result = self.okt.pos(sentence, stem=True)
        except:
            return targets

        cursor = 0
        for surface, pos in pos_result:
            dict_key = self.normalize_key(surface, pos)
            
            # 사전에도 있어야 하고 허용된 품사여야 함
            if dict_key and dict_key in gloss_dict:
                start = sentence.find(surface, cursor)
                if start == -1: start = sentence.find(surface)
                if start != -1:
                    end = start + len(surface)
                    cursor = end
                    targets.append(TargetToken(surface, dict_key, pos, start, end))
            
            if len(targets) >= Config.MAX_TARGETS_PER_SENT:
                break
        return targets