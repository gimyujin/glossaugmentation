import re
from typing import List, Dict, Optional, Tuple
from config import Config

def is_valid_text(s: str) -> bool:
    """한글, 영어, 숫자로만 이루어진 유효한 단어인지 검사"""
    s = s.strip()
    if not s: return False
    return bool(re.fullmatch(r"[가-힣a-zA-Z0-9]+", s))

def filter_candidates(
    raw_candidates: List[Dict],
    original_info: Dict, # 원본 단어의 surface, dict_key, pos 포함
    analyzer,            # KoreanAnalyzer 객체
    gloss_dict           # GlossDictionary 객체
) -> List[Dict]:
    filtered = []
    seen_keys = set()

    for cand in raw_candidates:
        surface = cand["candidate_surface"]

        # 1. 유효 텍스트 및 원본 동일 여부 체크
        if not is_valid_text(surface) or surface == original_info['surface']:
            continue

        # 2. 형태소 분석 및 품사 체크
        # analyzer.okt.pos를 직접 호출하거나 정규화 함수 활용
        try:
            pos_res = analyzer.okt.pos(surface, stem=True)
            if not pos_res: continue
            cand_surface, cand_pos = pos_res[0]
        except:
            continue

        # 3. 상세 조건 필터링
        # 품사 일치 여부
        if cand_pos != original_info['pos']:
            continue
            
        # 사전 존재 여부
        if cand_surface not in gloss_dict:
            continue
            
        # 원본과 의미(Key)가 중복되는지
        if cand_surface == original_info['dict_key']:
            continue

        # 중복된 후보 방지
        if cand_surface in seen_keys:
            continue

        seen_keys.add(cand_surface)
        filtered.append({
            "candidate_surface": surface,
            "candidate_dict_key": cand_surface,
            "candidate_pos": cand_pos,
            "score": cand["score"],
            "sequence": cand["sequence"]
        })

    return filtered