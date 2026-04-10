import re
from typing import List, Dict, Optional, Tuple
from config import Config
from tokenizer import TargetToken

class BRAugmenter:
    def __init__(self, dictionary, predictor, analyzer):
        self.dict = dictionary
        self.predictor = predictor
        self.analyzer = analyzer

    def _parse_gksl(self, gksl: str) -> Tuple[List[str], str]:
        if gksl is None:
            return [], "space"

        gksl = str(gksl).strip()
        if not gksl:
            return [], "space"

        # 1. bracket 형식 우선
        bracket_tokens = re.findall(r"\[([^\[\]]+)\]", gksl)
        if bracket_tokens:
            return [t.strip() for t in bracket_tokens if t.strip()], "bracket"

        # 2. bracket이 없으면 공백 기반 토큰으로 처리
        space_tokens = gksl.split()
        return [t.strip() for t in space_tokens if t.strip()], "space"

    def _rebuild_gksl(self, tokens: List[str], fmt: str) -> str:
        if fmt == "bracket":
            return "".join(f"[{t}]" for t in tokens)
        return " ".join(tokens)

    def augment_row(self, wkl: str, gksl: str) -> List[Dict]:
        outputs = []

        # 1. 대상 단어 추출
        targets = self.analyzer.extract_targets(wkl, self.dict)

        # GKSL은 target마다 다시 파싱할 필요 없음
        gksl_tokens, gksl_format = self._parse_gksl(gksl)

        if not gksl_tokens:
            return outputs

        for target in targets:
            gloss_candidates = self.dict.get_glosses(target.dict_key)

            # 2. 수어 문장에서 위치 매칭
            match_info = None
            for gloss in gloss_candidates:
                if gloss in gksl_tokens:
                    match_info = (gloss, gksl_tokens.index(gloss))
                    break

            if not match_info:
                continue

            original_gloss, gloss_idx = match_info

            # 3. MLM 예측 및 필터링
            from candidate import filter_candidates

            masked_sent = self.predictor.mask_by_span(wkl, target.start, target.end)
            raw_cands = self.predictor.predict(masked_sent)

            orig_info = {
                "surface": target.surface,
                "dict_key": target.dict_key,
                "pos": target.pos
            }
            filtered = filter_candidates(raw_cands, orig_info, self.analyzer, self.dict)

            # 4. 문장 생성
            local_count = 0
            for cand in filtered:
                new_word = cand["candidate_surface"]

                new_gloss_list = self.dict.get_glosses(cand["candidate_dict_key"])
                if not new_gloss_list:
                    continue
                new_gloss = new_gloss_list[0]

                # 한국어 교체
                aug_wkl = wkl[:target.start] + new_word + wkl[target.end:]

                # 수어 교체
                temp_tokens = list(gksl_tokens)
                temp_tokens[gloss_idx] = new_gloss
                aug_gksl = self._rebuild_gksl(temp_tokens, gksl_format)

                outputs.append({
                    "orig_wkl": wkl,
                    "orig_gksl": gksl,
                    "aug_wkl": aug_wkl,
                    "aug_gksl": aug_gksl,
                    "mlm_score": cand["score"]
                })

                local_count += 1
                if local_count >= Config.MAX_AUG_PER_SENT:
                    break

        return outputs