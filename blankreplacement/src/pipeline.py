import pandas as pd
from tqdm import tqdm
from config import Config

class AugmentationPipeline:
    def __init__(self, augmenter):
        self.augmenter = augmenter

    def run(self):
        print(f"데이터 로드 중: {Config.DATA_PATH}")
        df = pd.read_csv(Config.DATA_PATH)
        total_results = []

        # 진행바 출력
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
            wkl = str(row[Config.WKL_COL]).strip()
            gksl = str(row[Config.GKSL_COL]).strip()

            try:
                # 한 줄에 대해 여러 증강 데이터 생성
                aug_list = self.augmenter.augment_row(wkl, gksl)
                total_results.extend(aug_list)
            except Exception as e:
                # 에러가 나도 로그만 찍고 다음 행으로 진행
                continue

        # 최종 결과 저장
        result_df = pd.DataFrame(total_results)
        result_df.to_csv(Config.OUTPUT_PATH, index=False, encoding="utf-8-sig")
        print(f"저장 완료: {Config.OUTPUT_PATH} (총 {len(result_df)}행)")