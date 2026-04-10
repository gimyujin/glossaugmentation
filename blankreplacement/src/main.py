from config import Config
from dictionary import GlossDictionary
from mlm import MaskedWordPredictor
from tokenizer import KoreanAnalyzer
from augmenter import BRAugmenter
from pipeline import AugmentationPipeline

def main():
    # 1. 모든 부품 인스턴스화
    gloss_dict = GlossDictionary()
    predictor = MaskedWordPredictor()
    analyzer = KoreanAnalyzer()
    
    # 2. 증강 엔진 조립
    augmenter = BRAugmenter(gloss_dict, predictor, analyzer)
    
    # 3. 파이프라인 가동
    pipeline = AugmentationPipeline(augmenter)
    pipeline.run()

if __name__ == "__main__":
    main()