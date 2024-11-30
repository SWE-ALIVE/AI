import json
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity


class DialogEvaluator:
    def __init__(self, model_name="BM-K/KoSimCSE-roberta", weights=None):
        """
        대화 데이터셋 평가를 위한 평가기

        Args:
            model_name (str): 사용할 언어 모델 이름
            weights (dict): 평가 지표별 가중치 설정
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.weights = weights or {
            "structural": 0.25,  # 구조적 평가 가중치
            "semantic": 0.5,  # 의미적 평가 가중치
            "domain": 0.25,  # 도메인 평가 가중치
        }

    def get_embeddings(self, text: str) -> np.ndarray:
        """텍스트의 임베딩 벡터 추출"""
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def evaluate_structural_coherence(self, dialog_data: Dict) -> Dict[str, float]:
        """대화의 구조적 일관성 평가"""
        scores = {}

        # 1. 시스템-사용자 발화 페어링 검사
        dialog = dialog_data["dialog"]
        if len(dialog["sys"]) != len(dialog["usr"]):
            scores["dialog_pairing"] = 0.0
        else:
            scores["dialog_pairing"] = 1.0

        # 2. 슬롯 값 연속성 검사
        slot_continuity = self.check_slot_continuity(dialog_data)
        scores["slot_continuity"] = slot_continuity

        # 3. 도메인별 슬롯 검증
        domain_slot_validity = self.validate_domain_slots(dialog_data)
        scores["domain_slot_validity"] = domain_slot_validity

        return scores

    def check_slot_continuity(self, dialog_data: Dict) -> float:
        """슬롯 값의 연속성 검사"""
        score = 10.0

        # last_slot_values가 현재 slot_values에 적절히 반영되었는지 검사
        if dialog_data["last_slot_values"]:
            for last_slot in dialog_data["last_slot_values"]:
                if not any(
                    current_slot["domain"] == last_slot["domain"]
                    and current_slot["slot"] == last_slot["slot"]
                    and (
                        current_slot["value"] == last_slot["value"]
                        or any(
                            turn_slot["domain"] == last_slot["domain"]
                            and turn_slot["slot"] == last_slot["slot"]
                            for turn_slot in dialog_data["turn_slot_values"]
                        )
                    )
                    for current_slot in dialog_data["slot_values"]
                ):
                    score -= 2.0

        # turn_slot_values가 slot_values에 반영되었는지 검사
        for turn_slot in dialog_data["turn_slot_values"]:
            if not any(
                current_slot["domain"] == turn_slot["domain"]
                and current_slot["slot"] == turn_slot["slot"]
                and current_slot["value"] == turn_slot["value"]
                for current_slot in dialog_data["slot_values"]
            ):
                score -= 2.0

        return max(score, 0.0)

    def validate_domain_slots(self, dialog_data: Dict) -> float:
        """도메인별 슬롯 유효성 검사"""
        score = 10.0
        declared_domains = set(dialog_data["domains"])

        # 모든 슬롯이 선언된 도메인에 속하는지 검사
        for slot in dialog_data["slot_values"]:
            if slot["domain"] not in declared_domains:
                score -= 2.0

        return max(score, 0.0)

    def evaluate_semantic_coherence(self, dialog_data: Dict) -> Dict[str, float]:
        """의미적 일관성 평가"""
        scores = {}

        # 1. 사용자 발화와 시스템 응답 간의 의미적 연관성
        coherence_scores = []
        for usr, sys in zip(dialog_data["dialog"]["usr"], dialog_data["dialog"]["sys"]):
            if usr and sys:
                usr_emb = self.get_embeddings(usr)
                sys_emb = self.get_embeddings(sys)
                similarity = float(cosine_similarity(usr_emb, sys_emb)[0][0])
                coherence_scores.append(similarity)

        scores["semantic_coherence"] = (
            np.mean(coherence_scores) if coherence_scores else 0.0
        )

        # 2. 슬롯 값 변경과 대화 내용의 일치성
        slot_dialogue_consistency = self.evaluate_slot_dialogue_consistency(dialog_data)
        scores["slot_dialogue_consistency"] = slot_dialogue_consistency

        return scores

    def evaluate_slot_dialogue_consistency(self, dialog_data: Dict) -> float:
        """슬롯 값 변경과 대화 내용의 일치성 평가"""
        score = 10.0

        # turn_slot_values의 변경사항이 대화 내용에 반영되었는지 검사
        for turn_slot in dialog_data["turn_slot_values"]:
            domain = turn_slot["domain"].lower()
            slot = turn_slot["slot"].lower()
            value = str(turn_slot["value"]).lower()

            # 시스템 응답에 변경된 슬롯 정보가 언급되었는지 확인
            mentioned = False
            for sys_utterance in dialog_data["dialog"]["sys"]:
                sys_utterance = sys_utterance.lower()
                if domain in sys_utterance and value in sys_utterance:
                    mentioned = True
                    break

            if not mentioned:
                score -= 2.0

        return max(score, 0.0)

    def evaluate_domain_coverage(self, dialog_data: Dict) -> Dict[str, float]:
        """도메인 커버리지 평가"""
        scores = {}
        declared_domains = set(dialog_data["domains"])
        used_domains = set()

        # 슬롯에서 사용된 도메인 확인
        for slot in dialog_data["slot_values"]:
            used_domains.add(slot["domain"])

        # 도메인 사용률
        scores["domain_usage"] = (
            len(used_domains) / len(declared_domains) if declared_domains else 0.0
        )

        # 도메인 간 상호작용 평가
        if len(declared_domains) > 1:
            domain_interaction_score = self.evaluate_domain_interaction(dialog_data)
            scores["domain_interaction"] = domain_interaction_score
        else:
            scores["domain_interaction"] = 1.0

        return scores

    def evaluate_domain_interaction(self, dialog_data: Dict) -> float:
        """도메인 간 상호작용 평가"""
        score = 0.0
        domains_mentioned = defaultdict(list)

        # 각 턴에서 언급된 도메인 기록
        for i, (usr, sys) in enumerate(
            zip(dialog_data["dialog"]["usr"], dialog_data["dialog"]["sys"])
        ):
            turn_domains = set()
            for domain in dialog_data["domains"]:
                if domain.lower() in usr.lower() or domain.lower() in sys.lower():
                    turn_domains.add(domain)
                    domains_mentioned[domain].append(i)

            # 한 턴에서 여러 도메인이 상호작용하면 점수 부여
            if len(turn_domains) > 1:
                score += 1.0

        return (
            min(score / len(dialog_data["dialog"]["usr"]), 1.0)
            if dialog_data["dialog"]["usr"]
            else 0.0
        )

    def evaluate_single_turn(self, dialog_data: Dict) -> Dict[str, float]:
        """단일 턴 종합 평가"""
        # 1. 구조적 평가
        structural_scores = self.evaluate_structural_coherence(dialog_data)

        # 2. 의미적 평가
        semantic_scores = self.evaluate_semantic_coherence(dialog_data)

        # 3. 도메인 평가
        domain_scores = self.evaluate_domain_coverage(dialog_data)

        # 카테고리별 평균 계산
        category_scores = {
            "structural_coherence": np.mean(
                [
                    structural_scores["dialog_pairing"],
                    structural_scores["slot_continuity"],
                    structural_scores["domain_slot_validity"],
                ]
            ),
            "semantic_coherence": np.mean(
                [
                    semantic_scores["semantic_coherence"]
                    * 1.5,  # 의미적 일관성 가중치 증가
                    semantic_scores["slot_dialogue_consistency"],
                ]
            ),
            "domain_coverage": np.mean(
                [domain_scores["domain_usage"], domain_scores["domain_interaction"]]
            ),
        }

        # 가중치가 적용된 전체 점수 계산
        weighted_score = (
            category_scores["structural_coherence"] * self.weights["structural"]
            + category_scores["semantic_coherence"] * self.weights["semantic"]
            + category_scores["domain_coverage"] * self.weights["domain"]
        )

        # 세부 점수도 함께 반환
        all_scores = {
            **structural_scores,
            **semantic_scores,
            **domain_scores,
            "category_scores": category_scores,
            "overall_score": weighted_score,
        }

        return all_scores

    def evaluate_dataset(
        self, dataset_path: str = None, dataset: List[Dict] = None
    ) -> Dict:
        """전체 데이터셋 평가

        Args:
            dataset_path (str, optional): 데이터셋 JSON 파일 경로
            dataset (List[Dict], optional): 데이터셋 객체

        Returns:
            Dict: 평가 결과
        """
        if dataset_path is not None:
            try:
                with open(dataset_path, "r", encoding="utf-8") as f:
                    dataset = json.load(f)
            except Exception as e:
                raise Exception(f"Error loading dataset from {dataset_path}: {str(e)}")

        if dataset is None:
            raise ValueError("Either dataset_path or dataset must be provided")

        all_scores = []
        domain_usage = defaultdict(int)
        semantic_scores_distribution = []

        for dialog in dataset:
            # 각 턴별 평가
            turn_scores = self.evaluate_single_turn(dialog)
            all_scores.append(turn_scores)

            # 도메인 사용 통계
            for domain in dialog["domains"]:
                domain_usage[domain] += 1

            # 의미적 점수 분포 추적
            if "semantic_coherence" in turn_scores:
                semantic_scores_distribution.append(turn_scores["semantic_coherence"])

        # 전체 통계 계산
        aggregate_scores = {
            metric: float(
                np.mean([score[metric] for score in all_scores if metric in score])
            )
            for metric in all_scores[0].keys()
            if metric != "category_scores"
        }

        # 의미적 평가 관련 추가 통계
        semantic_stats = {
            "mean": float(np.mean(semantic_scores_distribution)),
            "median": float(np.median(semantic_scores_distribution)),
            "std": float(np.std(semantic_scores_distribution)),
            "min": float(np.min(semantic_scores_distribution)),
            "max": float(np.max(semantic_scores_distribution)),
        }

        # 도메인 분포 계산
        total_turns = len(dataset)
        domain_distribution = {
            domain: count / total_turns * 100 for domain, count in domain_usage.items()
        }

        return {
            "turn_scores": all_scores,
            "aggregate_scores": aggregate_scores,
            "semantic_statistics": semantic_stats,
            "domain_distribution": domain_distribution,
            "total_turns": total_turns,
            "weights_used": self.weights,
        }

    def generate_report(
        self,
        dataset_path: str = None,
        dataset: List[Dict] = None,
        output_file: str = "dialog_quality_report.json",
    ):
        """
        평가 리포트 생성

        Args:
            dataset_path (str, optional): 데이터셋 JSON 파일 경로
            dataset (List[Dict], optional): 데이터셋 객체
            output_file (str): 결과를 저장할 파일 경로

        Returns:
            Dict: 평가 결과
        """
        if dataset_path is None and dataset is None:
            raise ValueError("Either dataset_path or dataset must be provided")

        if dataset_path is not None:
            try:
                with open(dataset_path, "r", encoding="utf-8") as f:
                    dataset = json.load(f)
            except Exception as e:
                raise Exception(f"Error loading dataset from {dataset_path}: {str(e)}")

        results = self.evaluate_dataset(dataset)

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Could not save report to {output_file}: {str(e)}")

        return results
