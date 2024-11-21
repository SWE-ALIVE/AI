from sentence_transformers import SentenceTransformer
from scipy.spatial import KDTree
import numpy as np
import torch
import json
from typing import List
from app.data.create_dataset_openai import TurnData, Dialog


class KoreanDialogRetriever:
    def __init__(self, model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"):
        self.model = SentenceTransformer(model_name)
        self.dialogs: List[TurnData] = []
        self.embeddings = None
        self.kdtree = None

    def load_dialogs(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.dialogs = []
        for dialog in data:
            if not dialog["dialog"]["usr"] or not dialog["dialog"].get("sys"):
                continue

            self.dialogs.append(TurnData(**dialog))

        print(f"✅ {len(self.dialogs)}개의 대화 턴을 로드했습니다.")

    def _create_dialog_context(self, turn: TurnData) -> str:
        # 슬롯 값들을 정렬된 순서로 생성
        slot_values = []
        if turn.turn_slot_values:
            for sv in turn.turn_slot_values:
                slot_values.append(f"{sv.domain.lower()}-{sv.slot}: {sv.value}")

        # 컨텍스트를 예시 형식에 맞게 생성
        context_parts = [
            "[context] " + ", ".join(sorted(slot_values)),
            "[system] " + (turn.dialog.sys[-1] if turn.dialog.sys[-1] else ""),
            "Q: [user] " + turn.dialog.usr[-1],
        ]

        return "\n".join(context_parts)

    def _create_sql_query(self, turn: TurnData) -> str:
        # 해당 도메인 테이블에 대한 SQL 쿼리 생성
        domain = turn.domains[0].lower()
        sql = f"SELECT * FROM {domain}"

        # 슬롯 값이 있는 경우 WHERE 절 추가
        if turn.turn_slot_values:
            conditions = []
            for sv in turn.turn_slot_values:
                if isinstance(sv.value, (int, float)):
                    conditions.append(f"{sv.slot} = {sv.value}")
                else:
                    conditions.append(f"{sv.slot} = '{sv.value}'")

            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

        sql += ";"
        return sql

    def build_index(self):
        if not self.dialogs:
            raise ValueError("먼저 load_dialogs를 호출하여 대화 데이터를 로드해주세요.")

        print("임베딩 생성 중...")
        dialog_contexts = [self._create_dialog_context(turn) for turn in self.dialogs]

        with torch.no_grad():
            self.embeddings = self.model.encode(dialog_contexts, convert_to_numpy=True)

        self.embeddings = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )

        self.kdtree = KDTree(self.embeddings)
        print("✅ 검색 인덱스가 구축되었습니다.")

    def retrieve(self, query_turn: dict, top_k: int = 5) -> str:
        if self.kdtree is None:
            raise ValueError("먼저 build_index를 호출하여 검색 인덱스를 구축해주세요.")

        query = TurnData(
            ID="test",
            turn_id=1,  # 임시 turn_id
            domains=list(set(sv["domain"] for sv in query_turn["slot_values"])),
            dialog=Dialog(**query_turn["dialog"]),
            slot_values=query_turn["slot_values"],
            turn_slot_values=[],
            last_slot_values=[],
        )
        query_context = self._create_dialog_context(query)

        with torch.no_grad():
            query_embedding = self.model.encode(query_context, convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        distances, indices = self.kdtree.query(query_embedding.reshape(1, -1), k=top_k)

        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
            similar_turn = self.dialogs[idx]
            context = self._create_dialog_context(similar_turn)
            sql = self._create_sql_query(similar_turn)

            # 예시 형식에 맞게 출력 형식화
            example = f"""
Example #{i}
{context}
SQL: {sql}
"""
            results.append(example)

        return "\n".join(results)

    def format_examples(self, examples: List[dict]) -> str:
        """여러 예시를 포맷팅된 문자열로 변환"""
        formatted_examples = []
        for i, example in enumerate(examples, 1):
            formatted_example = f"""
Example #{i}
{self._create_dialog_context(example['turn'])}
SQL: {self._create_sql_query(example['turn'])}
"""
            formatted_examples.append(formatted_example)

        return "\n".join(formatted_examples)
