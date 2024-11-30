from app.retriever import KoreanDialogRetriever
from app.config.data_class import APIRequest

retriever = KoreanDialogRetriever()
retriever.load_dialogs("app/data/gpt4_dataset.json")
retriever.build_index()

query: APIRequest = {
    "dialog": {
        "sys": [
            "현재 공기청정기는 다음과 같은 모드가 있습니다: 자동, 수면, 터보. 어떤 모드로 설정해 드릴까요?"
        ],
        "usr": ["수면 모드로 설정해줘."],
    },
    "slot_values": [{"domain": "AIR_PURIFIER", "slot": "fan_speed", "value": "보통"}],
}


examples = retriever.retrieve(query, top_k=3)
print(examples)
