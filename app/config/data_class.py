from pydantic import BaseModel
from typing import Dict, List


class Dialog(BaseModel):
    sys: list[str]
    usr: list[str]


class SlotValue(BaseModel):
    domain: str
    slot: str
    value: str


class DomainMessage(BaseModel):
    domain: str
    message: str


class TurnData(BaseModel):
    ID: str
    turn_id: int
    domains: list[str]
    dialog: Dialog
    slot_values: list[SlotValue]
    turn_slot_values: list[SlotValue]
    last_slot_values: list[SlotValue]


class APIRequest(BaseModel):
    dialog: Dict[str, List[str]]
    slot_values: list[SlotValue]


class APIResponse(BaseModel):
    sql: str
    message: list[DomainMessage]
    context: list[SlotValue]
