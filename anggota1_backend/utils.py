"""ANGGOTA 1 - Helper serialisasi dokumen MongoDB -> JSON."""
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import HTTPException


def serialize(doc: dict) -> dict:
    out = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            out[k] = str(v)
        elif hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


def to_object_id(value: str) -> ObjectId:
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="ID tidak valid")
