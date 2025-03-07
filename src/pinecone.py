from typing import List, Dict, Tuple
import pinecone
from datetime import datetime


DEFAULT_ID_FIELD_VALUE = "NULL"


def prepare(
    point_ids: List[str], 
    payloads: List[Dict],
    embeddings: List[List[float]]
) -> Tuple[List[Dict], Dict]:
    item_index, vectors, rows = [], [], []

    for point_id, payload, embedding in zip(point_ids, payloads, embeddings):
        item_id = payload.get("vinted_id") 
        query_id = payload.get("query_id")

        if query_id or (item_id and item_id not in item_index):
            row = _create_row(point_id, payload)
            vector = _create_vector(point_id, payload, embedding)

            rows.append(row)
            vectors.append(vector)
            item_index.append(item_id)

    return vectors, rows


def upload(index: pinecone.Index, vectors: List[Dict], namespace: str) -> bool:
    if len(vectors) == 0:
        return False

    try:
        index.upsert(vectors=vectors, namespace=namespace)
        return True
    except:
        return False


def _create_vector(point_id: str, payload: Dict, embedding: List[float]) -> Dict:
    for id_field in ["item_id", "query_id"]:
        if not payload.get(id_field):
            payload[id_field] = DEFAULT_ID_FIELD_VALUE

    return {"id": point_id, "values": embedding, "metadata": payload}


def _create_row(point_id: str, payload: Dict) -> Dict:
    row = {
        "id": point_id,
        "created_at": datetime.now().isoformat()
    }

    for field in ["user_id", "item_id", "query_id"]:
        row[field] = payload.get(field)

    return row