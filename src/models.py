from typing import List, Dict, Any, Iterable

from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
from .utils import download_image_as_pil


class InteractionType(Enum):
    CLICK_OUT = "click_out"
    SAVED = "saved"



@dataclass
class UserDataset:
    user_id: str
    point_ids: List[str]
    payloads: List[Dict[str, Any]]
    images: List[Any] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        if len(self.images) == 0 and len(self.texts) == 0:
            return False
        elif len(self.images) > 0 and len(self.texts) > 0:
            return False
        else:
            return True
        
    def __len__(self) -> int:
        return len(self.point_ids)
    
    @classmethod
    def from_image_rows(cls, user_id: str, rows: Iterable) -> "UserDataset":
        point_ids, payloads, images = [], [], []
        
        for row in rows:
            image = download_image_as_pil(row["image_location"])
          
            if image:
                point_id = str(uuid4())
                point_ids.append(point_id)
                payloads.append(dict(row))
                images.append(image)
        
        if point_ids:
            return cls(
                user_id=user_id,
                point_ids=point_ids,
                payloads=payloads,
                images=images
            )
        
    @classmethod
    def from_text_rows(cls, user_id: str, rows: Iterable, min_text_size: int) -> "UserDataset":
        point_ids, payloads, texts = [], [], []
        
        for row in rows:
            if row.text and len(row.text.split()) > min_text_size:
                point_id = str(uuid4())
                point_ids.append(point_id)
                payloads.append(dict(row))
                texts.append(row.text)
        
        if point_ids:
            return cls(
                user_id=user_id,
                point_ids=point_ids,
                payloads=payloads,
                texts=texts,
            )