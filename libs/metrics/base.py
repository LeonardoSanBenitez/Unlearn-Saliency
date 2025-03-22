from pydantic import BaseModel, ConfigDict
from abc import ABC


class Metric(BaseModel, ABC):
    __abstract__ = True  # Explicitly mark as abstract
    model_config = ConfigDict(arbitrary_types_allowed=True)
