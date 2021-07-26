from pydantic import BaseModel


class Essay(BaseModel):
    essay: str
