from typing import List, Optional, Union
from datetime import datetime

from pydantic import BaseModel, Field

class Movie(BaseModel):
    mid: int
    genre: str
    rate: Union[int, float]
    tag: Optional[str] = None
    date: Optional[datetime] = None
    some_variable_list : List[int] = []

class User(BaseModel):
    uid: int
    name: str = Field(min_length=2, max_length=7)
    age:int = Field(gt=1, le=130)

temp_data = {
    'mid': '1',
    'genre': 'action',
    'rate': '9.5',
    'tag': 'superhero',
    'date': '2021-07-31',
}

temp_user_data = {
    'uid': 100,
    'name': 'soojin',
    'age': 30
}

tmp_movie = Movie(**temp_data)
tmp_user_data = User(**temp_user_data)
print(tmp_movie)
print(tmp_user_data)