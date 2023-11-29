from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import List

import model

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    objects = jsonable_encoder([item])
    preprocessed = model.preprocess(objects)
    return model.predict(preprocessed)[0]


@app.post("/predict_items")
def predict_items(items: Items) -> List[float]:
    objects = jsonable_encoder(items.objects)
    preprocessed = model.preprocess(objects)
    return list(model.predict(preprocessed))
