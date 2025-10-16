# post example

import uvicorn
from fastapi import FastAPI

from typing import Optional, Union
from pydantic import BaseModel


class Item(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None


app = FastAPI()

items = [
    Item(id=1, name="Coke", price=10),
    Item(id=2, name="Pepsi", price=15),
    Item(id=3, name="7 UP", price=20),
]


def find_item_by_id(item_id: int) -> tuple[int, Union[Item, None]]:
    for idx, element in enumerate(items):
        if element.id == item_id:
            return idx, element

    return -1, None


@app.post("/items")
async def create_item(item: Item):
    idx, current_item = find_item_by_id(item.id)

    if current_item is not None:
        return {"message": f"item_id {item.id} is already existed"}

    items.append(item)

    return item


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    idx, item = find_item_by_id(item_id)

    if item is None:
        return {"message": f"item_id {item_id} does not exist"}

    return item


@app.get("/items")
async def list_items():
    return items


if __name__ == "__main__":
    uvicorn.run("post:app", host="0.0.0.0", port=8080, reload=True)
