# put example

import uvicorn
from fastapi import FastAPI

from typing import Optional, Literal
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


def find_item_by_id(item_id: int) -> tuple[int, Item] | tuple[Literal[-1], None]:
    for idx, element in enumerate(items):
        if element.id == item_id:
            return idx, element

    return -1, None


@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    item_index, current_item = find_item_by_id(item_id)

    if current_item is None:
        return {"message": f"item_id {item.id} does not exist"}

    items[item_index] = item

    return item


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    item_index, item = find_item_by_id(item_id)

    if item is None:
        return {"message": f"item_id {item_id} does not exist"}

    return item


@app.get("/items")
async def list_items():
    return items


if __name__ == "__main__":
    uvicorn.run("put:app", host="0.0.0.0", port=8000, reload=True)
