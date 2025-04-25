from pydantic import BaseModel, Field


class ObjectModels(BaseModel):
    label: str = Field(..., description="The label of the object model.")
    box_2d: list[float] = Field(..., description="The bounding box coordinates of the object model.")


class ResponseObject(BaseModel):
    objects: list[ObjectModels] = Field(
        ..., description="List of detected objects with their labels and bounding boxes."
    )
