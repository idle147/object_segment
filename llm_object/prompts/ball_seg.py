from ..prompts.models.object_model import ObjectModels, ResponseObject
from .base_prompt import BasePrompt


class BallSegments(BasePrompt):
    def __init__(self, llm):
        super().__init__(llm, "object_detection.txt", ResponseObject)

    def run(self, image_data):
        if isinstance(image_data, list):
            input_info = {"image_data": image_data}
        else:
            input_info = {"image_data": [image_data]}
        rep_model = self.chain.invoke(input_info)
        return rep_model
