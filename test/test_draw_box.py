import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PIL import Image

from llm_object.processor.image_processor import ImageProcessor
from llm_object.prompts.models.object_model import ObjectModels, ResponseObject

if __name__ == "__main__":
    infos = ResponseObject(
        objects=[
            ObjectModels(label="篮筐完整结构", box_2d=[517, 528, 718, 796]),
            ObjectModels(label="篮球", box_2d=[382, 664, 517, 816]),
        ]
    )
    image = Image.open(r"E:\桌面\demo\resource\test_image\VID_20250408_193434_frame_000090.jpg")
    image = ImageProcessor().draw_box(image, infos, show_image=True)
