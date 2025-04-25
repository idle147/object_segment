import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from processor.image_processor import ImageProcessor
from prompts.ball_seg import BallSegments

sys.path.append(str(Path(__file__).resolve().parents[1]))
from loguru import logger

import utils
from config import GLOBAL_CONFIG


class ObjectRecognition:
    def __init__(self, output_dir, model_name="chatgpt", is_debug=False, max_num=2000):
        if model_name == "chatgpt":
            self.llm = ChatOpenAI(**GLOBAL_CONFIG.get_config())
        else:
            raise ValueError(f"不支持的模型: {model_name}")

        self.example_dir: Path = Path(__file__).parents[0] / "examples"
        self.max_num = max_num
        self.model_name = model_name
        self.is_debug = is_debug
        if self.is_debug is True:
            print("开启Debug模式")

        self.image_processor = ImageProcessor(max_width=128, max_height=128)
        self.ball_segments = BallSegments(self.llm)

        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

    def evaluate(self, target_image_path: Path):
        _, edited_img, _ = self.image_processor.load_image(target_image_path)
        edited_img_base64 = self.image_processor.get_base64(edited_img)
        image_info = self.get_image_info(edited_img_base64)
        response: AIMessage = self.ball_segments.run(image_info)
        # 将结果序列化为JSON字符串格式
        return response.content

    def get_image_info(self, edited_img_base64):
        content = [
            {
                "type": "text",
                "text": "下述图片是你需要识别的目标物体，请给出物体的名称和描述。",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/webp;base64,{edited_img_base64}"},
            },
        ]
        return HumanMessage(content=content)

    def run(self, dataset_path):
        # 读取json文件
        image_path_list = ImageProcessor.get_dataset(dataset_path)

        # 使用线程池并行处理任务
        ret = []
        if self.is_debug:
            # 单线程处理
            for image_path in image_path_list:
                ret.append(self.evaluate(image_path))
        else:
            # 多线程处理
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_path = {
                    executor.submit(self.evaluate, image_path): image_path for image_path in image_path_list
                }
                for future in tqdm(as_completed(future_to_path), total=len(future_to_path), desc="Processing images"):
                    img_path = future_to_path[future]
                    try:
                        result = future.result()
                        ret.append(result)
                    except Exception as e:
                        logger.error(f"Error processing {img_path}: {e}")
                        logger.error(traceback.format_exc())

        # 保存为json文件
        with open(self.output_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(ret, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # 创建实例并运行评估
    object_rec = ObjectRecognition("./output", is_debug=True)
    object_rec.run(r"E:\桌面\demo\output_keyframes")
