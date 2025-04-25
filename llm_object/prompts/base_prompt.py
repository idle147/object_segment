import asyncio
from abc import ABC, abstractmethod
from pathlib import Path

import markdown
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate


class BasePrompt(ABC):
    def __init__(self, llm, prompt_path, pydantic_object=None):
        # 获取当前py文件的path
        prompt_path = Path(prompt_path)
        if prompt_path.exists():
            self.prompt_path = prompt_path
        else:
            self.prompt_path = Path(__file__).parent / prompt_path
        assert self.prompt_path.exists(), f"Prompt file [{self.prompt_path}] does not exist"

        self.chat_template = None
        self.llm = llm

        # Load the prompt template from the file
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            file_content = f.read()
            # 判断图像的后缀， 如果是以md结尾则读取文本内容为Markdown
            if prompt_path.suffix == ".md":
                system_msg = markdown.markdown(file_content)
            else:
                sections = file_content.split("###")
                system_msg = sections[-1]

        self.system_msg = system_msg
        if pydantic_object:
            self.parser = PydanticOutputParser(pydantic_object=pydantic_object)
            self.template = self.load_template(self.parser)
            self.chain = self.template | self.llm | self.parser
        else:
            self.template = self.load_template()
            if self.template:
                self.chain = self.template | self.llm

    def load_template(self, output_parser: PydanticOutputParser = None):
        placeholders = [
            ("system", self.system_msg),
            ("system", "{format_instructions}"),
            ("placeholder", "{image_data}"),
        ]
        partial_vars = {
            "format_instructions": output_parser.get_format_instructions() if output_parser is not None else None
        }
        chat_template = ChatPromptTemplate(placeholders, partial_variables=partial_vars)
        return chat_template

    @abstractmethod
    def run(self, image_info, captions, *args, **kwargs):
        pass
