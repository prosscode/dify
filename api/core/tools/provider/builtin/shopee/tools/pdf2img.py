import io
import logging
import math
import time

import fitz
from PIL import Image

from core.file import FileType
from core.tools.tool.builtin_tool import BuiltinTool
from core.tools.entities.tool_entities import ToolInvokeMessage
from core.file.file_manager import download

from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


def get_average_font_size(page) -> float:
    font_sizes = []
    text = page.get_text("dict")
    for block in text["blocks"]:
        # 确保块中有行
        if "lines" in block:
            for line in block["lines"]:
                # 遍历行中的 span
                for span in line["spans"]:
                    font_sizes.append(span["size"])
    if len(font_sizes) == 0:
        return 0

    return sum(font_sizes) / len(font_sizes)


def check_for_vector_graphics(page):
    operators = page.get_drawings()
    if operators:
        return True
    return False


class Pdf2ImgTool(BuiltinTool):
    def _invoke(self,
                user_id: str,
                tool_parameters: Dict[str, Any],
                ) -> Union[ToolInvokeMessage, List[ToolInvokeMessage]]:
        """
            invoke tools
        """
        logger.info("run shopee pdf2img")
        file_variable = tool_parameters.get("file")
        vector_max_size = tool_parameters.get("vector_max_size")
        bit_max_size = tool_parameters.get("bit_max_size")
        quality = tool_parameters.get("quality")
        dpi = tool_parameters.get("dpi")
        direction = tool_parameters.get("direction")
        pages = tool_parameters.get("pages")

        logger.info(f'{file_variable}')
        # 不是pdf直接返回
        if file_variable.type != FileType.DOCUMENT:
            return self.create_file_message(file_variable)

        image_binary = download(file_variable)

        if not image_binary:
            return self.create_text_message("Image not found, please request user to generate image firstly.")

        pdf_stream = io.BytesIO(image_binary)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        if doc is None:
            pdf_stream.close()
            return self.create_text_message("open pdf failed")
        if doc.is_encrypted:
            if not doc.authenticate(""):
                pdf_stream.close()
                doc.close()

                return self.create_text_message("auth encrypted pdf with empty string failed")

        res = self.handle(doc, vector_max_size, bit_max_size, quality, direction, pages, dpi)
        if not res:
            pdf_stream.close()
            doc.close()
            return self.create_text_message("Pdf2Img error")

        results = [self.create_blob_message(blob=res, meta={"mime_type": "image/jpeg"}),
                   self.create_text_message(str(doc.page_count))]
        pdf_stream.close()
        doc.close()
        return results

    def handle(self, doc, vector_max_size, bit_max_size, quality, direction, pages, dpi=350):

        zoom = int(math.ceil(dpi / 72))
        matrix = fitz.Matrix(zoom, zoom)
        images = []

        target_pages = []
        if pages is not None and pages != "":
            for pageNo in pages.split(","):
                target_pages.append(doc.load_page(int(pageNo)))
        else:
            for pageNo in range(doc.page_count):
                target_pages.append(doc.load_page(pageNo))

        for page in target_pages:
            pix = page.get_pixmap(matrix=matrix)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            original_width, original_length = image.size
            font_size = get_average_font_size(page)
            # 矢量图
            if check_for_vector_graphics(page) and font_size > 0:
                ratio = min(vector_max_size / original_width, vector_max_size / original_length)
                if font_size < 9.0:
                    ratio = ratio / (font_size / 9.0)
            else:
                ratio = min(bit_max_size / original_width, bit_max_size / original_length)

            # 计算新的尺寸
            new_size = (int(original_width * ratio), int(original_length * ratio))
            image = image.resize(new_size, Image.LANCZOS)
            images.append(image)
        if direction == "horizontal":
            res = self.concat_images_horizontal(images, quality)
        elif direction == "vertical":
            res = self.concat_images_vertical(images, quality)
        else:
            new_images = [images[0]]
            res = self.concat_images_horizontal(new_images, quality)
        return res

    def concat_images_horizontal(self, images, quality):
        width = 0
        height = 0
        for image in images:
            cur_width, cur_height = image.size
            width += cur_width
            height = max(cur_height, height)

        result = Image.new('RGB', (width, height))
        last_image_width = 0
        for image in images:
            result.paste(image, (last_image_width, 0))
            tmp, _ = image.size
            last_image_width += tmp

        image_stream = io.BytesIO()
        result.save(image_stream, format='JPEG', quality=quality)
        image_binary_data = image_stream.getvalue()
        image_stream.close()
        return image_binary_data

    def concat_images_vertical(self, images, quality):
        width = 0
        height = 0
        for image in images:
            cur_width, cur_height = image.size
            height += cur_height
            width = max(cur_width, width)

        result = Image.new('RGB', (width, height))
        last_image_height = 0
        for image in images:
            result.paste(image, (0, last_image_height))
            _, tmp = image.size
            last_image_height += tmp

        image_stream = io.BytesIO()
        result.save(image_stream, format='JPEG', quality=quality)
        image_binary_data = image_stream.getvalue()
        image_stream.close()
        return image_binary_data


def test_func():
    doc = fitz.open("/Users/alan.li/Downloads/file (2)")
    # 创建一个字节流对象
    pdf_stream = io.BytesIO()

    # 将文档保存到字节流中
    doc.save(pdf_stream)

    # 获取二进制数据
    pdf_binary_data = pdf_stream.getvalue()

    res = Pdf2ImgTool().handle(pdf_binary_data)
    image_stream = io.BytesIO(res)

    # 打开图像
    image = Image.open(image_stream)
    # image.show()
