import io
import json
import logging

from core.file import FileType
from core.tools.tool.builtin_tool import BuiltinTool
from core.tools.entities.tool_entities import ToolInvokeMessage
from core.file.file_manager import download

from typing import Any, Dict, List, Union
from pdf2image import convert_from_path, convert_from_bytes

logger = logging.getLogger(__name__)


class Pdf2ImgTool(BuiltinTool):
    def _invoke(self,
                user_id: str,
                tool_parameters: Dict[str, Any],
                ) -> Union[ToolInvokeMessage, List[ToolInvokeMessage]]:
        """
            invoke tools
        """
        logger.info("run shopee pdfutils")
        files_variable = tool_parameters.get("files")
        configs = tool_parameters.get("configs")
        config_map = {}
        if configs is not None and configs != "":
            config_map = json.loads(configs)

        results = []
        for file_variable in files_variable:
            logger.info(f'{file_variable}')
            # 不是pdf直接返回
            if file_variable.type != FileType.DOCUMENT:
                return self.create_file_message(file_variable)

            image_binary = download(file_variable)
            config_map["pdf_file"] = image_binary
            images = convert_from_bytes(**config_map)
            image_stream = io.BytesIO()
            images[0].save(image_stream, format='JPEG')
            results.append(self.create_blob_message(blob=image_stream.getvalue(), meta={"mime_type": "image/jpeg"}))
        return results
