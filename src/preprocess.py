"""TODO:"""

from typing import List
from langchain_text_splitters import TokenTextSplitter


class Preprocess:

    """
    TODO:
    """
    def text_spliter(self, text_content, chunk_size, chunk_overlap) -> List[str]:
        """
        TODO:
        """

        splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        text_chunkies = splitter.split_text(text=text_content)

        return text_chunkies

