"""TODO:"""

from typing import List
from langchain_qdrant import Qdrant
from document import DocumentFactory
from preprocess import Preprocess
from folder import Folder
from vector_db import VectorDB


class IndexEngine:

    """
    TODO:
    """


    def add_texts_to_vector_store(self, text_chunkies: List[str],
                                 file_path: str, qdrant_client: Qdrant):
        """
        TODO:
        """
        total = len(text_chunkies)        
        metadata = [ {"path": file_path} for i in range(0, total)]


        print("Starting indexing...")
        qdrant_client.add_texts(texts=text_chunkies, metadatas=metadata)
        print(f"{total} texts were indexed")
        print("Finished indexing!")


    def document_indexing(self, folder_path, db_path, collection_name):
        """
        TODO:
        """

        print(f"Starting searching for files in the folder whith path: {folder_path}")

        folder = Folder()
        files = folder.get_files(folder_path=folder_path)
        folder_len = len(files)

        if len(files) == 0:

            print("The folder is empty. Nothing to do.")

        print(f"{folder_len} files were found")

        doc_factory = DocumentFactory()
        preprocess = Preprocess()
        data_base = VectorDB(path=db_path, device="mps") # Device -> cuda:0 or mps
        indexer = data_base.get_indexer_engine(collection_name=collection_name)

        print('Starting Indexing')

        for file_i in files:

            print(f'Starting {file_i} indexing...')

            doc_obj = doc_factory.get_doc_obj(file_obj=file_i)

            content = doc_obj.read()

            text_chunkies = preprocess.text_spliter(
                text_content=content,
                chunk_size=500,
                chunk_overlap=50
            )

            self.add_texts_to_vector_store(
                text_chunkies=text_chunkies,
                file_path=file_i, 
                qdrant_client=indexer
            )

        print("Indexing Concluded")




if __name__=="__main__":


    FOLDER_PATH = "/Users/egbertoaraujo/WorskSpace/R&D/LLM/generative_search_engine/TestFolder"
    DB_PATH = "/Users/egbertoaraujo/WorskSpace/R&D/LLM/generative_search_engine/qdrant"

    engine = IndexEngine()

    engine.document_indexing(
        folder_path=FOLDER_PATH,
        db_path=DB_PATH,
        collection_name="LocalCollection"
    )
