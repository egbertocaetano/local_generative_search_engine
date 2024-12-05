"""TODO:"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings



class VectorDB():
    """
    TODO:
    """
    path: str
    client: QdrantClient
    hugging_face_bge_embedding_config: dict


    def __init__(self,
                 path,
                 model_name='sentence-transformers/msmarco-bert-base-dot-v5',
                 device='cpu',
                 encode_kwargs={"normalize_embeddings": True}) -> None:
        """
        TODO:
        """

        if encode_kwargs is None:

            encode_kwargs = {"normalize_embeddings": True}

        if model_name is None:

            model_name='sentence-transformers/msmarco-bert-base-dot-v5'

        if device is None:

            device='cpu'

        self.path = path
        self.client = QdrantClient(path=self.path)
        self.hugging_face_bge_embedding_config = {
            "model_name": model_name,
            "model_kwargs": {"device": device},
            "encode_kwargs":encode_kwargs
        }


    def get_indexer_engine(self, collection_name) -> Qdrant:
        """
        TODO:
        """

        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)

        vec_params = VectorParams(
            size=768,
            distance=Distance.DOT
        )

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=vec_params
        )


        hugging_face_bge_embedding = HuggingFaceBgeEmbeddings(
            model_name=self.hugging_face_bge_embedding_config["model_name"],
            model_kwargs=self.hugging_face_bge_embedding_config["model_kwargs"],
            encode_kwargs=self.hugging_face_bge_embedding_config["encode_kwargs"]
        )

        qdrant_obj = Qdrant(
            client=self.client,
            collection_name=collection_name,
            embeddings=hugging_face_bge_embedding
        )

        return qdrant_obj
