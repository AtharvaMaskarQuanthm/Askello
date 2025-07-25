from config.faiss import FaissSetup
from sentence_transformers import SentenceTransformer
from time import time
from typing import Dict, List
from utils.logger import get_logger

import faiss
import numpy as np

class FaissRAG:
    def __init__(self, data: List[str]) -> None:
        """
        This is a constructor method for FaissRAG Engine

        Parameters:
            - data (List) : List of data

        Returns:
            - None
        """

        # prepare logger
        self.logger = get_logger("FaissRAG")

        self.model = SentenceTransformer(FaissSetup.model_name)
        self.logger.info(":Model Loaded Sucessfully")

        self.data = data

        # Setting up the index
        self.index = self._setup(
            model = self.model, 
            data = data
        )
        self.logger.info(f":FAISS Index setup complete")


    def search(self, query: str, top_k : int = 3) -> List[str]:
        """
        This function performs search based on the query

        Parameters:
            - query (str) -> Query you wanna search

        Returns:
            - results (List) -> List of results. 
        """

        start_time = time()

        query_vector = self.model.encode([query], convert_to_numpy=True).astype('float32')

        _, I = self.index.search(query_vector, top_k)

        results = [self.data[i] for _, i in enumerate(I[0])]

        self.logger.info(f"FAISS Search latency: {time() - start_time}s")

        return results

    @staticmethod
    def _setup(model: SentenceTransformer, 
               data: List[str], 
               ) -> List[Dict]:
        """
        This function sets up the RAG system. 

        Parameters:
            - model (SenteceTransformer) - model used to encode
        """

        # 1. Create vectors 
        vectors = model.encode(data, convert_to_numpy=True).astype('float32')

        # 2. Built FAISS IVF index
        quantizer = faiss.IndexFlatL2(FaissSetup.vector_embedding_dimension)  # base index used for clustering
        index = faiss.IndexIVFFlat(quantizer, FaissSetup.vector_embedding_dimension, FaissSetup.nlist, faiss.METRIC_L2)

        index.train(vectors)
        index.add(vectors)
        index.nprobe = FaissSetup.nprobe

        print(f":FAISS IVF index created successfully")

        return index


