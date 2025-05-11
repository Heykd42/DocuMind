import chromadb
from chromadb.errors import InternalError
from chromadb.config import Settings
import os
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.embeddings.embedder import Embedder

class ChromaDBManager:
    def __init__(self, persist_dir="data/chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedder = Embedder()
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english")

    def _get_collection(self, document_id: str):
        name = f"doc_{document_id}"
        return self.client.get_or_create_collection(name=name)

    def add_pdf(self, pdf_path: str, document_id: str):
        coll = self._get_collection(document_id)
        chunks = self.embedder.load_pdf(pdf_path)
        texts = [c.get("text", c) if isinstance(c, dict) else c for c in chunks]
        # TF-IDF keywords
        tfidf = self.tfidf_vectorizer.fit_transform(texts)
        feat = np.array(self.tfidf_vectorizer.get_feature_names_out())
        metadatas = []
        for i in range(len(texts)):
            kws = feat[tfidf[i].toarray().argsort().flatten()[-5:]]
            metadatas.append({"keywords": " ".join(kws)})

        batch = {"ids": [], "embeds": [], "docs": [], "metas": []}
        for i, text in enumerate(tqdm(texts, desc="Embedding Chunks")):
            emb = self.embedder.get_embedding(text)
            if emb is None:
                continue
            batch["ids"].append(f"{document_id}_chunk_{i}")
            batch["embeds"].append(emb)
            batch["docs"].append(text)
            batch["metas"].append(metadatas[i])
            if len(batch["ids"]) >= 16:
                coll.add(
                    ids=batch["ids"], embeddings=batch["embeds"],
                    documents=batch["docs"], metadatas=batch["metas"]
                )
                batch = {"ids": [], "embeds": [], "docs": [], "metas": []}
        if batch["ids"]:
            coll.add(
                ids=batch["ids"], embeddings=batch["embeds"],
                documents=batch["docs"], metadatas=batch["metas"]
            )

    def get_documents(self, document_id: str):
        try:
            coll = self._get_collection(document_id)
            res = coll.get()
            return res.get("documents", [])
        except Exception:
            return []

    def hybrid_query(self, query_text: str, document_id: str = None, top_k: int = 5):
        query_emb = self.embedder.get_embedding(query_text)
        if query_emb is None:
            return [], []
        keywords = set(query_text.lower().split())
        docs, metas = [], []
        try:
            if document_id:
                coll = self._get_collection(document_id)
                if coll.count() > 0:
                    res = coll.query(query_embeddings=[query_emb], n_results=top_k)
                    docs, metas = res["documents"][0], res["metadatas"][0]
            else:
                for coll_obj in self.client.list_collections():
                    name = getattr(coll_obj, 'name', coll_obj)
                    coll = self.client.get_or_create_collection(name=name)
                    try:
                        part = coll.query(query_embeddings=[query_emb], n_results=top_k)
                    except InternalError as e:
                        logging.getLogger(__name__).error(f"HNSW read error on '{name}': {e}")
                        continue
                    docs.extend(part["documents"][0])
                    metas.extend(part["metadatas"][0])
        except InternalError as e:
            logging.getLogger(__name__).error(f"HNSW read error, skipping vector lookup: {e}")
            return [], []

        # TF-IDF keyword filter
        filtered_docs, filtered_meta = [], []
        for doc, md in zip(docs, metas):
            if keywords & set(md.get("keywords", "").split()):
                filtered_docs.append(doc)
                filtered_meta.append(md)
        return filtered_docs, filtered_meta
