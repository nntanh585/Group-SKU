import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, HDBSCAN
import faiss
import json
import re
from typing import List, Dict, Any, Optional, Tuple

class ProductVectorManager:
    """
    Optimized manager for product vectors with batch clustering and FAISS retrieval.
    Supports:
    - DBSCAN/HDBSCAN for offline grouping
    - FAISS for fast online nearest neighbor (NN) lookup
    - Incremental updates via buffer
    - RAG prompt generation
    """

    def __init__(self,
                 embed_model_name: str,
                 eps_value: float = 0.2,
                 min_samples: int = 2,
                 use_hdbscan: bool = True,
                 similarity_threshold: float = 0.8):
        """
        Initializes the manager.

        Args:
            embed_model_name (str): S-BERT model name (e.g., 'paraphrase-multilingual-MiniLM-L12-v2')
            eps_value (float): Epsilon threshold for DBSCAN (if used)
            min_samples (int): Minimum samples (for DBSCAN / min_cluster_size for HDBSCAN)
            use_hdbscan (bool): True to use HDBSCAN (recommended), False for DBSCAN
            similarity_threshold (float): Similarity threshold (0.0 -> 1.0) for real-time assignment
        """
        self.embed_model_name = embed_model_name
        self.eps_value = eps_value
        self.min_samples = min_samples
        self.use_hdbscan = use_hdbscan
        self.similarity_threshold = similarity_threshold

        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None

        self._product_buffer: List[Tuple[pd.Series, np.ndarray]] = []

        self._load_model()

    # ----------------- MODEL -----------------
    def _load_model(self):
        print(f"Loading S-BERT model: {self.embed_model_name}...")
        self.model = SentenceTransformer(self.embed_model_name)
        print("Model loaded successfully.")

    # ----------------- DATA PREP -----------------
    def _read_json(self, json_file: str) -> dict:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _flatten_attributes(self, attr_products: Optional[dict]) -> dict:
        if not attr_products or 'data' not in attr_products:
            return {}
        result = {}
        for a in attr_products.get('data', []):
            key = a.get('attributes', {}).get('name') or f"attr_{a.get('attribute_id')}"
            value = a.get('attribute_values', {}).get('name') if a.get('attribute_values') else None
            if key and value:
                result[key] = value
        return result

    def _clean_text(self, text: str) -> str:
        if text is None:
            return ""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text) # remove punctuation
        text = re.sub(r"\s+", " ", text).strip() # remove extra whitespace
        return text

    def _transform_data_from_json(self, json_file: str) -> pd.DataFrame:
        products = self._read_json(json_file)['data']
        result_list = []

        for product in products:
            skus = product.get('product_skus', {}).get('data', [])
            sku_detail = skus[0].get('product_sku_detail', {}) if skus else {}

            product_info = {
                "product_id": product.get("id"),
                "product_name": product.get("name"),
                "brand_name": product.get("brand", {}).get("name"),
                "platform_name": product.get("shop", {}).get("platform_name"),
                "shop_country": product.get("shop", {}).get("country"),
                "short_description": product.get("short_description"),
                "price": sku_detail.get("price"),
                "quantity": sku_detail.get("quantity", 0),
                "attribute_products": {
                    "data": self._flatten_attributes(product.get("attribute_products"))
                },
            }
            result_list.append(product_info)
        return pd.DataFrame(result_list)

    #Can custome
    def _get_text_for_embedding(self, product_row: pd.Series) -> str:
        name = self._clean_text(product_row.get('product_name', ''))
        desc = self._clean_text(product_row.get('short_description', ''))
        brand = self._clean_text(product_row.get('brand_name', ''))

        natural_text = f"{name} by {brand}. {desc}"

        keyword_set = set()
        if name: keyword_set.add(name)
        if brand: keyword_set.add(brand)
        attributes_data = product_row.get('attribute_products', {}).get('data', {})
        if isinstance(attributes_data, dict):
            for key, value in attributes_data.items():
                if value:
                    keyword_set.add(self._clean_text(str(value)))

        keyword_text = " | ".join(sorted(list(keyword_set)))
        return f"Product: {natural_text}\nKeywords: {keyword_text}"

    # ----------------- BATCH BUILD -----------------
    def build_from_json(self, json_file: str):
        print(f"Building from {json_file}...")
        self.df = self._transform_data_from_json(json_file)
        texts = self.df.apply(self._get_text_for_embedding, axis=1).tolist()

        # Encode in batches and ensure float32
        self.embeddings = embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=64
        ).astype('float32')

        faiss.normalize_L2(embeddings)  # Normalize for Cosine/IP

        # ----------- CLUSTERING -----------
        if self.use_hdbscan:
            print("Using HDBSCAN for clustering...")
            clusterer = HDBSCAN(
                min_cluster_size=self.min_samples,
                metric='cosine'
            )
            clusters = clusterer.fit_predict(embeddings)
        else:
            print("Using DBSCAN for clustering...")
            dbscan = DBSCAN(
                eps=self.eps_value,
                min_samples=self.min_samples,
                metric='cosine'
            )
            clusters = dbscan.fit_predict(embeddings)

        self.df['group_sku_id'] = clusters
        self.df.to_csv('result.csv', index=False)

        # ----------- BUILD FAISS INDEX -----------
        d = embeddings.shape[1]
        nlist = int(self.df['group_sku_id'].max() + 1)
        print(f"Building FAISS IndexIVFFlat with nlist={nlist}...")

        quantizer = faiss.IndexFlatIP(d)  # coarse quantizer for IVF
        self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

        # Train before adding
        if not self.index.is_trained:
            print("Training FAISS index (coarse quantizer)...")
            self.index.train(embeddings)
            print("Training complete.")

        # Add all embeddings to IVF index
        self.index.add(embeddings)

        # Config search behavior
        self.index.nprobe = min(10, nlist)  # search in up to 10 clusters
        print(f"FAISS IVF Index built with {self.index.ntotal} vectors (nprobe={self.index.nprobe}).")


    # ----------------- CHECK -----------------
    def _check_ready(self):
        if self.index is None or self.df is None or self.model is None:
            raise Exception("Manager not ready. Run build_from_json() first.")

    # ----------------- ASSIGN NEW PRODUCT -----------------
    def assign_new_product(self, product_dict: Dict[str, Any]) -> int:
        """
        Assigns a Group SKU ID to a new product and adds it to the buffer.
        Flushes the buffer to the main index when it's full.
        """
        self._check_ready()
        new_series = pd.Series(product_dict)
        new_text = self._get_text_for_embedding(new_series)

        # Minor tweak: ensure encode output is float32
        new_vector = self.model.encode([new_text]).astype('float32')
        faiss.normalize_L2(new_vector)

        # Search FAISS
        D, I = self.index.search(new_vector, k=1)
        nearest_index = I[0][0]
        similarity = D[0][0]

        if similarity >= self.similarity_threshold:
            new_group_id = self.df.loc[nearest_index, 'group_sku_id']
        else:
            new_group_id = -1  # outlier

        new_series['group_sku_id'] = new_group_id

        # Batch update FAISS for performance
        self._product_buffer.append((new_series, new_vector))
        if len(self._product_buffer) >= 50:  # Buffer flush threshold
            print(f"Flushing {len(self._product_buffer)} products from buffer to index...")
            series_to_add = [item[0] for item in self._product_buffer]
            vectors_to_add = np.vstack([item[1] for item in self._product_buffer])

            # Add to FAISS (no re-encoding)
            self.index.add(vectors_to_add)

            # Add to DataFrame
            self.df = pd.concat([self.df, pd.DataFrame(series_to_add)], ignore_index=True)

            # Clear buffer
            self._product_buffer = []

        return int(new_group_id)

    def flush_buffer(self):
        """
        Manually flushes any remaining products in the buffer.
        Should be called before application shutdown.
        """
        if not self._product_buffer:
            print("Buffer is empty, no flush needed.")
            return

        print(f"Flushing {len(self._product_buffer)} remaining products from buffer...")
        series_to_add = [item[0] for item in self._product_buffer]
        vectors_to_add = np.vstack([item[1] for item in self._product_buffer])

        self.index.add(vectors_to_add)
        self.df = pd.concat([self.df, pd.DataFrame(series_to_add)], ignore_index=True)
        self._product_buffer = []
        print("Buffer flush complete.")

#     # ----------------- RAG PROMPT -----------------
#     def generate_rag_prompt(self, user_question: str, k: int = 5) -> str:
#         self._check_ready()

#         PROMPT_TEMPLATE = """
# Based on the following context:

# --- Context ---
# {context}
# --- End Context ---

# Please answer the user's question in a friendly manner, using only the information from the context.

# Question: {question}

# Your Answer:
# """
#         question_vector = self.model.encode([user_question]).astype('float32')
#         faiss.normalize_L2(question_vector)

#         D, I = self.index.search(question_vector, k=k)
#         context_strings = []
#         for idx in I[0]:
#             if idx < 0: continue # FAISS can return -1 if index is empty/small
#             product_row = self.df.iloc[idx]
#             context_strings.append(self._get_text_for_embedding(product_row))

#         context = "\n".join([f"- Related Product: {txt}" for txt in context_strings])
#         prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
#             context=context,
#             question=user_question
#         )
#         return context, prompt

EMBED_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
EPS_VALUE = 0.023 # The value you found from your k-distance plot
MIN_SAMPLES = 2

manager = ProductVectorManager(
    embed_model_name=EMBED_MODEL,
    eps_value=EPS_VALUE,
    min_samples=MIN_SAMPLES,
    use_hdbscan=True
)
manager.build_from_json('./example.json')

#ADD NEW Product
new_product_data = {
    'product_id': 541187,
    'product_name': 'Omni Item (đỏ)',
    'brand_name': 'No Brand',
    'platform_name': 'Preny Test',
    'shop_country': 'vn',
    'short_description': None,
    'price': 1000000,
    'quantity': 111106,
    'attribute_products': {'data': {'brand': 'No Brand',
    'warranty_type': 'Warranty Paper and Invoice',
    'is_hazardous': 'None'}}
}

try:
    assigned_id = manager.assign_new_product(new_product_data)
    print(f"Result:\nNew product has been assigned Group SKU ID: {assigned_id}")
except Exception as e:
    print(f"Error: {e}")
