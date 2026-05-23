import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Try importing FAISS; if it fails, the system seamlessly falls back to our pure-numpy engine
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

class RAGEngine:
    def __init__(self, data_path: str = "outputs/inventory_recommendations.csv", model_name: str = "all-MiniLM-L6-v2"):
        self.data_path = data_path
        self.model_name = model_name
        
        # Load embedding model
        print(f"[RAG] Loading SentenceTransformer embedding model '{model_name}'...")
        self.encoder = SentenceTransformer(model_name)
        
        # Initialize storage
        self.documents = []
        self.metadata = []
        self.embeddings = None
        self.faiss_index = None
        
        # Ingest and index data
        self.ingest_data()

    def ingest_data(self):
        """
        Reads inventory predictions from the static CSV path and indexes it.
        """
        if not self.data_path or not os.path.exists(self.data_path):
            print(f"[RAG WARNING] Source data file '{self.data_path}' is missing. Initializing empty RAG index.")
            return
            
        df = pd.read_csv(self.data_path)
        self.build_index(df)

    def build_index(self, df: pd.DataFrame):
        """
        Reads a DataFrame of predictions and compiles textual documents ready for semantic embedding.
        """
        print(f"[RAG] Ingesting {len(df)} records for vector index...")
        
        docs = []
        meta = []
        
        # Select representative records (e.g. unique date-product combinations) to prevent excessive compute
        # We index a subset if dataset is too large, or index all 1245 test rows since they are fast to compute.
        df_subset = df.head(500) # Ingest top 500 records for responsive loading
        
        for idx, row in df_subset.iterrows():
            date_str = str(row.get('date', 'Unknown Date'))
            # Format clean date (just YYYY-MM-DD if full timestamp)
            if ' ' in date_str:
                date_str = date_str.split(' ')[0]
                
            product = row.get('product_name', 'FMCG Product')
            demand = row.get('predicted_demand', 0.0)
            net_stock = row.get('net_stock', 0.0)
            reorder = row.get('reorder_required', 0)
            reorder_qty = row.get('safety_adjusted_reorder_qty', 0)
            status = row.get('inventory_status', 'Normal')
            risk = row.get('stock_risk_level', 'Low Risk')
            explanation = row.get('forecast_explanation', 'Demand is stable.')
            
            # Construct a descriptive, feature-rich textual log
            doc = (
                f"Product: {product} | Date: {date_str} | "
                f"Predicted Demand: {demand:.1f} units | Current Net Stock: {net_stock:.1f} units | "
                f"Inventory Status: {status} ({risk}) | Reorder Recommendation: "
                f"{'Required' if reorder == 1 else 'Not Required'} with safety adjusted quantity of {reorder_qty} units. "
                f"Operational Analysis: {explanation}"
            )
            
            docs.append(doc)
            meta.append({
                "date": date_str,
                "product_name": product,
                "predicted_demand": float(demand),
                "net_stock": float(net_stock),
                "reorder_required": bool(reorder),
                "safety_adjusted_reorder_qty": int(reorder_qty),
                "inventory_status": status,
                "stock_risk_level": risk,
                "forecast_explanation": explanation
            })
            
        self.documents = docs
        self.metadata = meta
        
        if docs:
            # Generate dense embedding vectors
            print(f"[RAG] Generating vector embeddings for {len(docs)} records (this may take a few seconds)...")
            self.embeddings = self.encoder.encode(docs, show_progress_bar=False)
            
            # Setup Indexing Engine
            if HAS_FAISS:
                print("[RAG] Indexing using native FAISS vector engine...")
                dimension = self.embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatL2(dimension)
                # Convert to float32 for FAISS alignment
                self.faiss_index.add(np.array(self.embeddings).astype('float32'))
            else:
                print("[RAG] FAISS library not found. Seamlessly falling back to robust Pure-Numpy Cosine Similarity search engine.")
        else:
            self.embeddings = None
            self.faiss_index = None
            print("[RAG] Empty dataset loaded. No embeddings generated.")

    def search(self, query: str, k: int = 3, user_id: int = None) -> list:
        """
        Performs semantic search across indexed logs to retrieve contextually matching insights.
        """
        if not self.documents or self.embeddings is None:
            return [{
                "document": "No records loaded in RAG index.",
                "score": 0.0,
                "metadata": {}
            }]
            
        # Encode search query into vector space
        query_vector = self.encoder.encode([query])[0]
        
        results = []
        
        if HAS_FAISS and self.faiss_index is not None:
            # Execute FAISS L2 flat distance search
            query_arr = np.array([query_vector]).astype('float32')
            distances, indices = self.faiss_index.search(query_arr, k)
            
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents) and idx >= 0:
                    # Map L2 distance to simulated similarity score
                    score = float(1.0 / (1.0 + distances[0][i]))
                    results.append({
                        "document": self.documents[idx],
                        "score": score,
                        "metadata": self.metadata[idx]
                    })
        else:
            # Execute Pure-Numpy Cosine Similarity vector search
            # Compute cosine similarity: (A . B) / (||A|| * ||B||)
            query_norm = np.linalg.norm(query_vector)
            norms = np.linalg.norm(self.embeddings, axis=1)
            
            # Handle divide by zero edge cases
            norms = np.where(norms == 0, 1e-9, norms)
            query_norm = query_norm if query_norm > 0 else 1e-9
            
            dot_products = np.dot(self.embeddings, query_vector)
            similarities = dot_products / (norms * query_norm)
            
            # Sort indices based on similarity score in descending order
            top_k_indices = np.argsort(similarities)[::-1][:k]
            
            for idx in top_k_indices:
                results.append({
                    "document": self.documents[idx],
                    "score": float(similarities[idx]),
                    "metadata": self.metadata[idx]
                })
                
        # Persist queries and results to rag_query_log database table
        from database_setup import get_connection, log_rag_query
        conn = get_connection()
        try:
            top_docs = [res.get("metadata", {}).get("product_name", "Unknown Product") for res in results]
            response_text = "\n\n".join([res["document"] for res in results])
            log_rag_query(conn, user_id or 1, query, top_docs, response_text)
        except Exception as e:
            print(f"[RAG DB Log Error] Failed to log query: {str(e)}")
        finally:
            conn.close()
            
        return results
