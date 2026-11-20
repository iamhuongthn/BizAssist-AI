import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class KnowledgeRetriever:
    def __init__(self,knowledge,use_embeddings=True):
        self.docs=knowledge.reset_index(drop=True)
        self.texts=(self.docs.title.fillna('')+' '+self.docs.content.fillna('')).tolist()
        self.mode='tfidf'
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                self.matrix=self.model.encode(self.texts,normalize_embeddings=True,show_progress_bar=False)
                self.mode='embeddings'
            except (ImportError, OSError) as e:
                print('Embedding model unavailable, using TF-IDF fallback:',e)
        if self.mode=='tfidf':
            self.vectorizer=TfidfVectorizer(ngram_range=(1,2),stop_words='english')
            self.matrix=self.vectorizer.fit_transform(self.texts)

    def search(self,query,k=3):
        if self.mode=='embeddings':
            q=self.model.encode([query],normalize_embeddings=True)
            scores=np.asarray(q @ self.matrix.T).ravel()
        else:
            scores=cosine_similarity(self.vectorizer.transform([query]),self.matrix).ravel()
        indices=np.argsort(-scores,kind='stable')[:k]
        return [{**self.docs.iloc[int(j)].to_dict(),'score':float(scores[j])} for j in indices]
