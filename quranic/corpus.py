from .engine import SemanticSearch, CocoSearch
import numpy as np
import torch
import pickle
from functools import cache
from pathlib import Path

DATA = Path(__file__).parent / "data"

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class SearchEngine:
    """This class is used to store a corpus of documents and perform semantic search on it."""

    def __init__(self, name: str):
        books = {
            "quran": Quran,
            "bukhari": HadithCollection,
        }

        self.name = name
        self.book = books[name](name)
        self.documents = self.book.documents
        self.doc_embeddings: list = []
        self.load_search_engine()
    
    def load_search_engine(self, name="sgpt-small"):
        self.model = SemanticSearch(name)
        self.load_embeddings(DATA / f"{self.name}-embeddings")
        return self
    
    @staticmethod
    def sim_matrix(a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    @cache
    def search(self, query: str, k: int=5):
        """Search the corpus for the given query and return the top k results."""
        
        query_embedding = self.model.encode([query], is_query=True).cpu()
        similarities = self.sim_matrix(self.doc_embeddings, query_embedding)
        zipped = list(zip(similarities.tolist(), self.documents))

        return sorted(zipped, key=lambda x: x[0], reverse=True)[:k]
    
    def _deprecated_old_search(self, query: str, k: int = 5):
        """Search the corpus for the given query and return the top k results."""
        query_embedding = self.model.encode([query], is_query=True)[0].cpu()

        results = []

        for i, doc_embedding in enumerate(self.doc_embeddings):

            score = 1 - cosine_similarity(query_embedding, doc_embedding.cpu())
            results.append((score, self.documents[i]))
        
        sorted_results = sorted(results, key=lambda x: x[0])

        return sorted_results[:k]

    def batch_add_documents(self, documents: list):
        """Generate embeddings for the documents and add them to the corpus."""
        embeddings = self.model.encode(documents, is_query=False)
        self.doc_embeddings.extend(embeddings)
    
    def add_document(self, document):
        """Generate embeddings for the documents and add them to the corpus."""
        embedding = self.model.encode([document], is_query=False)[0]
        self.doc_embeddings.append(embedding)

    
    def save_embeddings(self, path: str):
        """Save precomputed embeddings to a file."""

        if isinstance(self.doc_embeddings, list):
            self.doc_embeddings = torch.stack([t.cpu() for t in self.doc_embeddings])
            print("Converted to torch tensor")

        with open(path, "wb") as f:
            pickle.dump(self.doc_embeddings, f)
    
    def load_embeddings(self, path: str):
        """Load precomputed embeddings from a file."""
        with open(path, "rb") as f:
            x = pickle.load(f)
            self.doc_embeddings = x

class Surah:
    with open(DATA / "surah_names.txt") as f:
        names = [x.strip() for x in f.readlines()]
    
    def __init__(self, name, number):
        self.name = name.split(".")[-1].strip()
        self.number = number
        self.verses = []
        
    def __repr__(self) -> str:
        return self.name
    
    def __eq__(self, __o: object) -> bool:
        return self.name == __o.name
    

class Verse:
    def __init__(self, translation):
        self.raw_translation = translation
        chapter_number, number, text = translation.strip().split("|")
        self.translation = text
        self.number = int(number)
        self.chapter_name = Surah.names[int(chapter_number)-1]
        self.surah = Surah(self.chapter_name, chapter_number)

    def __repr__(self):
        return f"({self.surah.number}:{self.number}) {self.translation}"
    

class Quran:
    def __init__(self, _=None):
        with open(DATA / "translation-clear-quran.txt") as f:
            self.verses = [Verse(x) for x in f.readlines()]
        self.surahs = []
        self.add_surahs()
        self.documents = self.verses
    
    def add_surahs(self):
        current_surah = self.verses[0].surah

        for verse in self.verses:
            if verse.surah != current_surah:
                self.surahs.append(current_surah)
                current_surah = verse.surah
            verse.surah = current_surah
            current_surah.verses.append(verse)
        
        self.surahs.append(current_surah)

class Hadith:
    def __init__(self, text):
        self.raw_text = text
        number, text = text.split("|")
        self.text = text.strip()
        self.number = number
    
    def __repr__(self):
        return self.raw_text

class HadithCollection:
    def __init__(self, name):
        self.name = name
        with open(DATA / f"{name}.txt") as f:
            raw_hadiths = f.readlines()
        self.hadiths = [Hadith(x.strip()) for x in raw_hadiths]
        self.documents = self.hadiths
    
    def __repr__(self):
        return self.name



if __name__ == "__main__":
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    corpus = SearchEngine("bukhari")

    from tqdm import tqdm

    with tqdm(total=len(documents)) as bar:
        for chunk in all_chunks:
            bar.set_description(f"{chunk[0].number}")
            corpus.batch_add_documents([str(v)[:2000] for v in chunk])
            bar.update(len(chunk))

    # save embeddings to file:
    corpus.save_embeddings("bukhari-embeddings")
    results = corpus.search("deaf dumb and blind")
    print(results)