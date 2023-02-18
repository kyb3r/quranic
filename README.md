# Quran and Hadith Semantic Search

Using sentence embeddings to perform semantic text search on Quran and Hadith translations

In this project I used a variety of sentence embedding models to generate embeddings for each verse/hadith. You can then easily search through the embeddings by checking vector similarity with a query. The best performing model was [Instructor Embedding](https://github.com/HKUNLP/instructor-embedding).

```py
from quranic.corpus import SearchEngine

quran = SearchEngine("quran")

scores, verses = quran.search(query, k=5)
for verse in verses:
    print(f"Surah {verse.surah.name}\n  {verse}\n")
```

## Installation
```
pip install git+https://github.com/kyb3r/quranic
```

