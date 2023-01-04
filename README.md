# Quran and Hadith Semantic Search

Using sentence embeddings to perform semantic text search on Quran and Hadith translations

In this project I used the [SGPT](https://arxiv.org/abs/2202.08904) model to generate embeddings for each verse/hadith. You can then easily search through the embeddings by checking vector similarity with a query.

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

