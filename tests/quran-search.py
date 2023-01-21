from quranic.corpus import SearchEngine

print("Loading model...")
corpus = SearchEngine("quran")

print(len(corpus.doc_embeddings))
print("Model loaded")


queries = [
    "its not the eyes that are blind, but the hearts",
    "deaf dumb and blind",
    "respecting and being dutiful to parents",
    "lusting after men",
]

import time

start = time.time()

for query in queries:
    print("QUERY:", query)
    print()
    results = corpus.search(query, k=5)
    for i, result in enumerate(results):
        verse = result[1]
        print(f"Surah {verse.surah.name}\n  {verse}")
        print()
    print("---------------------\n")

print(time.time() - start)


while True:
    query = input("QUERY: ")
    results = corpus.search(query, k=10)

    for _, verse in results:
        print(f"Surah {verse.surah.name}\n  {verse}")
        print()
    print("---------------------\n")
