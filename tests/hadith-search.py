from quranic.corpus import SearchEngine
import time


corpus = SearchEngine("bukhari")
# quran = SearchEngine("quran")

queries = ["use reasoning instead of following your forefather"]

# for query in queries:
#     print("QUERY:", query)
#     print()
#     results = quran.search(query, k=10)
#     for socre, hadith in results:
#         print(hadith)
#         print()
#     print("---------------------\n")

while True:
    query = input("QUERY: ")
    print()
    results = corpus.search(query, k=10)

    for _, hadith in results:
        print(hadith)
        print()
    print("---------------------\n")
