import torch
from transformers import AutoModel, AutoTokenizer
from transformers import logging


from scipy.spatial.distance import cosine

logging.set_verbosity_error()

class SemanticSearch:
    """
    This class is used to perform semantic search on a given corpus of documents.

    Parameters
    ----------
    model : str
        The model to use for semantic search.
    
    Examples
    --------
    >>> from search import SemanticSearch
    >>> model = SemanticSearch("sgpt-small")
    >>> model.encode(["This is a test"], is_query=True)

    """


    def __init__(self, model="sgpt-small"):
        available = {
            "sgpt-large": "Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
            "sgpt-small": "Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit",
        }
        self.tokenizer = AutoTokenizer.from_pretrained(available[model])
        self._modify_tokenizer()
        self.model = AutoModel.from_pretrained(available[model])
        self.model.eval() # Deactivate Dropout 


        self.documents = None

    
    def _modify_tokenizer(self):
        """This function modifies the tokenizer to add special brackets"""
        self.SPECB_QUE_BOS = self.tokenizer.encode("[", add_special_tokens=False)[0]
        self.SPECB_QUE_EOS = self.tokenizer.encode("]", add_special_tokens=False)[0]
        self.SPECB_DOC_BOS = self.tokenizer.encode("{", add_special_tokens=False)[0]
        self.SPECB_DOC_EOS = self.tokenizer.encode("}", add_special_tokens=False)[0]


    def tokenize_with_specb(self, texts, is_query):
        # Tokenize without padding
        batch_tokens = self.tokenizer(texts, padding=False, truncation=True)   
        # Add special brackets & pay attention to them
        for seq, att in zip(batch_tokens["input_ids"], batch_tokens["attention_mask"]):
            if is_query:
                seq.insert(0, self.SPECB_QUE_BOS)
                seq.append(self.SPECB_QUE_EOS)
            else:
                seq.insert(0, self.SPECB_DOC_BOS)
                seq.append(self.SPECB_DOC_EOS)
            att.insert(0, 1)
            att.append(1)
        # Add padding
        batch_tokens = self.tokenizer.pad(batch_tokens, padding=True, return_tensors="pt")
        return batch_tokens

    def get_weightedmean_embedding(self, batch_tokens, model):
        # Get the embeddings
        with torch.no_grad():
            # Get hidden state of shape [bs, seq_len, hid_dim]
            last_hidden_state = model(**batch_tokens, output_hidden_states=True, return_dict=True).last_hidden_state

        # Get weights of shape [bs, seq_len, hid_dim]
        weights = (
            torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float().to(last_hidden_state.device)
        )

        # Get attn mask of shape [bs, seq_len, hid_dim]
        input_mask_expanded = (
            batch_tokens["attention_mask"]
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )

        # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
        sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

        embeddings = sum_embeddings / sum_mask

        return embeddings

    def encode(self, texts, is_query=False):
        # Tokenize the texts
        batch_tokens = self.tokenize_with_specb(texts, is_query)
        # Get the weighted mean embeddings
        embeddings = self.get_weightedmean_embedding(batch_tokens, self.model)
        return embeddings
    
    def query(self, query, top_k=5):
        return self.encode(query, is_query=True)

print("loading model...")
model = SemanticSearch("sgpt-small")

print("model loaded")

queries = [
    "Which planet is 40 light years away from Earth?",
]

docs = [
    "Neptune is the eighth and farthest-known Solar planet from the Sun. In the Solar System, it is the fourth-largest planet by diameter, the third-most-massive planet, and the densest giant planet. It is 17 times the mass of Earth, slightly more massive than its near-twin Uranus.",
    "TRAPPIST-1d, also designated as 2MASS J23062928-0502285 d, is a small exoplanet (about 30% the mass of the earth), which orbits on the inner edge of the wearable zone of the ultracool dwarf star TRAPPIST-1 approximately 40 light-years (12.1 parsecs, or nearly 3.7336×1014 km) away from Earth in the constellation of Aquarius.",
    "A harsh desert world orbiting twin suns in the galaxy’s Outer Rim, Tatooine is a lawless place ruled by Hutt gangsters. Many settlers scratch out a living on moisture farms, while spaceport cities such as Mos Eisley and Mos Espa serve as home base for smugglers, criminals, and other rogues.",
] 

# Encode the queries and documents

import time

start = time.time()
doc_embeddings = model.encode(docs)

end = time.time()
print("Time taken to encode the documents: ", end - start)

# print the amount of queries per seconds
print("Queries per second: ", len(docs) / (end - start))



start = time.time()
query_embeddings = model.query(queries)

end = time.time()
print("Time taken to encode the queries: ", end - start)

# print(query_embeddings)

# Calculate cosine similarities
# Cosine similarities are in [-1, 1]. Higher means more similar
cosine_sim_0_1 = 1 - cosine(query_embeddings[0], doc_embeddings[0])
cosine_sim_0_2 = 1 - cosine(query_embeddings[0], doc_embeddings[1])
cosine_sim_0_3 = 1 - cosine(query_embeddings[0], doc_embeddings[2])

print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (queries[0], docs[0][:20] + "...", cosine_sim_0_1))
print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (queries[0], docs[1][:20] + "...", cosine_sim_0_2))
print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (queries[0], docs[2][:20] + "...", cosine_sim_0_3))