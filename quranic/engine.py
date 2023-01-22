import torch
from transformers import AutoModel, AutoTokenizer
from transformers import logging

from InstructorEmbedding import INSTRUCTOR

logging.set_verbosity_error()


class SemanticSearch:
    """
    This class is used to perform semantic search on a given corpus of documents.

    Parameters
    ----------
    model : str
        The model to use for semantic search.
    device : str
        The device to use for inference.
        (cpu or cuda:0)

    Examples
    --------
    >>> from search import SemanticSearch
    >>> model = SemanticSearch("sgpt-small")
    >>> model.encode(["This is a test"], is_query=True)

    """

    def __init__(self, model="sgpt-tiny", device="cuda:0"):
        available = {
            "sgpt-large": "Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
            "sgpt-tiny": "Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit",
            "sgpt-small": "Muennighoff/SGPT-1.3B-weightedmean-msmarco-specb-bitfit",
            "sgpt-medium": "Muennighoff/SGPT-2.7B-weightedmean-msmarco-specb-bitfit",
        }
        self.tokenizer = AutoTokenizer.from_pretrained(available[model])
        self._modify_tokenizer()
        self.model = AutoModel.from_pretrained(available[model])
        self.model.to(device)
        self.model.eval()  # Deactivate Dropout

    def _modify_tokenizer(self):
        """This function modifies the tokenizer to add special brackets"""
        self.SPECB_QUE_BOS = self.tokenizer.encode("[", add_special_tokens=False)[0]
        self.SPECB_QUE_EOS = self.tokenizer.encode("]", add_special_tokens=False)[0]
        self.SPECB_DOC_BOS = self.tokenizer.encode("{", add_special_tokens=False)[0]
        self.SPECB_DOC_EOS = self.tokenizer.encode("}", add_special_tokens=False)[0]

    def tokenize_with_specb(self, texts, is_query):
        """This function tokenizes the texts and adds special brackets"""
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
        batch_tokens = self.tokenizer.pad(
            batch_tokens, padding=True, return_tensors="pt"
        ).to(self.model.device)
        return batch_tokens

    def get_weightedmean_embedding(self, batch_tokens, model):
        """This function computes the weighted mean embedding of the last hidden state"""
        # Get the embeddings
        with torch.no_grad():
            # Get hidden state of shape [bs, seq_len, hid_dim]
            last_hidden_state = model(
                **batch_tokens, output_hidden_states=True, return_dict=True
            ).last_hidden_state

        # Get weights of shape [bs, seq_len, hid_dim]
        weights = (
            torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
            .to(last_hidden_state.device)
        )

        # Get attn mask of shape [bs, seq_len, hid_dim]
        input_mask_expanded = (
            batch_tokens["attention_mask"]
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )

        # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
        sum_embeddings = torch.sum(
            last_hidden_state * input_mask_expanded * weights, dim=1
        )
        sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

        embeddings = sum_embeddings / sum_mask

        return embeddings

    def encode(self, texts, is_query=False):
        """This function encodes the texts into embeddings"""
        # Tokenize the texts
        batch_tokens = self.tokenize_with_specb(texts, is_query)
        # Get the weighted mean embeddings
        embeddings = self.get_weightedmean_embedding(batch_tokens, self.model)
        return embeddings


class CocoSearch:
    def __init__(self, device="cuda:0"):
        self.model = AutoModel.from_pretrained("OpenMatch/cocodr-large-msmarco")
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("OpenMatch/cocodr-large-msmarco")

    def encode(self, texts, is_query=False):
        with torch.no_grad():
            inputs = self.tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            ).to(self.model.device)
            embeddings = (
                self.model(**inputs, output_hidden_states=True, return_dict=True)
                .hidden_states[-1][:, :1]
                .squeeze(1)
            )
        return embeddings.to("cpu")


class Instructor:
    def __init__(self, device="cuda:0"):
        self.model = INSTRUCTOR("hkunlp/instructor-large")

    def encode(self, texts, is_query=False):
        return self.model.encode(texts)


if __name__ == "__main__":
    model = Instructor()
