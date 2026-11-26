from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class TrainableGraniteTextTower(nn.Module):
    def __init__(self, encoder, tokenizer, context_length=256):
        super().__init__()
        self.encoder, self.tokenizer = encoder, tokenizer
        self.context_length = int(context_length)
        self.output_dim = int(encoder.config.hidden_size)
        if not 2 <= self.context_length <= int(encoder.config.max_position_embeddings):
            raise ValueError("Granite context_length exceeds the encoder's supported range")
        self.cls_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
        if self.cls_token_id is None or tokenizer.pad_token_id is None:
            raise ValueError("Granite tokenizer must define CLS and padding tokens")
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "left"
        self.encoder.config.reference_compile = False
        self._token_cache = {}

    @classmethod
    def from_pretrained(cls, model_path, context_length):
        model_path = Path(model_path).resolve()
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        config.reference_compile = False
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                                                  clean_up_tokenization_spaces=False)

        encoder, report = AutoModel.from_pretrained(
            model_path, config=config, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=torch.float32,
            attn_implementation="sdpa", output_loading_info=True)
        if any(report.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"Granite checkpoint is incomplete or incompatible: {report}")
        return cls(encoder, tokenizer, context_length)

    @staticmethod
    def _texts(texts):
        if isinstance(texts, str) or not len(texts):
            raise ValueError("Granite expects a nonempty batch of text strings")
        return [str(text).replace("[SEP]", "\n") for text in texts]

    def tokenize(self, texts):
        keys = self._texts(texts)
        missing = list(dict.fromkeys(key for key in keys if key not in self._token_cache))
        if missing:
            encoded = self.tokenizer(missing, padding=False,
                                     truncation=True, max_length=self.context_length)
            for index, key in enumerate(missing):
                self._token_cache[key] = {name: torch.tensor(encoded[name][index], dtype=torch.long)
                                          for name in ("input_ids", "attention_mask")}
        padded = {}
        for name, padding_value in (("input_ids", self.tokenizer.pad_token_id), ("attention_mask", 0)):
            sequences = [self._token_cache[key][name] for key in keys]
            padded[name] = pad_sequence(sequences, batch_first=True, padding_value=padding_value)
        return padded

    def encode_tokens(self, input_ids, attention_mask):
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape or input_ids.shape[0] == 0:
            raise ValueError("Granite input IDs and attention mask must have matching nonempty [batch, length] shapes")
        if input_ids.shape[1] > self.context_length:
            raise ValueError("Granite token batch exceeds context_length")
        mask = attention_mask.bool()
        if not mask.any(dim=-1).all():
            raise ValueError("Granite attention mask cannot contain an all-padding row")
        first = mask.long().argmax(-1)
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        if not input_ids[rows, first].eq(self.cls_token_id).all():
            raise ValueError("Granite requires CLS at the first unmasked token")
        device = next(self.encoder.parameters()).device
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        positions = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              position_ids=positions, return_dict=True).last_hidden_state
        first = first.to(device)
        features = hidden[torch.arange(hidden.shape[0], device=device), first].float()
        if features.shape != (input_ids.shape[0], self.output_dim):
            raise RuntimeError("Granite CLS output has an unexpected native dimension")
        if not torch.isfinite(features).all():
            raise FloatingPointError("Non-finite Granite CLS features; inspect the model input and weights")
        return features

    def forward(self, texts):
        tokens = self.tokenize(texts)
        return self.encode_tokens(tokens["input_ids"], tokens["attention_mask"])
