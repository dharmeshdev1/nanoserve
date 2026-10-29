import pytest
import torch
from microllm.model import GPT, ModelConfig


def make_model(seed=0, vocab=40, block=64, n_layer=2, n_embd=32, n_head=4, n_kv_head=2, scale=3.0):
    torch.manual_seed(seed)
    model = GPT(
        ModelConfig(
            vocab_size=vocab, block_size=block, n_layer=n_layer, n_head=n_head,
            n_kv_head=n_kv_head, n_embd=n_embd,
        )
    ).eval()
    with torch.no_grad():  # larger weights make logits well separated, so greedy has no near-ties
        for p in model.parameters():
            p.mul_(scale)
    return model


@pytest.fixture(scope="module")
def model():
    return make_model()
