import fm
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.utils.sequences import dna_to_rna


def rna_fn(df: pd.DataFrame, model_path: str, batch_size: int = 128) -> pd.DataFrame:
    """Adds the amino acid sequence corresponding to the mutated nucleotides."""
    model, alphabet = fm.pretrained.rna_fm_t12(model_path)
    batch_converter = alphabet.get_batch_converter()

    torch.device("mps")
    model.to("mps")
    model.eval()

    model_input = [
        (str(i), dna_to_rna(seq)) for i, seq in enumerate(df["mutated_wildtype_dna"])
    ]

    mean_embeddings = []
    batch_size = 256

    for i in tqdm(range(0, len(model_input), batch_size)):
        batch = model_input[i : i + batch_size]
        _, _, batch_tokens = batch_converter(batch)

        with torch.no_grad():
            results = model(batch_tokens.to("mps"), repr_layers=[12])

        token_embeddings = results["representations"][12]
        mean_embeddings.append(token_embeddings.mean(axis=1).cpu().numpy())

    mean_embeddings = np.concatenate(mean_embeddings)
    df[[f"fn_{i}" for i in range(640)]] = mean_embeddings

    return df
