"""Construção das matrizes de biadjacência esparsas do grafo tripartido.

Carrega os dados de avaliações (`ratings.csv`) e etiquetas (`tags.csv`) do
MovieLens e expõe as funções públicas `carregar_dados` e `montar_matrizes`
para construir as matrizes esparsas B (|U| x |I|) e T (|I| x |T|) alinhadas.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

def carregar_dados(
    dir: str = "data/ml-latest-small",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    """Lê os arquivos CSV e gera os mapeamentos globais fixos de índices.

    Args:
        dir: Caminho do diretório contendo `ratings.csv` e `tags.csv`.

    Returns:
        Tupla (ratings, tags, mapeamentos), onde `mapeamentos` contém os
        dicionários 'usuario', 'filme' e 'tag'.
    """
    ratings = pd.read_csv(f"{dir}/ratings.csv")
    tags = pd.read_csv(f"{dir}/tags.csv")

    tags_limpas = tags["tag"].dropna().str.strip().str.lower().unique()

    # Mapear IDs para índices para uso em matrizes
    mapeamentos = {
        "usuario": {uid: k for k, uid in enumerate(sorted(ratings["userId"].unique()))},
        "filme": {mid: k for k, mid in enumerate(sorted(ratings["movieId"].unique()))},
        "tag": {tag: k for k, tag in enumerate(sorted(tags_limpas))},
    }
    return ratings, tags, mapeamentos


def _montar_B(
    df: pd.DataFrame,
    idx_usuario: dict,
    idx_filme: dict,
) -> csr_matrix:
    """Constrói a matriz de biadjacência ponderada B (Usuários x Filmes)."""
    centrada = df["rating"] - df.groupby("userId")["rating"].transform("mean")
    norma = np.sqrt((centrada ** 2).groupby(df["userId"]).transform("sum"))
    pesos = (centrada / norma).fillna(0.0)

    linhas = df["userId"].map(idx_usuario)
    colunas = df["movieId"].map(idx_filme)

    return csr_matrix(
        (pesos, (linhas, colunas)),
        shape=(len(idx_usuario), len(idx_filme)),
    )


def _montar_T(
    df: pd.DataFrame,
    idx_filme: dict,
    idx_tag: dict,
) -> csr_matrix:
    """Constrói a matriz de biadjacência binária T (Filmes x Tags)."""
    pares = (
        pd.DataFrame({
            "i_idx": df["movieId"].map(idx_filme),
            "t_idx": df["tag"].str.strip().str.lower().map(idx_tag),
        })
        .dropna()
        .drop_duplicates()
    )

    valores = np.ones(len(pares), dtype=np.float64)

    return csr_matrix(
        (valores, (pares["i_idx"].astype(int), pares["t_idx"].astype(int))),
        shape=(len(idx_filme), len(idx_tag)),
    )


def montar_matrizes(
    ratings_df: pd.DataFrame,
    tags_df: pd.DataFrame,
    mapeamentos: dict[str, dict],
) -> tuple[csr_matrix, csr_matrix]:
    """Constrói em conjunto as matrizes B (Usuários x Filmes) e T (Filmes x Tags).

    Args:
        ratings_df: DataFrame com as colunas 'userId', 'movieId' e 'rating'.
        tags_df: DataFrame com as colunas 'movieId' e 'tag'.
        mapeamentos: Dicionário retornado por `carregar_dados`.

    Returns:
        Tupla (B, T) com as matrizes esparsas alinhadas pelo índice de filmes.
    """
    B = _montar_B(ratings_df, mapeamentos["usuario"], mapeamentos["filme"])
    T = _montar_T(tags_df, mapeamentos["filme"], mapeamentos["tag"])
    return B, T

if __name__ == "__main__":
    ratings, tags, mapeamentos = carregar_dados()
    B, T = montar_matrizes(ratings, tags, mapeamentos)
    print(f"Matriz B (Usuários x Filmes): shape={B.shape}, arestas={B.nnz}")
    print(f"Matriz T (Filmes x Tags):     shape={T.shape}, arestas={T.nnz}")