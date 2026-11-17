"""Recomendador baseline baseado na popularidade (grau dos filmes).

Implementa `recomendar_popularidade`, que ranqueia os filmes ainda não vistos
por cada usuário de acordo com seu grau no grafo (número total de avaliações
recebidas). Quando executado diretamente, exibe o Top-10 para o
primeiro usuário da base.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def recomendar_popularidade(B: csr_matrix, K: int = 10) -> np.ndarray:
    """Retorna os índices dos Top-K filmes mais populares não vistos por usuário.

    Args:
        B: Matriz de biadjacência esparsa Usuários x Filmes (|U| x |I|).
        K: Número de recomendações por usuário.

    Returns:
        Array de inteiros de dimensão (|U|, K) com os índices das colunas.
    """
    graus = B.getnnz(axis=0).astype(np.float64)
    scores = np.tile(graus, (B.shape[0], 1))
    scores[B.nonzero()] = -np.inf
    return np.argsort(-scores, axis=1)[:, :K]


if __name__ == "__main__":
    from matriz import carregar_dados, montar_matrizes

    ratings, tags, mapeamentos = carregar_dados()
    B, _ = montar_matrizes(ratings, tags, mapeamentos)

    movies = pd.read_csv("data/ml-latest-small/movies.csv")
    filme_por_idx = {idx: mid for mid, idx in mapeamentos["filme"].items()}

    recs = recomendar_popularidade(B, K=10)
    primeiro_uid = next(iter(mapeamentos["usuario"]))
    u_idx = mapeamentos["usuario"][primeiro_uid]
    top_ids = [filme_por_idx[i] for i in recs[u_idx]]

    top_filmes = movies.set_index("movieId").loc[top_ids, ["title", "genres"]]
    print(f"Top-10 recomendações por popularidade para o usuário {primeiro_uid}:\n")
    print(top_filmes)