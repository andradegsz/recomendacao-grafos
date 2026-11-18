"""Recomendador colaborativo e híbrido baseado em passeios de comprimento 3.

Calcula a matriz de pontuações S = B·Bᵀ·B (grafo bipartido) ou
S = B·Bᵀ·B + alpha·(B·T·Gᵀ) (grafo tripartido), bloqueando itens já
avaliados com -inf. Quando executado diretamente, exibe as Top-K
recomendações para o primeiro usuário da base.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def pontuacao_bbtb(
    B: csr_matrix,
    T: csr_matrix | None = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Calcula a pontuação S para cada par (usuário, filme) e mascara vistos.

    Args:
        B: Matriz de biadjacência esparsa Usuários x Filmes (|U| x |I|).
        T: Matriz de biadjacência esparsa Filmes x Gêneros (|I| x |T|), opcional.
        alpha: Peso relativo dos caminhos que passam pelas tags (U -> I -> T -> I).

    Returns:
        Matriz densa S de dimensão (|U|, |I|) com -inf nos itens já avaliados.
    """
    S = B @ B.T @ B
    if T is not None and alpha > 0:
        S = S + alpha * (B @ T @ T.T)

    S = S.toarray()
    coo = B.tocoo()  # posições guardadas, incluindo pesos 0 (nota = média do usuário)
    S[coo.row, coo.col] = -np.inf
    return S


def recomendar_bbtb(
    B: csr_matrix,
    T: csr_matrix | None = None,
    alpha: float = 0.5,
    K: int = 10,
) -> np.ndarray:
    """Retorna os índices dos Top-K filmes recomendados para cada usuário.

    Args:
        B: Matriz de biadjacência esparsa Usuários x Filmes (|U| x |I|).
        T: Matriz de biadjacência esparsa Filmes x Tags (|I| x |T|), opcional.
        alpha: Peso relativo da componente tripartida (tags).
        K: Número de recomendações por usuário.

    Returns:
        Array de inteiros de dimensão (|U|, K) com os índices das colunas.
    """
    S = pontuacao_bbtb(B, T=T, alpha=alpha)
    return np.argsort(-S, axis=1)[:, :K]


if __name__ == "__main__":
    from matriz import carregar_dados, montar_matrizes

    ratings, tags, mapeamentos = carregar_dados()
    B, T = montar_matrizes(ratings, tags, mapeamentos)

    movies = pd.read_csv("data/ml-latest-small/movies.csv")
    filme_por_idx = {idx: mid for mid, idx in mapeamentos["filme"].items()}

    recs = recomendar_bbtb(B, T=T, alpha=0.5, K=10)
    primeiro_uid = next(iter(mapeamentos["usuario"]))
    u_idx = mapeamentos["usuario"][primeiro_uid]
    top_ids = [filme_por_idx[i] for i in recs[u_idx]]

    top_filmes = movies.set_index("movieId").loc[top_ids, ["title", "genres"]]
    print(f"Top-10 recomendações tripartidas para o usuário {primeiro_uid}:\n")
    print(top_filmes)