import numpy as np
import pandas as pd

from matriz import B, usuarios, filmes, idx_filme


def pontuacao_bbtb(B):
    """Matriz de pontuações S = B·Bᵀ·B (|U| x |I|), com os filmes já vistos zerados.

    S[u,i] = nº de passeios de comprimento 3 de u até i.
    """
    S = B @ B.T @ B       # B·Bᵀ·B, da esquerda para a direita (B·Bᵀ é só |U| x |U|)
    S = S.toarray()       # S é quase toda não nula (~99%), então densa é mais eficiente
    S[B.nonzero()] = 0    # zera S[u,i] para cada aresta (u,i): filmes já vistos
    return S


def recomendar_bbtb(B, K=10):
    """Top-K por B·Bᵀ·B. Devolve uma matriz (|U|, K) com os índices das colunas recomendadas."""
    S = pontuacao_bbtb(B)
    # Para cada linha, os índices das colunas em ordem decrescente de pontuação
    return np.argsort(-S, axis=1)[:, :K]


if __name__ == "__main__":
    K = 10
    S = pontuacao_bbtb(B)
    top_k = recomendar_bbtb(B, K)   # formato (|U|, K)

    print(S.shape)
    print(S[0, idx_filme[1]])   # usuário 1 viu o filme 1 -> tem que dar 0

    # Traduz índice de coluna -> movieId -> título, para mostrar o resultado
    movies = pd.read_csv("data/ml-latest-small/movies.csv")
    titulo = dict(zip(movies["movieId"], movies["title"]))

    u = 0   # linha 0 = userId 1
    print(f"\nTop-{K} para o usuário {usuarios[u]}:")
    for posicao, i in enumerate(top_k[u], start=1):
        print(f"{posicao:2d}. {titulo[filmes[i]]}  (pontuação {S[u, i]})")
