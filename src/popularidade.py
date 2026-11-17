import numpy as np
import pandas as pd

from matriz import B, usuarios, filmes


def recomendar_popularidade(B, K=10):
    """Top-K por popularidade: os filmes de maior grau que o usuário ainda não viu.

    Devolve uma matriz (|U|, K) com os índices das colunas recomendadas.
    """
    # Grau de cada filme = soma da coluna de B = nº de usuários que o viram
    grau_filme = np.asarray(B.sum(axis=0)).ravel()

    # Mesma pontuação para todos os usuários: cada linha é o vetor de graus
    P = np.tile(grau_filme, (B.shape[0], 1))

    # Zera os filmes já vistos (todo filme tem grau >= 1, então eles vão para o fim)
    P[B.nonzero()] = 0

    return np.argsort(-P, axis=1)[:, :K]


if __name__ == "__main__":
    grau_filme = np.asarray(B.sum(axis=0)).ravel()
    print(grau_filme.shape, grau_filme.max())   # (9724,) 329

    top_k = recomendar_popularidade(B)

    movies = pd.read_csv("data/ml-latest-small/movies.csv")
    titulo = dict(zip(movies["movieId"], movies["title"]))

    u = 0   # linha 0 = userId 1
    print(f"\nTop-10 por popularidade para o usuário {usuarios[u]}:")
    for posicao, i in enumerate(top_k[u], start=1):
        print(f"{posicao:2d}. {titulo[filmes[i]]}  (grau {grau_filme[i]})")
