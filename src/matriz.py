import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def montar_B(arestas, idx_usuario, idx_filme):
    """Monta a matriz de biadjacência B (|U| x |I|) a partir de uma tabela de arestas.

    arestas: DataFrame com as colunas 'userId' e 'movieId' (pode ser o ratings
             inteiro ou só o treino).
    idx_usuario, idx_filme: mapeamentos id -> índice, calculados com a base
             inteira, para que a linha/coluna k seja sempre o mesmo vértice.
    """
    linhas = arestas['userId'].map(idx_usuario)   # Aplicando o mapeamento para as linhas da matriz
    colunas = arestas['movieId'].map(idx_filme)   # Aplicando o mapeamento para as colunas da matriz
    valores = np.ones(len(arestas), dtype=int)

    # B[u,i] = 1 se (u,i) ∈ E
    return csr_matrix((valores, (linhas, colunas)), shape=(len(idx_usuario), len(idx_filme)))


ratings = pd.read_csv("data/ml-latest-small/ratings.csv")

usuarios = sorted(ratings['userId'].unique())  # os userId distintos
filmes = sorted(ratings['movieId'].unique())   # os movieId distintos

idx_usuario = {uid: k for k, uid in enumerate(usuarios)}  # mapeando os usuarios
idx_filme = {mid: k for k, mid in enumerate(filmes)}      # mapeando os filmes

B = montar_B(ratings, idx_usuario, idx_filme)


if __name__ == "__main__":
    print(len(usuarios), len(filmes))   # 610 9724
    print(B.shape, B.nnz)               # (610, 9724) 100836
