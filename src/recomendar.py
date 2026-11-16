import numpy as np
import pandas as pd

from matriz import B, usuarios, filmes, idx_usuario, idx_filme

S = B @ B.T @ B  # B·Bᵀ·B

print(type(S))
print(S.shape)
print(S.nnz)

S = S.toarray()       # S é quase toda não nula (~99%), então densa é mais eficiente
S[B.nonzero()] = 0    # zera S[u,i] para cada aresta (u,i): filmes já vistos

print(S[0, idx_filme[1]])   # usuário 1 viu o filme 1 -> tem que dar 0

# Top-10: para cada linha, os índices das colunas em ordem decrescente de pontuação
K = 10
top_k = np.argsort(-S, axis=1)[:, :K]   # formato (|U|, K)

# Traduz índice de coluna -> movieId -> título, para mostrar o resultado
movies = pd.read_csv("data/ml-latest-small/movies.csv")
titulo = dict(zip(movies["movieId"], movies["title"]))

u = 0   # linha 0 = userId 1
print(f"\nTop-{K} para o usuário {usuarios[u]}:")
for posicao, i in enumerate(top_k[u], start=1):
    print(f"{posicao:2d}. {titulo[filmes[i]]}  (pontuação {S[u, i]})")