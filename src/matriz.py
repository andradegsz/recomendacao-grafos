import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

ratings = pd.read_csv("data/ml-latest-small/ratings.csv")

usuarios = sorted(ratings['userId'].unique())  # os userId distintos
filmes = sorted(ratings['movieId'].unique())   # os movieId distintos

idx_usuario = {uid: k for k, uid in enumerate(usuarios)} # mapeando os usuarios
idx_filme = {mid: k for k, mid in enumerate(filmes)}     # mapeando os filmes

print(len(usuarios), len(filmes))

linhas = ratings['userId'].map(idx_usuario) # Aplicando o mapeamento para as linhas da matriz
colunas = ratings['movieId'].map(idx_filme) # Aplicando o mapeamento para as colunas da matriz

print(len(linhas), len(colunas))
print(linhas.max(), colunas.max())     

valores = np.ones(len(ratings), dtype=int)

B = csr_matrix((valores, (linhas, colunas)), shape=(len(usuarios), len(filmes))) # B[u,i] = 1 se (u,i) ∈ E

print(B.shape, B.nnz)