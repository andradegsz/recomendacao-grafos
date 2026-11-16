import pandas as pd

ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
print(ratings.head(10))

# |U| , |I|, |E|
n_usuarios = ratings["userId"].nunique()   # nunique = nº de valores distintos
n_itens = ratings["movieId"].nunique()
n_arestas = len(ratings)                   # cada linha é uma aresta
print(f"|U| = {n_usuarios}, |I| = {n_itens}, |E| = {n_arestas}")

# Pares (usuário, filme) repetidos? Se houver, |E| != nº de linhas
duplicados = ratings.duplicated(subset=["userId", "movieId"]).sum()
print(f"Pares repetidos: {duplicados}")

# Graus: value_counts conta quantas vezes cada id aparece = nº de arestas do vértice
grau_usuario = ratings["userId"].value_counts()
grau_item = ratings["movieId"].value_counts()
print("Grau dos usuários:\n", grau_usuario.describe())
print("Grau dos filmes:\n", grau_item.describe())

# Densidade: fração das entradas de B que valem 1
densidade = n_arestas / (n_usuarios * n_itens)
print(f"Densidade = {densidade:.4f}")

# Lema do aperto de mãos (versão bipartida)
assert grau_usuario.sum() == grau_item.sum() == n_arestas
print("Aperto de mãos: OK")