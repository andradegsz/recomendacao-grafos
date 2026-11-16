import numpy as np

usuarios = ["Ana", "Bruno", "Carla"]
filmes = ["Matrix", "Titanic", "Toy Story", "Up"]

arestas = [("Ana", "Matrix"), ("Ana", "Titanic"),
           ("Bruno", "Matrix"), ("Bruno", "Toy Story"), ("Bruno", "Up"),
           ("Carla", "Titanic"), ("Carla", "Toy Story")]

# Mapeamento nome -> índice (linha/coluna da matriz)
idx_usuario = {nome: k for k, nome in enumerate(usuarios)}
idx_filme = {nome: k for k, nome in enumerate(filmes)}

# Matriz de zeros |U| x |I|
B = np.zeros((len(usuarios), len(filmes)), dtype=int)

# Uma aresta (u, i) vira um 1 na posição [linha de u, coluna de i]
for u, i in arestas:
    B[idx_usuario[u], idx_filme[i]] = 1
print('B')
print(B)

print('B @ B.T')
print(B @ B.T)

print('B @ B.T @ B')
print(B @ B.T @ B)

S = B @ B.T @ B      # S de "score" (pontuação)
S[B == 1] = 0        # zera onde o usuário já viu o filme
print('S')
print(S)

for u, i in enumerate(np.argmax(S, axis=1)):
    print(usuarios[u], '->', filmes[i])    