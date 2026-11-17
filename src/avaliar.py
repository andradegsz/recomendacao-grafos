import numpy as np

from matriz import montar_B, ratings, idx_usuario, idx_filme
from recomendar import recomendar_bbtb
from popularidade import recomendar_popularidade


def dividir_treino_teste(ratings, frac_teste=0.2, semente=42):
    """Esconde, para cada usuário, uma fração aleatória das suas avaliações.

    Devolve (treino, teste), dois DataFrames que juntos formam o ratings original.
    A semente fixa garante que o sorteio seja sempre o mesmo (reprodutibilidade).
    """
    # groupby('userId') separa as arestas de cada usuário; .sample sorteia 20% de cada grupo
    teste = ratings.groupby('userId').sample(frac=frac_teste, random_state=semente)
    # O treino é tudo que não foi sorteado para o teste
    treino = ratings.drop(teste.index)
    return treino, teste


def conjuntos_de_teste(teste, idx_usuario, idx_filme):
    """Para cada linha u de B, o conjunto de colunas (filmes) escondidas no teste.

    Exemplo: {0: {12, 507, 3301}, 1: {...}, ...}
    """
    linhas = teste['userId'].map(idx_usuario)
    colunas = teste['movieId'].map(idx_filme)
    escondidos = {}
    for u, i in zip(linhas, colunas):
        escondidos.setdefault(u, set()).add(i)   # cria o conjunto na 1ª vez e adiciona i
    return escondidos


def metricas(top_k, escondidos):
    """precision@K, recall@K e hit rate@K, como médias sobre os usuários.

    top_k: matriz (|U|, K) com as colunas recomendadas para cada usuário.
    escondidos: dicionário {u: conjunto de colunas do teste}.
    """
    K = top_k.shape[1]
    precisions, recalls, hits = [], [], []
    for u, teste_u in escondidos.items():
        acertos = len(set(top_k[u]) & teste_u)   # & = interseção de conjuntos
        precisions.append(acertos / K)
        recalls.append(acertos / len(teste_u))
        hits.append(1 if acertos > 0 else 0)
    return {
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "hit_rate": np.mean(hits),
    }


def avaliar(metodos, K=10, frac_teste=0.2, semente=42):
    """Divide a base, monta B só com o treino e mede cada método no teste."""
    treino, teste = dividir_treino_teste(ratings, frac_teste, semente)
    B_treino = montar_B(treino, idx_usuario, idx_filme)   # mesmo mapeamento da base inteira
    escondidos = conjuntos_de_teste(teste, idx_usuario, idx_filme)

    resultados = {}
    for nome, recomendar in metodos.items():
        top_k = recomendar(B_treino, K)
        resultados[nome] = metricas(top_k, escondidos)
    return resultados


if __name__ == "__main__":
    metodos = {
        "Popularidade": recomendar_popularidade,
        "B·B^T·B": recomendar_bbtb,   # ^T em vez de ᵀ: o console do Windows não imprime ᵀ
    }

    treino, teste = dividir_treino_teste(ratings)
    print(f"Treino: {len(treino)} arestas | Teste: {len(teste)} arestas")

    resultados = avaliar(metodos)

    print(f"\n{'Método':<14}{'precision@10':>14}{'recall@10':>12}{'hit rate@10':>14}")
    for nome, m in resultados.items():
        print(f"{nome:<14}{m['precision']:>14.4f}{m['recall']:>12.4f}{m['hit_rate']:>14.4f}")
