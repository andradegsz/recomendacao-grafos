"""Avaliação offline dos algoritmos de recomendação em grafo bipartido e tripartido.

Divide as avaliações em conjuntos de treino e teste por usuário, reconstrói
as matrizes de biadjacência apenas com as arestas de treino e compara quatro
estratégias (Popularidade, Bipartido Binário, Bipartido Ponderado e Tripartido
Ponderado) usando precision@K, recall@K e hit rate@K sobre itens relevantes
(nota >= 4.0).
"""
import numpy as np
import pandas as pd

from matriz import carregar_dados, montar_matrizes
from popularidade import recomendar_popularidade
from recomendar import recomendar_bbtb


def dividir_treino_teste(
    df: pd.DataFrame, frac_teste: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa aleatoriamente uma fração das avaliações de cada usuário para teste."""
    teste = df.groupby("userId").sample(frac=frac_teste, random_state=seed)
    treino = df.drop(teste.index)
    return treino, teste


def conjuntos_de_teste(
    teste: pd.DataFrame,
    idx_usuario: dict,
    idx_filme: dict,
    nota_minima: float = 4.0,
) -> dict[int, set[int]]:
    """Agrupa por índice de usuário os índices dos filmes relevantes no teste."""
    relevantes = teste[teste["rating"] >= nota_minima]

    u_idx = relevantes["userId"].map(idx_usuario)
    i_idx = relevantes["movieId"].map(idx_filme)

    return i_idx.groupby(u_idx).apply(set).to_dict()


def metricas(
    recomendacoes: np.ndarray, teste: dict[int, set[int]]
) -> dict[str, float]:
    """Calcula precision@K, recall@K e hit rate@K médios entre os usuários."""
    K = recomendacoes.shape[1]
    precisions, recalls, hits = [], [], []

    for u_idx, gabarito in teste.items():
        acertos = len(set(recomendacoes[u_idx]) & gabarito)
        precisions.append(acertos / K)
        recalls.append(acertos / len(gabarito))
        hits.append(1.0 if acertos > 0 else 0.0)

    return {
        f"precision@{K}": float(np.mean(precisions)),
        f"recall@{K}": float(np.mean(recalls)),
        f"hit_rate@{K}": float(np.mean(hits)),
    }


def avaliar(K: int = 10, alpha: float = 0.5) -> pd.DataFrame:
    """Executa a avaliação comparativa dos quatro modelos e retorna a tabela."""
    ratings, tags, mapeamentos = carregar_dados()
    treino, teste_df = dividir_treino_teste(ratings)

    B_treino, T = montar_matrizes(treino, tags, mapeamentos)

    # Versão não ponderada (0/1) preservando todas as arestas de treino
    B_bin = B_treino.copy()
    B_bin.data = np.ones_like(B_bin.data)

    gabarito = conjuntos_de_teste(
        teste_df,
        mapeamentos["usuario"],
        mapeamentos["filme"],
        nota_minima=4.0,
    )

    modelos = {
        "1. Popularidade": recomendar_popularidade(B_bin, K=K),
        "2. Bipartido Binário": recomendar_bbtb(B_bin, K=K),
        "3. Bipartido Ponderado": recomendar_bbtb(B_treino, K=K),
        "4. Tripartido Ponderado": recomendar_bbtb(B_treino, T=T, alpha=alpha, K=K),
    }

    resultados = {
        nome: metricas(recs, gabarito) for nome, recs in modelos.items()
    }
    return pd.DataFrame(resultados).T


if __name__ == "__main__":
    tabela = avaliar(K=10, alpha=0.5)
    print(tabela.round(4))