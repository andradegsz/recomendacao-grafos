"""Avaliação dos algoritmos de recomendação em grafo bipartido e tripartido.

Divide as avaliações em conjuntos de treino e teste por usuário, reconstrói
as matrizes de biadjacência apenas com as arestas de treino e compara quatro
estratégias (Popularidade, Bipartido Binário, Bipartido Ponderado e Tripartido
Ponderado) usando precision@K, recall@K, hit_rate@K, recall_baixo_grau@K,
cobertura@K (em porcentagem) e grau_medio sobre itens relevantes (nota >= 4.0).
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
    recomendacoes: np.ndarray,
    teste: dict[int, set[int]],
    graus_treino: np.ndarray,
    limiar_baixo_grau: int = 10,
) -> dict[str, float]:
    """Calcula métricas de predição de links (%) e propriedades topológicas do grafo."""
    K = recomendacoes.shape[1]
    precisions, recalls, hits, recalls_bg = [], [], [], []

    filmes_baixo_grau = set(np.where(graus_treino <= limiar_baixo_grau)[0])

    for u_idx, gabarito in teste.items():
        recs_u = set(recomendacoes[u_idx])
        acertos = len(recs_u & gabarito)

        precisions.append(acertos / K)
        recalls.append(acertos / len(gabarito))
        hits.append(1.0 if acertos > 0 else 0.0)

        gabarito_bg = gabarito & filmes_baixo_grau
        if gabarito_bg:
            acertos_bg = len(recs_u & gabarito_bg)
            recalls_bg.append(acertos_bg / len(gabarito_bg))

    cobertura = np.unique(recomendacoes).size / len(graus_treino)
    grau_medio = float(np.mean(graus_treino[recomendacoes]))

    return {
        f"precision@{K} (%)": float(np.mean(precisions)) * 100.0,
        f"recall@{K} (%)": float(np.mean(recalls)) * 100.0,
        f"hit_rate@{K} (%)": float(np.mean(hits)) * 100.0,
        f"recall_baixo_grau@{K} (%)": (
            float(np.mean(recalls_bg)) * 100.0 if recalls_bg else 0.0
        ),
        f"cobertura@{K} (%)": float(cobertura) * 100.0,
        "grau_medio": grau_medio,
    }


def buscar_melhor_alpha(
    B_treino,
    T,
    gabarito: dict[int, set[int]],
    graus_treino: np.ndarray,
    alphas: list[float] | np.ndarray = np.arange(0.0, 0.55, 0.05),
    K: int = 10,
    criterio: str | None = None,
) -> tuple[float, pd.DataFrame]:
    """Testa uma grade de valores de alpha e retorna o melhor segundo o critério.

    Pré-computa as matrizes colaborativa e de tags uma única vez para acelerar
    a busca em grade.
    """
    if criterio is None:
        criterio = f"precision@{K} (%)"

    S_colab = (B_treino @ B_treino.T @ B_treino).toarray()
    S_tags = (B_treino @ T @ T.T).toarray()
    vistos = B_treino.nonzero()

    resultados = {}
    for alpha in alphas:
        S = S_colab + alpha * S_tags
        S[vistos] = -np.inf
        recs = np.argsort(-S, axis=1)[:, :K]
        resultados[round(float(alpha), 4)] = metricas(recs, gabarito, graus_treino)

    tabela_alphas = pd.DataFrame(resultados).T
    tabela_alphas.index.name = "alpha"
    melhor_alpha = float(tabela_alphas[criterio].idxmax())

    return melhor_alpha, tabela_alphas


def avaliar(K: int = 10, alpha: float | None = None) -> pd.DataFrame:
    """Executa a avaliação comparativa dos quatro modelos e retorna a tabela."""
    ratings, tags, mapeamentos = carregar_dados()
    treino, teste_df = dividir_treino_teste(ratings)

    B_treino, T = montar_matrizes(treino, tags, mapeamentos)

    B_bin = B_treino.copy()
    B_bin.data = np.ones_like(B_bin.data)
    graus_treino = B_bin.getnnz(axis=0)

    gabarito = conjuntos_de_teste(
        teste_df,
        mapeamentos["usuario"],
        mapeamentos["filme"],
        nota_minima=4.0,
    )

    if alpha is None:
        alpha, _ = buscar_melhor_alpha(
            B_treino, T, gabarito, graus_treino, K=K, criterio=f"precision@{K} (%)"
        )

    modelos = {
        "1. Popularidade": recomendar_popularidade(B_bin, K=K),
        "2. Bipartido Binário": recomendar_bbtb(B_bin, K=K),
        "3. Bipartido Ponderado": recomendar_bbtb(B_treino, K=K),
        f"4. Tripartido Ponderado (alpha={alpha})": recomendar_bbtb(
            B_treino, T=T, alpha=alpha, K=K
        ),
    }

    resultados = {
        nome: metricas(recs, gabarito, graus_treino) for nome, recs in modelos.items()
    }
    return pd.DataFrame(resultados).T


if __name__ == "__main__":
    tabela = avaliar()
    print(tabela.round(2).to_string())