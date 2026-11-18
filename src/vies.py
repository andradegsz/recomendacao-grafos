"""Reproduz as tabelas de viés de popularidade do relatório ("Resultados preliminares").

Usa o modelo bipartido binário da teoria (B[u, i] = 1 se u avaliou i) sobre a
base inteira, sem divisão treino/teste, e imprime:

1. o Top-10 por S = B·Bᵀ·B para o usuário 1, com as pontuações;
2. os indicadores de viés: filmes distintos recomendados, sobreposição com a
   popularidade, grau mediano dos recomendados, correlação entre S e o grau
   e quantos usuários recebem Star Wars: Episode IV;
3. a fração de entradas não nulas de S.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from matriz import carregar_dados
from popularidade import recomendar_popularidade
from recomendar import recomendar_bbtb

STAR_WARS_IV = 260  # movieId de "Star Wars: Episode IV - A New Hope (1977)"


def montar_B_binaria(ratings: pd.DataFrame, idx_usuario: dict, idx_filme: dict) -> csr_matrix:
    """Matriz de biadjacência binária: B[u, i] = 1 se (u, i) é uma aresta."""
    linhas = ratings["userId"].map(idx_usuario)
    colunas = ratings["movieId"].map(idx_filme)
    valores = np.ones(len(ratings), dtype=np.float64)  # float: o bloqueio usa -inf
    return csr_matrix((valores, (linhas, colunas)), shape=(len(idx_usuario), len(idx_filme)))


def indicadores_de_vies(
    B: csr_matrix, S: np.ndarray, recs: np.ndarray, recs_pop: np.ndarray, u: int, i_star_wars: int
) -> dict[str, float]:
    """Calcula os indicadores da tabela de viés de popularidade."""
    graus = B.getnnz(axis=0)
    nao_vistos = B[u].toarray().ravel() == 0

    sobreposicao = np.mean([len(set(a) & set(b)) for a, b in zip(recs, recs_pop)])

    return {
        "Filmes distintos somando os top-10": int(np.unique(recs).size),
        "Sobreposição média com o top-10 por popularidade": round(float(sobreposicao), 1),
        "Grau mediano dos filmes recomendados": float(np.median(graus[recs])),
        "Correlação entre S_ui e deg(i) (usuário 1, não vistos)": round(
            float(np.corrcoef(S[u, nao_vistos], graus[nao_vistos])[0, 1]), 2
        ),
        "Usuários que recebem Star Wars IV": int((recs == i_star_wars).any(axis=1).sum()),
    }


if __name__ == "__main__":
    ratings, _, mapeamentos = carregar_dados()
    movies = pd.read_csv("data/ml-latest-small/movies.csv").set_index("movieId")
    filme_por_idx = {idx: mid for mid, idx in mapeamentos["filme"].items()}

    B = montar_B_binaria(ratings, mapeamentos["usuario"], mapeamentos["filme"])
    S = (B @ B.T @ B).toarray()  # pontuação sem bloqueio, para mostrar os valores de S

    recs = recomendar_bbtb(B, K=10)
    recs_pop = recomendar_popularidade(B, K=10)

    # 1. Top-10 do usuário 1
    u = mapeamentos["usuario"][1]
    top10 = pd.DataFrame({
        "Filme": [movies.loc[filme_por_idx[i], "title"] for i in recs[u]],
        "S_ui": [int(S[u, i]) for i in recs[u]],
    }, index=range(1, 11))
    print("1. TOP-10 POR B·Bᵀ·B PARA O USUÁRIO 1\n", top10.to_string())

    # 2. Indicadores de viés
    tabela = indicadores_de_vies(
        B, S, recs, recs_pop, u, mapeamentos["filme"][STAR_WARS_IV]
    )
    print("\n2. INDICADORES DE VIÉS DE POPULARIDADE")
    for medida, valor in tabela.items():
        print(f"   {medida}: {valor}")

    # 3. Esparsidade de S
    print(f"\n3. Entradas não nulas de S: {100 * np.count_nonzero(S) / S.size:.1f}%")
