"""Experimento comparativo entre partições tripartidas: Gêneros (U-I-G) vs. Tags (U-I-T)."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from benchmark import conjuntos_de_teste, dividir_treino_teste, metricas
from matriz import carregar_dados, montar_matrizes
from recomendar import recomendar_bbtb


def _montar_G(movies: pd.DataFrame, idx_filme: dict) -> csr_matrix:
    """Constrói a matriz esparsa Filmes x Gêneros normalizada por norma L2."""
    generos_validos = sorted(
        g
        for g in movies["genres"].dropna().str.split("|").explode().unique()
        if g and g != "(no genres listed)"
    )
    idx_genero = {g: k for k, g in enumerate(generos_validos)}

    explodido = movies[["movieId", "genres"]].assign(
        genero=movies["genres"].str.split("|")
    ).explode("genero")

    pares = (
        pd.DataFrame({
            "i_idx": explodido["movieId"].map(idx_filme),
            "g_idx": explodido["genero"].map(idx_genero),
        })
        .dropna()
        .drop_duplicates()
    )

    qtd_por_filme = pares.groupby("i_idx")["g_idx"].transform("count")
    valores = 1.0 / np.sqrt(qtd_por_filme)

    return csr_matrix(
        (valores, (pares["i_idx"].astype(int), pares["g_idx"].astype(int))),
        shape=(len(idx_filme), len(idx_genero)),
    )


def propriedades_particao(X: csr_matrix, nome: str) -> dict[str, float | int]:
    """Calcula métricas estruturais da partição X e da projeção induzida X·Xᵀ."""
    n_filmes, n_atr = X.shape
    graus_atr = X.getnnz(axis=0)
    filmes_conectados = int((X.getnnz(axis=1) > 0).sum())

    projecao = X @ X.T
    pares_conectados = projecao.nnz - filmes_conectados
    densidade_proj = pares_conectados / (n_filmes * (n_filmes - 1))

    return {
        "Partição": nome,
        "Vértices (|X|)": n_atr,
        "Arestas": X.nnz,
        "Cobertura em I (%)": round(100 * filmes_conectados / n_filmes, 2),
        "Grau Médio d(x)": round(float(graus_atr.mean()), 2),
        "Grau Máximo d_max(x)": int(graus_atr.max()),
        "Densidade Projeção X·Xᵀ (%)": round(100 * densidade_proj, 2),
    }


def executar_experimento(pasta_dados: str = "data/ml-latest-small", K: int = 10) -> None:
    ratings, tags, mapeamentos = carregar_dados(pasta_dados)
    movies = pd.read_csv(f"{pasta_dados}/movies.csv")

    treino, teste_df = dividir_treino_teste(ratings)
    B_treino, T = montar_matrizes(treino, tags, mapeamentos)
    G = _montar_G(movies, mapeamentos["filme"])

    B_bin = B_treino.copy()
    B_bin.data = np.ones_like(B_bin.data)
    graus_treino = B_bin.getnnz(axis=0)
    gabarito = conjuntos_de_teste(
        teste_df, mapeamentos["usuario"], mapeamentos["filme"], nota_minima=4.0
    )

    # 1. Propriedades topológicas
    tabela_estrutura = pd.DataFrame([
        propriedades_particao(G, "Gêneros (U-I-G)"),
        propriedades_particao(T, "Tags (U-I-T)"),
    ]).set_index("Partição")
    print("1. PROPRIEDADES TOPOLÓGICAS DA TERCEIRA PARTIÇÃO\n", tabela_estrutura.to_string())

    # 2. Sensibilidade a alpha
    S_colab = (B_treino @ B_treino.T @ B_treino).toarray()
    S_gen = (B_treino @ G @ G.T).toarray()
    S_tag = (B_treino @ T @ T.T).toarray()
    vistos = B_treino.nonzero()

    linhas_alpha = []
    for a in np.round(np.arange(0.0, 0.45, 0.05), 2):
        Sg = S_colab + a * S_gen
        Sg[vistos] = -np.inf
        mg = metricas(np.argsort(-Sg, axis=1)[:, :K], gabarito, graus_treino)

        St = S_colab + a * S_tag
        St[vistos] = -np.inf
        mt = metricas(np.argsort(-St, axis=1)[:, :K], gabarito, graus_treino)

        linhas_alpha.append({
            "alpha": a,
            "Gêneros_prec@10": mg[f"precision@{K}"],
            "Tags_prec@10": mt[f"precision@{K}"],
            "Gêneros_hit@10": mg[f"hit_rate@{K}"],
            "Tags_hit@10": mt[f"hit_rate@{K}"],
        })
    print("\n2. SENSIBILIDADE A ALPHA\n", pd.DataFrame(linhas_alpha).set_index("alpha").round(4).to_string())

    # 3. Impacto final
    modelos = {
        "1. Bipartido Ponderado (alpha=0.00)": recomendar_bbtb(B_treino, K=K),
        "2. Tripartido Gêneros (alpha=0.05)": recomendar_bbtb(B_treino, T=G, alpha=0.05, K=K),
        "3. Tripartido Gêneros (alpha=0.35)": recomendar_bbtb(B_treino, T=G, alpha=0.35, K=K),
        "4. Tripartido Tags    (alpha=0.35)": recomendar_bbtb(B_treino, T=T, alpha=0.35, K=K),
    }
    tabela_final = pd.DataFrame(
        {nome: metricas(recs, gabarito, graus_treino) for nome, recs in modelos.items()}
    ).T
    print("\n3. IMPACTO FINAL NAS MÉTRICAS\n", tabela_final.round(4).to_string())


if __name__ == "__main__":
    executar_experimento()