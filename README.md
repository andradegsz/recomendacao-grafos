# recomendacao-grafos

Sistema de recomendação de filmes baseado em grafos bipartidos e tripartidos — trabalho de Matemática Discreta.

Usuários e filmes são vértices de um grafo bipartido; há uma aresta entre o usuário *u* e o filme *i* quando *u* avaliou *i*. Com a matriz de biadjacência **B** (|U| × |I|), a pontuação de cada par é

```
S = B · Bᵀ · B
```

que soma os passeios de comprimento 3 `u → i' → v → i` ("você viu i', o usuário v também viu i', e v viu i"). Para cada usuário, recomendamos os filmes ainda não vistos de maior pontuação.

O modelo também é estendido para um **grafo tripartido**, acrescentando tags (ou gêneros) como terceiro conjunto de vértices:

```
S = B · Bᵀ · B  +  α · B · T · Tᵀ
```

onde **T** é a matriz filme × tag e α controla o peso da parcela de conteúdo.

## Estrutura

```
recomendacao-grafos/
├── data/                  # base MovieLens (não versionada, ver abaixo)
├── relatorio/
│   └── relatorio.tex      # relatório em LaTeX
├── src/
│   ├── exemplo_pequeno.py # exemplo didático com 3 usuários e 4 filmes
│   ├── explorar.py        # estatísticas do grafo (|U|, |I|, |E|, graus, densidade)
│   ├── matriz.py          # carrega os dados e monta as matrizes B e T
│   ├── recomendar.py      # recomendador B·Bᵀ·B (bipartido) e tripartido
│   ├── popularidade.py    # baseline: filmes de maior grau
│   ├── vies.py            # tabelas de viés de popularidade do relatório (B binária)
│   ├── benchmark.py       # avaliação treino/teste comparando os modelos
│   └── topologias.py      # comparação Gêneros × Tags como terceira partição
└── requirements.txt
```

## Como rodar

### 1. Pré-requisitos

- Python 3.10 ou mais recente
- Dependências:

```bash
pip install -r requirements.txt
```

### 2. Dados

Usamos a base **MovieLens latest-small** (610 usuários, 9.724 filmes avaliados, 100.836 avaliações), do GroupLens. Ela não está no repositório.

1. Baixe o arquivo em <https://files.grouplens.org/datasets/movielens/ml-latest-small.zip> (página da base: <https://grouplens.org/datasets/movielens/latest/>).
2. Extraia dentro da pasta `data/`, de modo que fique assim:

```
data/ml-latest-small/
├── links.csv
├── movies.csv
├── ratings.csv
├── tags.csv
└── README.txt
```

### 3. Scripts

Todos os comandos devem ser executados **a partir da raiz do repositório**, pois os caminhos dos dados são relativos a ela.

| Comando | O que faz |
|---|---|
| `python src/exemplo_pequeno.py` | Mostra B, B·Bᵀ, B·Bᵀ·B e a recomendação no exemplo pequeno (não precisa da base) |
| `python src/explorar.py` | Estatísticas do grafo e verificação do lema do aperto de mãos |
| `python src/matriz.py` | Monta B e T e mostra suas dimensões |
| `python src/recomendar.py` | Top-10 tripartido para o usuário 1 |
| `python src/popularidade.py` | Top-10 por popularidade para o usuário 1 |
| `python src/vies.py` | Top-10 por B·Bᵀ·B binária para o usuário 1 e indicadores de viés de popularidade (base inteira) |
| `python src/benchmark.py` | Tabela comparativa: Popularidade, Bipartido Binário, Bipartido Ponderado e Tripartido |
| `python src/topologias.py` | Propriedades das partições de gêneros e tags, sensibilidade a α e comparação final |

O `benchmark.py` e o `topologias.py` levam alguns segundos, pois calculam matrizes densas de 610 × 9.724.

> **Windows:** se aparecer `UnicodeEncodeError` ao imprimir símbolos como `ᵀ`, rode antes `$env:PYTHONIOENCODING="utf-8"` (PowerShell) ou `set PYTHONIOENCODING=utf-8` (cmd).

## Avaliação

Para cada usuário, 20% das avaliações são separadas para teste (semente fixa 42) e as matrizes são montadas só com o restante. São considerados relevantes os filmes do teste com nota ≥ 4,0. As métricas, todas com K = 10, são:

- **precision@K**, **recall@K** e **hit rate@K**: acerto das recomendações;
- **recall de baixo grau**, **cobertura** e **grau médio**: viés de popularidade (o quanto o modelo se concentra em filmes muito vistos).

## Relatório

O relatório está em `relatorio/relatorio.pdf`, junto com o código-fonte em LaTeX (`relatorio/relatorio.tex`).

## Autores

Alice Barros Lourencini Palaoro, Felipe Augusto da Silva Cruz, Gabriel Andrade Souza e Guilherme Henrique Pereira Cunha.

## Referência da base

F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. <https://doi.org/10.1145/2827872>
