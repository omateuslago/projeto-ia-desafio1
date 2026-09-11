import os
import html

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_CSV = "reunioes/resultado_reuniao.csv"

PASTA_RELATORIO = "reunioes/relatorio"

LIMITE_CONFIANCA_PESSOA = 0.45
LIMITE_CONFIANCA_EMOCAO = 0.45


# ============================================================
# PREPARAÇÃO
# ============================================================

os.makedirs(PASTA_RELATORIO, exist_ok=True)


if not os.path.exists(ARQUIVO_CSV):

    print("ERRO: arquivo CSV não encontrado.")
    print(f"Esperado em: {ARQUIVO_CSV}")
    print()
    print("Execute primeiro:")
    print("python src/06_analisar_reuniao.py")

    raise SystemExit


print("=" * 60)
print("       GERAÇÃO DO RELATÓRIO DA REUNIÃO")
print("=" * 60)
print()


# ============================================================
# LEITURA DOS DADOS
# ============================================================

df = pd.read_csv(ARQUIVO_CSV)

if df.empty:

    print("ERRO: o CSV está vazio.")
    print("É necessário analisar uma reunião primeiro.")

    raise SystemExit


print(f"Trechos encontrados: {len(df)}")


# ============================================================
# CONVERSÃO DOS DADOS
# ============================================================

df["inicio"] = pd.to_numeric(
    df["inicio"],
    errors="coerce"
)

df["fim"] = pd.to_numeric(
    df["fim"],
    errors="coerce"
)

df["conf_pessoa"] = pd.to_numeric(
    df["conf_pessoa"],
    errors="coerce"
)

df["conf_emocao"] = pd.to_numeric(
    df["conf_emocao"],
    errors="coerce"
)


df["duracao"] = (
    df["fim"] - df["inicio"]
)


df["duracao"] = df["duracao"].clip(
    lower=0
)


# ============================================================
# INFORMAÇÕES GERAIS
# ============================================================

inicio_reuniao = df["inicio"].min()

fim_reuniao = df["fim"].max()

duracao_total = fim_reuniao - inicio_reuniao

quantidade_trechos = len(df)

participantes = sorted(
    df["pessoa"]
    .dropna()
    .unique()
)


# Não contar desconhecido como participante registrado

participantes_registrados = [
    pessoa
    for pessoa in participantes
    if pessoa != "desconhecido"
]


print()
print(f"Duração aproximada: {duracao_total:.1f} segundos")
print(
    f"Participantes identificados: "
    f"{len(participantes_registrados)}"
)


# ============================================================
# TEMPO DE FALA POR PESSOA
# ============================================================

tempo_por_pessoa = (
    df[df["pessoa"] != "desconhecido"]
    .groupby("pessoa")["duracao"]
    .sum()
    .sort_values(ascending=False)
)


trechos_por_pessoa = (
    df[df["pessoa"] != "desconhecido"]
    .groupby("pessoa")
    .size()
    .sort_values(ascending=False)
)


# ============================================================
# EMOÇÕES
# ============================================================

df_validos_emocao = df[
    df["emocao"].notna()
]


contagem_emocoes = (
    df_validos_emocao["emocao"]
    .value_counts()
)


percentual_emocoes = (
    df_validos_emocao["emocao"]
    .value_counts(
        normalize=True
    ) * 100
)


# ============================================================
# EMOÇÕES POR PARTICIPANTE
# ============================================================

df_participantes = df[
    df["pessoa"] != "desconhecido"
].copy()


tabela_emocoes_pessoa = pd.crosstab(
    df_participantes["pessoa"],
    df_participantes["emocao"],
    normalize="index"
) * 100


# Garantir que as quatro emoções apareçam

emocoes_esperadas = [
    "alegre",
    "neutro",
    "triste",
    "irritado"
]


for emocao in emocoes_esperadas:

    if emocao not in tabela_emocoes_pessoa.columns:

        tabela_emocoes_pessoa[emocao] = 0


tabela_emocoes_pessoa = (
    tabela_emocoes_pessoa[
        emocoes_esperadas
    ]
    .fillna(0)
)


# ============================================================
# BAIXA CONFIANÇA
# ============================================================

baixa_confianca_pessoa = df[
    df["conf_pessoa"]
    < LIMITE_CONFIANCA_PESSOA
].copy()


baixa_confianca_emocao = df[
    df["conf_emocao"]
    < LIMITE_CONFIANCA_EMOCAO
].copy()


# ============================================================
# GRÁFICO 1 — TEMPO DE FALA
# ============================================================

if not tempo_por_pessoa.empty:

    plt.figure(figsize=(10, 6))

    tempo_minutos = (
        tempo_por_pessoa / 60
    )

    tempo_minutos.plot(
        kind="bar"
    )

    plt.title(
        "Tempo estimado de fala por participante"
    )

    plt.xlabel("Participante")

    plt.ylabel("Tempo de fala (minutos)")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    caminho_grafico = os.path.join(
        PASTA_RELATORIO,
        "tempo_fala.png"
    )

    plt.savefig(
        caminho_grafico,
        dpi=150
    )

    plt.close()


# ============================================================
# GRÁFICO 2 — EMOÇÕES POR PARTICIPANTE
# ============================================================

if not tabela_emocoes_pessoa.empty:

    plt.figure(figsize=(11, 6))

    tabela_emocoes_pessoa.plot(
        kind="bar",
        stacked=True,
        ax=plt.gca()
    )

    plt.title(
        "Distribuição das emoções estimadas por participante"
    )

    plt.xlabel("Participante")

    plt.ylabel("Percentual dos trechos (%)")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.legend(
        title="Emoção"
    )

    plt.tight_layout()

    caminho_grafico = os.path.join(
        PASTA_RELATORIO,
        "emocoes_por_pessoa.png"
    )

    plt.savefig(
        caminho_grafico,
        dpi=150
    )

    plt.close()


# ============================================================
# GRÁFICO 3 — EMOÇÕES GERAIS
# ============================================================

if not contagem_emocoes.empty:

    plt.figure(figsize=(8, 6))

    contagem_emocoes.plot(
        kind="bar"
    )

    plt.title(
        "Distribuição geral das emoções estimadas"
    )

    plt.xlabel("Emoção")

    plt.ylabel("Quantidade de trechos")

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    caminho_grafico = os.path.join(
        PASTA_RELATORIO,
        "emocoes_geral.png"
    )

    plt.savefig(
        caminho_grafico,
        dpi=150
    )

    plt.close()


# ============================================================
# TABELA DE BAIXA CONFIANÇA
# ============================================================

colunas_baixa_confianca = [
    "inicio",
    "fim",
    "pessoa",
    "conf_pessoa",
    "emocao",
    "conf_emocao",
    "arquivo"
]


df_baixa_confianca = df[
    (
        df["conf_pessoa"]
        < LIMITE_CONFIANCA_PESSOA
    )
    |
    (
        df["conf_emocao"]
        < LIMITE_CONFIANCA_EMOCAO
    )
][colunas_baixa_confianca]


# ============================================================
# EXPORTAR TABELA DE BAIXA CONFIANÇA
# ============================================================

arquivo_baixa_confianca = os.path.join(
    PASTA_RELATORIO,
    "baixa_confianca.csv"
)

df_baixa_confianca.to_csv(
    arquivo_baixa_confianca,
    index=False
)


# ============================================================
# RESUMO DE TEXTO
# ============================================================

if not contagem_emocoes.empty:

    emocao_mais_frequente = (
        contagem_emocoes.idxmax()
    )

    quantidade_emocao_frequente = (
        contagem_emocoes.max()
    )

else:

    emocao_mais_frequente = "não disponível"

    quantidade_emocao_frequente = 0


# ============================================================
# HTML
# ============================================================

def formatar_tabela(df_tabela):

    if df_tabela.empty:

        return (
            "<p>Nenhum trecho apresentou "
            "baixa confiança.</p>"
        )

    tabela = df_tabela.copy()

    tabela["conf_pessoa"] = (
        tabela["conf_pessoa"]
        .map(lambda x: f"{x:.1%}")
    )

    tabela["conf_emocao"] = (
        tabela["conf_emocao"]
        .map(lambda x: f"{x:.1%}")
    )

    tabela["inicio"] = (
        tabela["inicio"]
        .map(lambda x: f"{x:.2f}s")
    )

    tabela["fim"] = (
        tabela["fim"]
        .map(lambda x: f"{x:.2f}s")
    )

    return tabela.to_html(
        index=False,
        classes="tabela"
    )


# ============================================================
# TABELA DE PARTICIPANTES
# ============================================================

dados_participantes = []

for pessoa in participantes_registrados:

    tempo = tempo_por_pessoa.get(
        pessoa,
        0
    )

    quantidade = trechos_por_pessoa.get(
        pessoa,
        0
    )

    dados_participantes.append({
        "pessoa": pessoa,
        "tempo": f"{tempo / 60:.2f} min",
        "trechos": quantidade
    })


df_participantes_relatorio = pd.DataFrame(
    dados_participantes
)


if not df_participantes_relatorio.empty:

    tabela_participantes_html = (
        df_participantes_relatorio
        .rename(
            columns={
                "pessoa": "Participante",
                "tempo": "Tempo estimado",
                "trechos": "Quantidade de trechos"
            }
        )
        .to_html(
            index=False,
            classes="tabela"
        )
    )

else:

    tabela_participantes_html = (
        "<p>Nenhum participante registrado.</p>"
    )


# ============================================================
# TABELA DE EMOÇÕES
# ============================================================

if not tabela_emocoes_pessoa.empty:

    tabela_emocoes_html = (
        tabela_emocoes_pessoa
        .round(2)
        .to_html(
            classes="tabela"
        )
    )

else:

    tabela_emocoes_html = (
        "<p>Não existem dados suficientes.</p>"
    )


# ============================================================
# HTML FINAL
# ============================================================

caminho_html = os.path.join(
    PASTA_RELATORIO,
    "relatorio_reuniao.html"
)


html_relatorio = f"""
<!DOCTYPE html>

<html lang="pt-BR">

<head>

<meta charset="UTF-8">

<title>Relatório - Sala de Reunião Inteligente</title>

<style>

body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    line-height: 1.5;
}}

h1 {{
    margin-bottom: 5px;
}}

h2 {{
    margin-top: 35px;
}}

.resumo {{
    background: #f2f2f2;
    padding: 20px;
    border-radius: 8px;
}}

.tabela {{
    border-collapse: collapse;
    width: 100%;
    margin-top: 15px;
}}

.tabela th,
.tabela td {{
    border: 1px solid #cccccc;
    padding: 8px;
    text-align: center;
}}

.tabela th {{
    background: #eeeeee;
}}

img {{
    max-width: 900px;
    width: 100%;
    margin-top: 15px;
}}

.aviso {{
    padding: 15px;
    background: #fff3cd;
    border: 1px solid #ffe69c;
}}

</style>

</head>

<body>

<h1>Sala de Reunião Inteligente</h1>

<p>
Relatório automático da análise da reunião.
</p>


<h2>1. Resumo da reunião</h2>

<div class="resumo">

<p>
<strong>Duração aproximada:</strong>
{duracao_total / 60:.2f} minutos
</p>

<p>
<strong>Quantidade de trechos analisados:</strong>
{quantidade_trechos}
</p>

<p>
<strong>Participantes identificados:</strong>
{len(participantes_registrados)}
</p>

<p>
<strong>Emoção estimada mais frequente:</strong>
{html.escape(str(emocao_mais_frequente))}
</p>

<p>
<strong>Quantidade de trechos nessa classificação:</strong>
{quantidade_emocao_frequente}
</p>

</div>


<h2>2. Participantes e tempo estimado de fala</h2>

{tabela_participantes_html}


<h2>3. Tempo estimado de fala</h2>

<img src="tempo_fala.png">


<h2>4. Emoções estimadas por participante</h2>

<p>
Os percentuais abaixo representam a distribuição dos
trechos classificados pelo modelo para cada participante.
Eles não representam necessariamente o estado emocional
real da pessoa.
</p>

{tabela_emocoes_html}


<h2>5. Distribuição das emoções estimadas</h2>

<img src="emocoes_por_pessoa.png">


<h2>6. Distribuição geral das emoções</h2>

<img src="emocoes_geral.png">


<h2>7. Trechos com baixa confiança</h2>

<div class="aviso">

<p>
Foram considerados trechos de baixa confiança aqueles
em que a confiança do modelo de pessoa ou de emoção
ficou abaixo de 45%.
</p>

<p>
Esses trechos devem ser analisados com atenção, pois
uma baixa confiança indica maior incerteza da classificação.
</p>

</div>

{formatar_tabela(df_baixa_confianca)}


<h2>8. Observação metodológica</h2>

<p>
O sistema realiza uma classificação automática baseada
nas características acústicas dos trechos de áudio.
A classificação de emoção deve ser interpretada como
uma <strong>estimativa da emoção vocal</strong>, e não como
uma confirmação do sentimento real do participante.
</p>

<p>
Da mesma forma, a identificação de pessoas depende da
qualidade do áudio, das características capturadas durante
o treinamento e das condições de gravação.
</p>


<h2>9. Arquivo de dados</h2>

<p>
Os resultados completos podem ser consultados no arquivo:
<strong>resultado_reuniao.csv</strong>
</p>

</body>

</html>
"""


with open(
    caminho_html,
    "w",
    encoding="utf-8"
) as arquivo:

    arquivo.write(
        html_relatorio
    )


# ============================================================
# FINALIZAÇÃO
# ============================================================

print()
print("=" * 60)
print("RELATÓRIO GERADO COM SUCESSO")
print("=" * 60)

print()
print(
    f"Relatório:"
    f"\n{caminho_html}"
)

print()
print(
    f"Gráficos e tabelas:"
    f"\n{PASTA_RELATORIO}"
)

print()
print(
    "Para visualizar, abra o arquivo "
    "relatorio_reuniao.html no navegador."
)
