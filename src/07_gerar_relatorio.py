import os
import html
import json
import re
import sys

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_RAIZ = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PASTA_REUNIOES = os.path.join(
    PASTA_RAIZ,
    "reunioes"
)

ARQUIVO_METADADOS_PESSOAS = os.path.join(
    PASTA_RAIZ,
    "modelos",
    "metadados_pessoas.json"
)


def obter_pasta_reuniao():
    """
    Usa a reunião informada na linha de comando ou, quando
    omitida, seleciona a reunião numerada mais recente.
    """

    if len(sys.argv) > 2:
        raise SystemExit(
            "Uso: python src/07_gerar_relatorio.py "
            "[reuniao_XX]"
        )

    if len(sys.argv) == 2:

        nome_reuniao = sys.argv[1]

        if not re.fullmatch(r"reuniao_\d+", nome_reuniao):
            raise SystemExit(
                "Nome de reunião inválido. "
                "Use o formato reuniao_01."
            )

        return os.path.join(
            PASTA_REUNIOES,
            nome_reuniao
        )

    if not os.path.exists(PASTA_REUNIOES):
        raise SystemExit(
            "Nenhuma reunião foi encontrada. "
            "Execute primeiro: "
            "python src/06_analisar_reuniao.py"
        )

    reunioes = []

    for nome in os.listdir(PASTA_REUNIOES):

        correspondencia = re.fullmatch(
            r"reuniao_(\d+)",
            nome
        )

        if correspondencia:

            caminho = os.path.join(
                PASTA_REUNIOES,
                nome
            )

            if os.path.isfile(
                os.path.join(caminho, "registros.csv")
            ):
                reunioes.append(
                    (int(correspondencia.group(1)), caminho)
                )

    if not reunioes:
        raise SystemExit(
            "Nenhuma reunião com registros.csv foi encontrada."
        )

    return max(reunioes)[1]


PASTA_REUNIAO = obter_pasta_reuniao()

ARQUIVO_CSV = os.path.join(
    PASTA_REUNIAO,
    "registros.csv"
)

PASTA_GRAFICOS = os.path.join(
    PASTA_REUNIAO,
    "graficos"
)

ARQUIVO_RELATORIO_TXT = os.path.join(
    PASTA_REUNIAO,
    "relatorio.txt"
)

LIMITE_CONFIANCA_PESSOA = 0.45
LIMITE_CONFIANCA_EMOCAO = 0.45


# ============================================================
# PREPARAÇÃO
# ============================================================

if not os.path.exists(ARQUIVO_CSV):

    print("ERRO: arquivo CSV não encontrado.")
    print(f"Esperado em: {ARQUIVO_CSV}")
    print()
    print("Execute primeiro:")
    print("python src/06_analisar_reuniao.py")

    raise SystemExit(1)


print("=" * 60)
print("       GERAÇÃO DO RELATÓRIO DA REUNIÃO")
print("=" * 60)
print()


# ============================================================
# LEITURA DOS DADOS
# ============================================================

try:
    df = pd.read_csv(ARQUIVO_CSV)
except pd.errors.EmptyDataError:
    raise SystemExit("ERRO: o CSV está vazio.")

if df.empty:

    print("ERRO: o CSV está vazio.")
    print("É necessário analisar uma reunião primeiro.")

    raise SystemExit(1)


os.makedirs(PASTA_GRAFICOS, exist_ok=True)


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

participantes_identificados = sorted(
    df["pessoa"]
    .dropna()
    .unique()
)


# Não contar desconhecido como participante identificado

participantes_identificados = [
    pessoa
    for pessoa in participantes_identificados
    if pessoa != "desconhecido"
]


participantes_registrados = list(
    participantes_identificados
)


if os.path.exists(ARQUIVO_METADADOS_PESSOAS):

    try:

        with open(
            ARQUIVO_METADADOS_PESSOAS,
            "r",
            encoding="utf-8"
        ) as arquivo:

            metadados_pessoas = json.load(arquivo)

        participantes_registrados = sorted(
            {
                str(pessoa)
                for pessoa in metadados_pessoas.get(
                    "classes",
                    []
                )
            }
            | set(participantes_identificados)
        )

    except (OSError, ValueError, TypeError):
        participantes_registrados = list(
            participantes_identificados
        )


print()
print(f"Duração aproximada: {duracao_total:.1f} segundos")
print(
    f"Participantes identificados: "
    f"{len(participantes_identificados)}"
)


# ============================================================
# TEMPO DE FALA POR PESSOA
# ============================================================

tempo_por_pessoa = (
    df[df["pessoa"] != "desconhecido"]
    .groupby("pessoa")["duracao"]
    .sum()
    .reindex(
        participantes_registrados,
        fill_value=0
    )
    .sort_values(ascending=False)
)


trechos_por_pessoa = (
    df[df["pessoa"] != "desconhecido"]
    .groupby("pessoa")
    .size()
    .reindex(
        participantes_registrados,
        fill_value=0
    )
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
    .reindex(participantes_registrados)
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
        PASTA_GRAFICOS,
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
        PASTA_GRAFICOS,
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
        PASTA_GRAFICOS,
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
    PASTA_GRAFICOS,
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
    PASTA_REUNIAO,
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
{len(participantes_identificados)}
</p>

<p>
<strong>Participantes cadastrados:</strong>
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

<img src="graficos/tempo_fala.png">


<h2>4. Emoções estimadas por participante</h2>

<p>
Os percentuais abaixo representam a distribuição dos
trechos classificados pelo modelo para cada participante.
Eles não representam necessariamente o estado emocional
real da pessoa.
</p>

{tabela_emocoes_html}


<h2>5. Distribuição das emoções estimadas</h2>

<img src="graficos/emocoes_por_pessoa.png">


<h2>6. Distribuição geral das emoções</h2>

<img src="graficos/emocoes_geral.png">


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
<strong>registros.csv</strong>
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


linhas_relatorio = [
    "RELATÓRIO DA REUNIÃO",
    "=====================",
    "",
    f"Duração aproximada: {duracao_total / 60:.2f} minutos",
    f"Trechos analisados: {quantidade_trechos}",
    (
        "Participantes identificados: "
        f"{len(participantes_identificados)}"
    ),
    (
        "Participantes cadastrados: "
        f"{len(participantes_registrados)}"
    ),
    f"Emoção mais frequente: {emocao_mais_frequente}",
    (
        "Trechos nessa classificação: "
        f"{quantidade_emocao_frequente}"
    ),
    f"Trechos com baixa confiança: {len(df_baixa_confianca)}",
    "",
    "RESULTADOS POR PARTICIPANTE",
    "==========================="
]


for pessoa in participantes_registrados:

    tempo = float(
        tempo_por_pessoa.get(pessoa, 0)
    )

    quantidade = int(
        trechos_por_pessoa.get(pessoa, 0)
    )

    linhas_relatorio.extend([
        "",
        str(pessoa).upper(),
        f"Tempo estimado de fala: {tempo / 60:.2f} minutos",
        f"Trechos analisados: {quantidade}",
        "Classificações vocais:"
    ])

    for emocao in emocoes_esperadas:

        percentual = 0.0

        if pessoa in tabela_emocoes_pessoa.index:
            percentual = float(
                tabela_emocoes_pessoa.loc[pessoa, emocao]
            )

        linhas_relatorio.append(
            f"- {emocao}: {percentual:.2f}%"
        )

    percentual_irritado = 0.0

    if pessoa in tabela_emocoes_pessoa.index:
        percentual_irritado = float(
            tabela_emocoes_pessoa.loc[pessoa, "irritado"]
        )

    linhas_relatorio.append(
        "Observação: "
        f"{percentual_irritado:.2f}% dos trechos de fala foram "
        "classificados como irritado. Isso não comprova o "
        "sentimento real do participante."
    )


linhas_relatorio.extend([
    "",
    "OBSERVAÇÃO METODOLÓGICA",
    "=======================",
    (
        "As classificações representam estimativas da emoção "
        "vocal dos trechos. Elas não comprovam o sentimento real "
        "dos participantes."
    ),
    (
        "A identificação de pessoas depende da qualidade do "
        "dataset, do microfone e das condições do ambiente."
    )
])


with open(
    ARQUIVO_RELATORIO_TXT,
    "w",
    encoding="utf-8"
) as arquivo:

    arquivo.write(
        "\n".join(linhas_relatorio) + "\n"
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
    f"\n{PASTA_GRAFICOS}"
)

print()
print(
    "Para visualizar, abra o arquivo "
    "relatorio_reuniao.html no navegador."
)
