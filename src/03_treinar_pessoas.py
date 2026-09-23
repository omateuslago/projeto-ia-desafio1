import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from features import extrair_caracteristicas


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_RAIZ = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PASTA_DATASET = os.path.join(
    PASTA_RAIZ,
    "dataset_pessoas"
)
PASTA_MODELOS = os.path.join(
    PASTA_RAIZ,
    "modelos"
)
PASTA_RESULTADOS = os.path.join(
    PASTA_RAIZ,
    "resultados"
)

NOME_MODELO = "modelo_pessoas.pkl"

TAMANHO_MINIMO_DATASET = 20
QUANTIDADE_MINIMA_PESSOAS = 5

RANDOM_STATE = 42


# ============================================================
# FUNÇÃO PARA CRIAR PASTAS
# ============================================================

def criar_pastas():
    """
    Cria as pastas necessárias para salvar os modelos
    e os resultados do treinamento.
    """

    os.makedirs(PASTA_MODELOS, exist_ok=True)
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)


# ============================================================
# CARREGAMENTO DO DATASET
# ============================================================

def carregar_dataset():
    """
    Percorre o dataset de pessoas.

    Cada pasta representa uma pessoa:

        dataset_pessoas/
            gabriel/
            joao/
            mateus/
            caio/
            arthur/

    Retorna:

        X -> características dos áudios
        y -> nome da pessoa correspondente
    """

    X = []
    y = []

    print()
    print("=" * 70)
    print("CARREGANDO DATASET DE PESSOAS")
    print("=" * 70)

    if not os.path.exists(PASTA_DATASET):

        raise FileNotFoundError(
            f"A pasta '{PASTA_DATASET}' não foi encontrada."
        )

    pessoas = sorted(
        [
            pasta
            for pasta in os.listdir(PASTA_DATASET)
            if os.path.isdir(
                os.path.join(PASTA_DATASET, pasta)
            )
        ]
    )

    if len(pessoas) < QUANTIDADE_MINIMA_PESSOAS:

        raise ValueError(
            f"É necessário ter pelo menos "
            f"{QUANTIDADE_MINIMA_PESSOAS} pessoas "
            "no dataset para realizar o treinamento."
        )

    print()
    print(f"Pessoas encontradas: {len(pessoas)}")
    print()

    quantidade_por_pessoa = {}

    for pessoa in pessoas:

        pasta_pessoa = os.path.join(
            PASTA_DATASET,
            pessoa
        )

        arquivos = sorted(
            [
                arquivo
                for arquivo in os.listdir(pasta_pessoa)
                if arquivo.lower().endswith(".wav")
            ]
        )

        quantidade_por_pessoa[pessoa] = 0

        print(
            f"{pessoa:<20} "
            f"{len(arquivos):>3} arquivos"
        )

        if len(arquivos) < TAMANHO_MINIMO_DATASET:

            raise ValueError(
                f"A pessoa '{pessoa}' possui apenas "
                f"{len(arquivos)} arquivos. "
                f"São necessários pelo menos "
                f"{TAMANHO_MINIMO_DATASET}."
            )

        for arquivo in arquivos:

            caminho = os.path.join(
                pasta_pessoa,
                arquivo
            )

            try:

                caracteristicas = extrair_caracteristicas(
                    caminho
                )

                X.append(caracteristicas)
                y.append(pessoa)
                quantidade_por_pessoa[pessoa] += 1

            except Exception as erro:

                print()
                print(
                    f"Erro ao processar: {caminho}"
                )

                print(
                    f"Motivo: {erro}"
                )

        validos = quantidade_por_pessoa[pessoa]
        if validos < TAMANHO_MINIMO_DATASET:
            raise ValueError(
                f"A classe '{pessoa}' tem apenas {validos} áudios válidos; "
                f"são necessários {TAMANHO_MINIMO_DATASET}."
            )

    if len(X) == 0:

        raise ValueError(
            "Nenhum áudio válido foi encontrado no dataset."
        )

    print()
    print("-" * 70)
    print(f"Total de áudios processados: {len(X)}")
    print(f"Total de características: {len(X[0])}")
    print("-" * 70)

    return (
        np.array(X),
        np.array(y),
        quantidade_por_pessoa
    )


# ============================================================
# VERIFICAÇÃO DO DATASET
# ============================================================

def verificar_dataset(y):
    """
    Verifica se cada classe possui quantidade suficiente
    de exemplos para permitir uma divisão estratificada.
    """

    classes, quantidades = np.unique(
        y,
        return_counts=True
    )

    print()
    print("=" * 70)
    print("VERIFICAÇÃO DAS CLASSES")
    print("=" * 70)

    for classe, quantidade in zip(
        classes,
        quantidades
    ):

        print(
            f"{classe:<20} "
            f"{quantidade:>3} amostras"
        )

    print()

    if len(classes) < QUANTIDADE_MINIMA_PESSOAS:
        raise ValueError(f"São necessárias pelo menos {QUANTIDADE_MINIMA_PESSOAS} classes válidas.")
    if np.min(quantidades) < TAMANHO_MINIMO_DATASET:
        raise ValueError(f"Cada classe precisa de {TAMANHO_MINIMO_DATASET} áudios válidos.")


# ============================================================
# TREINAMENTO
# ============================================================

def treinar_modelo(X_treino, y_treino):
    """
    Cria e treina o classificador Random Forest.
    """

    print()
    print("=" * 70)
    print("TREINANDO RANDOM FOREST")
    print("=" * 70)

    modelo = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    modelo.fit(
        X_treino,
        y_treino
    )

    print()
    print("✓ Treinamento concluído.")

    print()
    print("Configurações do modelo:")
    print(f"- Algoritmo: Random Forest")
    print(f"- Árvores: {modelo.n_estimators}")
    print(f"- Random State: {RANDOM_STATE}")
    print(f"- Classes: {list(modelo.classes_)}")

    return modelo


# ============================================================
# AVALIAÇÃO
# ============================================================

def avaliar_modelo(modelo, X_teste, y_teste):
    """
    Avalia o modelo utilizando o conjunto de teste.
    """

    print()
    print("=" * 70)
    print("AVALIAÇÃO DO MODELO")
    print("=" * 70)

    y_pred = modelo.predict(X_teste)

    acuracia = accuracy_score(
        y_teste,
        y_pred
    )

    print()
    print(
        f"Acurácia: {acuracia * 100:.2f}%"
    )

    print()
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print("-" * 70)

    relatorio = classification_report(
        y_teste,
        y_pred,
        zero_division=0
    )

    print(relatorio)

    return y_pred, acuracia, relatorio


# ============================================================
# MATRIZ DE CONFUSÃO
# ============================================================

def gerar_matriz_confusao(
    y_teste,
    y_pred,
    classes
):
    """
    Gera e salva a matriz de confusão.
    """

    matriz = confusion_matrix(
        y_teste,
        y_pred,
        labels=classes
    )

    exibicao = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=classes
    )

    figura, eixo = plt.subplots(
        figsize=(9, 7)
    )

    exibicao.plot(
        ax=eixo,
        cmap="Blues",
        values_format="d"
    )

    eixo.set_title(
        "Matriz de Confusão - Reconhecimento de Pessoas"
    )

    eixo.set_xlabel(
        "Pessoa prevista"
    )

    eixo.set_ylabel(
        "Pessoa real"
    )

    plt.tight_layout()

    caminho = os.path.join(
        PASTA_RESULTADOS,
        "matriz_confusao_pessoas.png"
    )

    plt.savefig(
        caminho,
        dpi=300
    )

    plt.close()

    print()
    print(
        f"✓ Matriz de confusão salva em:"
    )
    print(caminho)

    return matriz


# ============================================================
# SALVAR MODELO
# ============================================================

def salvar_modelo(modelo):
    """
    Salva o Random Forest em formato PKL.
    """

    caminho = os.path.join(
        PASTA_MODELOS,
        NOME_MODELO
    )

    joblib.dump(
        modelo,
        caminho
    )

    print()
    print("✓ Modelo salvo com sucesso:")
    print(caminho)

    return caminho


# ============================================================
# SALVAR METADADOS
# ============================================================

def salvar_metadados(
    modelo,
    quantidade_por_pessoa,
    acuracia,
    quantidade_treino,
    quantidade_teste
):
    """
    Salva informações importantes do treinamento
    em um arquivo JSON.

    Isso será útil posteriormente para o relatório.
    """

    metadados = {

        "modelo": "Random Forest",

        "objetivo": (
            "Identificação da pessoa pela voz"
        ),

        "taxa_amostragem": 16000,

        "caracteristicas": [
            "MFCC",
            "Delta MFCC",
            "RMS",
            "Zero Crossing Rate",
            "Spectral Centroid",
            "Spectral Bandwidth",
            "Spectral Rolloff",
            "Pitch"
        ],

        "quantidade_arvores": modelo.n_estimators,

        "random_state": RANDOM_STATE,

        "classes": [
            str(classe)
            for classe in modelo.classes_
        ],

        "quantidade_por_pessoa": {
            str(pessoa): int(quantidade)
            for pessoa, quantidade
            in quantidade_por_pessoa.items()
        },

        "amostras_treinamento": int(
            quantidade_treino
        ),

        "amostras_teste": int(
            quantidade_teste
        ),

        "acuracia": float(
            acuracia
        )
    }

    caminho = os.path.join(
        PASTA_MODELOS,
        "metadados_pessoas.json"
    )

    with open(
        caminho,
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            metadados,
            arquivo,
            indent=4,
            ensure_ascii=False
        )

    print()
    print("✓ Metadados salvos:")
    print(caminho)

    caminho_geral = os.path.join(
        PASTA_MODELOS,
        "metadados.json"
    )

    metadados_gerais = {}

    if os.path.exists(caminho_geral):

        try:

            with open(
                caminho_geral,
                "r",
                encoding="utf-8"
            ) as arquivo:

                metadados_gerais = json.load(arquivo)

        except (OSError, ValueError, TypeError):
            metadados_gerais = {}

    metadados_gerais["pessoas"] = metadados

    with open(
        caminho_geral,
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            metadados_gerais,
            arquivo,
            indent=4,
            ensure_ascii=False
        )

    print(caminho_geral)


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def main():

    print()
    print("=" * 70)
    print("       TREINAMENTO DO MODELO DE RECONHECIMENTO")
    print("                    DE PESSOAS")
    print("=" * 70)

    criar_pastas()

    # --------------------------------------------------------
    # Carregar dataset
    # --------------------------------------------------------

    X, y, quantidade_por_pessoa = carregar_dataset()

    # --------------------------------------------------------
    # Verificar dataset
    # --------------------------------------------------------

    verificar_dataset(y)

    # --------------------------------------------------------
    # Divisão treino/teste
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DIVIDINDO DATASET")
    print("=" * 70)

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print()
    print(
        f"Dados para treinamento: {len(X_treino)}"
    )

    print(
        f"Dados para teste:        {len(X_teste)}"
    )

    # --------------------------------------------------------
    # Treinar
    # --------------------------------------------------------

    modelo = treinar_modelo(
        X_treino,
        y_treino
    )

    # --------------------------------------------------------
    # Avaliar
    # --------------------------------------------------------

    y_pred, acuracia, relatorio = avaliar_modelo(
        modelo,
        X_teste,
        y_teste
    )

    # --------------------------------------------------------
    # Matriz de confusão
    # --------------------------------------------------------

    matriz = gerar_matriz_confusao(
        y_teste,
        y_pred,
        modelo.classes_
    )

    # --------------------------------------------------------
    # Salvar modelo
    # --------------------------------------------------------

    salvar_modelo(modelo)

    # --------------------------------------------------------
    # Salvar metadados
    # --------------------------------------------------------

    salvar_metadados(
        modelo,
        quantidade_por_pessoa,
        acuracia,
        len(X_treino),
        len(X_teste)
    )

    # --------------------------------------------------------
    # Resultado final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TREINAMENTO FINALIZADO")
    print("=" * 70)

    print()
    print(
        f"Acurácia final: {acuracia * 100:.2f}%"
    )

    print()
    print("Arquivos gerados:")

    print(
        f"- {os.path.join(PASTA_MODELOS, NOME_MODELO)}"
    )

    print(
        f"- {os.path.join(PASTA_MODELOS, 'metadados_pessoas.json')}"
    )

    print(
        f"- {os.path.join(PASTA_MODELOS, 'metadados.json')}"
    )

    print(
        f"- {os.path.join(PASTA_RESULTADOS, 'matriz_confusao_pessoas.png')}"
    )

    print()
    print("✓ Processo concluído.")


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()
