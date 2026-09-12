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
    "dataset_emocoes"
)
PASTA_MODELOS = os.path.join(
    PASTA_RAIZ,
    "modelos"
)
PASTA_RESULTADOS = os.path.join(
    PASTA_RAIZ,
    "resultados"
)

NOME_MODELO = "modelo_emocoes.pkl"

TAMANHO_MINIMO_DATASET = 25
QUANTIDADE_MINIMA_EMOCOES = 4

RANDOM_STATE = 42


# ============================================================
# CRIAÇÃO DAS PASTAS
# ============================================================

def criar_pastas():
    """
    Cria as pastas necessárias para armazenar
    o modelo e os resultados.
    """

    os.makedirs(PASTA_MODELOS, exist_ok=True)
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)


# ============================================================
# CARREGAMENTO DO DATASET
# ============================================================

def carregar_dataset():
    """
    Carrega os áudios organizados por emoção.

    Estrutura esperada:

        dataset_emocoes/
            alegre/
            neutro/
            triste/
            irritado/

    Cada pasta representa uma classe.
    """

    X = []
    y = []

    print()
    print("=" * 70)
    print("CARREGANDO DATASET DE EMOÇÕES")
    print("=" * 70)

    if not os.path.exists(PASTA_DATASET):

        raise FileNotFoundError(
            f"A pasta '{PASTA_DATASET}' não foi encontrada."
        )

    emocoes = sorted(
        [
            pasta
            for pasta in os.listdir(PASTA_DATASET)
            if os.path.isdir(
                os.path.join(PASTA_DATASET, pasta)
            )
        ]
    )

    if len(emocoes) < QUANTIDADE_MINIMA_EMOCOES:

        raise ValueError(
            f"É necessário possuir pelo menos "
            f"{QUANTIDADE_MINIMA_EMOCOES} emoções "
            "para realizar o treinamento."
        )

    print()
    print(f"Emoções encontradas: {len(emocoes)}")
    print()

    quantidade_por_emocao = {}

    for emocao in emocoes:

        pasta_emocao = os.path.join(
            PASTA_DATASET,
            emocao
        )

        arquivos = sorted(
            [
                arquivo
                for arquivo in os.listdir(pasta_emocao)
                if arquivo.lower().endswith(".wav")
            ]
        )

        quantidade_por_emocao[emocao] = len(arquivos)

        print(
            f"{emocao:<20} "
            f"{len(arquivos):>3} arquivos"
        )

        if len(arquivos) < TAMANHO_MINIMO_DATASET:

            raise ValueError(
                f"A emoção '{emocao}' possui apenas "
                f"{len(arquivos)} arquivos. "
                f"São necessários pelo menos "
                f"{TAMANHO_MINIMO_DATASET}."
            )

        for arquivo in arquivos:

            caminho = os.path.join(
                pasta_emocao,
                arquivo
            )

            try:

                caracteristicas = extrair_caracteristicas(
                    caminho
                )

                X.append(caracteristicas)
                y.append(emocao)

            except Exception as erro:

                print()
                print(
                    f"Erro ao processar: {caminho}"
                )

                print(
                    f"Motivo: {erro}"
                )

    if len(X) == 0:

        raise ValueError(
            "Nenhum áudio válido foi encontrado."
        )

    print()
    print("-" * 70)
    print(f"Total de áudios processados: {len(X)}")
    print(f"Total de características: {len(X[0])}")
    print("-" * 70)

    return (
        np.array(X),
        np.array(y),
        quantidade_por_emocao
    )


# ============================================================
# VERIFICAÇÃO DO DATASET
# ============================================================

def verificar_dataset(y):
    """
    Verifica a quantidade de exemplos por classe.
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

    if len(classes) < 2:

        raise ValueError(
            "O dataset precisa possuir pelo menos "
            "duas emoções diferentes."
        )

    if np.min(quantidades) < 2:

        raise ValueError(
            "Cada emoção precisa possuir pelo menos "
            "2 áudios para realizar a divisão entre "
            "treino e teste."
        )


# ============================================================
# TREINAMENTO
# ============================================================

def treinar_modelo(X_treino, y_treino):
    """
    Treina o Random Forest para classificação das emoções.
    """

    print()
    print("=" * 70)
    print("TREINANDO RANDOM FOREST - EMOÇÕES")
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
    print("Configurações:")
    print("- Algoritmo: Random Forest")
    print(f"- Árvores: {modelo.n_estimators}")
    print(f"- Random State: {RANDOM_STATE}")
    print(f"- Classes: {list(modelo.classes_)}")

    return modelo


# ============================================================
# AVALIAÇÃO
# ============================================================

def avaliar_modelo(modelo, X_teste, y_teste):
    """
    Calcula a acurácia e o relatório de classificação.
    """

    print()
    print("=" * 70)
    print("AVALIAÇÃO DO MODELO DE EMOÇÕES")
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
    Gera e salva a matriz de confusão das emoções.
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
        "Matriz de Confusão - Reconhecimento de Emoções"
    )

    eixo.set_xlabel(
        "Emoção prevista"
    )

    eixo.set_ylabel(
        "Emoção real"
    )

    plt.tight_layout()

    caminho = os.path.join(
        PASTA_RESULTADOS,
        "matriz_confusao_emocoes.png"
    )

    plt.savefig(
        caminho,
        dpi=300
    )

    plt.close()

    print()
    print("✓ Matriz de confusão salva em:")
    print(caminho)

    return matriz


# ============================================================
# SALVAR MODELO
# ============================================================

def salvar_modelo(modelo):
    """
    Salva o modelo treinado.
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
    print("✓ Modelo salvo:")
    print(caminho)

    return caminho


# ============================================================
# SALVAR METADADOS
# ============================================================

def salvar_metadados(
    modelo,
    quantidade_por_emocao,
    acuracia,
    quantidade_treino,
    quantidade_teste
):
    """
    Salva informações do treinamento em JSON.
    """

    metadados = {

        "modelo": "Random Forest",

        "objetivo": (
            "Classificação das emoções presentes na voz"
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

        "quantidade_por_emocao": {
            str(emocao): int(quantidade)
            for emocao, quantidade
            in quantidade_por_emocao.items()
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
        "metadados_emocoes.json"
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


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def main():

    print()
    print("=" * 70)
    print("       TREINAMENTO DO MODELO DE RECONHECIMENTO")
    print("                    DE EMOÇÕES")
    print("=" * 70)

    criar_pastas()

    # --------------------------------------------------------
    # Carregar dataset
    # --------------------------------------------------------

    X, y, quantidade_por_emocao = carregar_dataset()

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
    # Treinar modelo
    # --------------------------------------------------------

    modelo = treinar_modelo(
        X_treino,
        y_treino
    )

    # --------------------------------------------------------
    # Avaliar modelo
    # --------------------------------------------------------

    y_pred, acuracia, relatorio = avaliar_modelo(
        modelo,
        X_teste,
        y_teste
    )

    # --------------------------------------------------------
    # Matriz de confusão
    # --------------------------------------------------------

    gerar_matriz_confusao(
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
        quantidade_por_emocao,
        acuracia,
        len(X_treino),
        len(X_teste)
    )

    # --------------------------------------------------------
    # Resultado final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TREINAMENTO DE EMOÇÕES FINALIZADO")
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
        f"- {os.path.join(PASTA_MODELOS, 'metadados_emocoes.json')}"
    )

    print(
        f"- {os.path.join(PASTA_RESULTADOS, 'matriz_confusao_emocoes.png')}"
    )

    print()
    print("✓ Processo concluído.")


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()