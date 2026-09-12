import os
import sys
import joblib
import numpy as np

from features import extrair_caracteristicas


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_RAIZ = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PASTA_MODELOS = os.path.join(
    PASTA_RAIZ,
    "modelos"
)

LIMITE_CONFIANCA_PESSOA = 0.45

ARQUIVO_MODELO_PESSOAS = os.path.join(
    PASTA_MODELOS,
    "modelo_pessoas.pkl"
)

ARQUIVO_MODELO_EMOCOES = os.path.join(
    PASTA_MODELOS,
    "modelo_emocoes.pkl"
)


# ============================================================
# CARREGAR MODELOS
# ============================================================

def carregar_modelos():
    """
    Carrega os dois modelos Random Forest treinados.
    """

    print()
    print("=" * 70)
    print("CARREGANDO MODELOS")
    print("=" * 70)

    if not os.path.exists(ARQUIVO_MODELO_PESSOAS):

        raise FileNotFoundError(
            "Modelo de pessoas não encontrado:\n"
            f"{ARQUIVO_MODELO_PESSOAS}"
        )

    if not os.path.exists(ARQUIVO_MODELO_EMOCOES):

        raise FileNotFoundError(
            "Modelo de emoções não encontrado:\n"
            f"{ARQUIVO_MODELO_EMOCOES}"
        )

    modelo_pessoas = joblib.load(
        ARQUIVO_MODELO_PESSOAS
    )

    modelo_emocoes = joblib.load(
        ARQUIVO_MODELO_EMOCOES
    )

    print()
    print("✓ Modelo de pessoas carregado.")
    print("✓ Modelo de emoções carregado.")

    return modelo_pessoas, modelo_emocoes


# ============================================================
# CALCULAR CONFIANÇA
# ============================================================

def calcular_confianca(modelo, caracteristicas):
    """
    Obtém a probabilidade da classe prevista.

    O Random Forest disponibiliza as probabilidades
    através de predict_proba().
    """

    probabilidades = modelo.predict_proba(
        caracteristicas
    )[0]

    indice_maior = np.argmax(
        probabilidades
    )

    confianca = probabilidades[
        indice_maior
    ]

    classe = modelo.classes_[
        indice_maior
    ]

    return classe, float(confianca)


# ============================================================
# CLASSIFICAR ÁUDIO
# ============================================================

def classificar_audio(
    caminho_audio,
    modelo_pessoas,
    modelo_emocoes
):
    """
    Extrai as características do áudio e realiza
    as duas classificações.

    Retorna:

        pessoa
        confianca_pessoa
        emocao
        confianca_emocao
    """

    print()
    print("=" * 70)
    print("ANALISANDO ÁUDIO")
    print("=" * 70)

    print()
    print(f"Arquivo: {caminho_audio}")

    # --------------------------------------------------------
    # Extração das características
    # --------------------------------------------------------

    print()
    print("Extraindo características...")

    caracteristicas = extrair_caracteristicas(
        caminho_audio
    )

    # Random Forest espera uma matriz:
    #
    # (quantidade_amostras, quantidade_caracteristicas)
    #
    # Como estamos analisando apenas um áudio,
    # adicionamos uma dimensão.

    entrada = caracteristicas.reshape(
        1,
        -1
    )

    print(
        f"Características extraídas: "
        f"{len(caracteristicas)}"
    )

    # --------------------------------------------------------
    # Modelo de pessoas
    # --------------------------------------------------------

    pessoa, confianca_pessoa = calcular_confianca(
        modelo_pessoas,
        entrada
    )

    # --------------------------------------------------------
    # Modelo de emoções
    # --------------------------------------------------------

    emocao, confianca_emocao = calcular_confianca(
        modelo_emocoes,
        entrada
    )

    if confianca_pessoa < LIMITE_CONFIANCA_PESSOA:
        pessoa = "desconhecido"

    return (
        pessoa,
        confianca_pessoa,
        emocao,
        confianca_emocao
    )


# ============================================================
# EXIBIR RESULTADO
# ============================================================

def exibir_resultado(
    pessoa,
    confianca_pessoa,
    emocao,
    confianca_emocao
):
    """
    Exibe o resultado da classificação.
    """

    print()
    print("=" * 70)
    print("RESULTADO DA ANÁLISE")
    print("=" * 70)

    print()

    print(
        f"Pessoa identificada: "
        f"{pessoa}"
    )

    print(
        f"Confiança da identificação: "
        f"{confianca_pessoa * 100:.2f}%"
    )

    print()

    print(
        f"Emoção identificada: "
        f"{emocao}"
    )

    print(
        f"Confiança da emoção: "
        f"{confianca_emocao * 100:.2f}%"
    )

    print()
    print("=" * 70)


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def main():

    print()
    print("=" * 70)
    print("       TESTE INTEGRADO DOS MODELOS DE IA")
    print("=" * 70)

    # --------------------------------------------------------
    # Verificar argumento
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print()
        print("Uso:")
        print()
        print(
            "python src/05_testar_modelos.py "
            "caminho_do_audio.wav"
        )

        print()
        print("Exemplo:")
        print()
        print(
            "python src/05_testar_modelos.py "
            "teste.wav"
        )

        return

    caminho_audio = sys.argv[1]

    # --------------------------------------------------------
    # Verificar áudio
    # --------------------------------------------------------

    if not os.path.exists(caminho_audio):

        print()
        print(
            f"ERRO: arquivo não encontrado:"
        )

        print(caminho_audio)

        return

    if not caminho_audio.lower().endswith(".wav"):

        print()
        print(
            "ERRO: o arquivo precisa estar "
            "no formato WAV."
        )

        return

    try:

        # ----------------------------------------------------
        # Carregar modelos
        # ----------------------------------------------------

        modelo_pessoas, modelo_emocoes = carregar_modelos()

        # ----------------------------------------------------
        # Classificar
        # ----------------------------------------------------

        (
            pessoa,
            confianca_pessoa,
            emocao,
            confianca_emocao
        ) = classificar_audio(
            caminho_audio,
            modelo_pessoas,
            modelo_emocoes
        )

        # ----------------------------------------------------
        # Exibir
        # ----------------------------------------------------

        exibir_resultado(
            pessoa,
            confianca_pessoa,
            emocao,
            confianca_emocao
        )

    except Exception as erro:

        print()
        print("=" * 70)
        print("ERRO DURANTE A ANÁLISE")
        print("=" * 70)

        print()
        print(erro)


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()
