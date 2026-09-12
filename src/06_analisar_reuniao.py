import csv
import os
import sys
import time
import wave

import joblib
import numpy as np
import pyaudio

from features import extrair_caracteristicas


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_RAIZ = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TAXA_AMOSTRAGEM = 16000
CANAIS = 1
FORMATO = pyaudio.paInt16
DURACAO_BLOCO = 4

LIMITE_CONFIANCA_PESSOA = 0.45

PASTA_REUNIOES = os.path.join(
    PASTA_RAIZ,
    "reunioes"
)
NOME_REUNIAO = "reuniao_01"
PASTA_REUNIAO = os.path.join(
    PASTA_REUNIOES,
    NOME_REUNIAO
)
PASTA_BLOCOS_AUDIO = os.path.join(
    PASTA_REUNIAO,
    "blocos_audio"
)
PASTA_MODELOS = os.path.join(
    PASTA_RAIZ,
    "modelos"
)

ARQUIVO_CSV = os.path.join(
    PASTA_REUNIAO,
    "registros.csv"
)

MODELO_PESSOAS = os.path.join(
    PASTA_MODELOS,
    "modelo_pessoas.pkl"
)

MODELO_EMOCOES = os.path.join(
    PASTA_MODELOS,
    "modelo_emocoes.pkl"
)


# ============================================================
# PREPARAÇÃO
# ============================================================

os.makedirs(PASTA_BLOCOS_AUDIO, exist_ok=True)

if not os.path.exists(MODELO_PESSOAS):
    print("ERRO: modelo de pessoas não encontrado.")
    print(f"Esperado em: {MODELO_PESSOAS}")
    sys.exit(1)

if not os.path.exists(MODELO_EMOCOES):
    print("ERRO: modelo de emoções não encontrado.")
    print(f"Esperado em: {MODELO_EMOCOES}")
    sys.exit(1)


print("Carregando modelos...")

modelo_pessoas = joblib.load(MODELO_PESSOAS)
modelo_emocoes = joblib.load(MODELO_EMOCOES)

print("Modelos carregados com sucesso.")


# ============================================================
# MICROFONE
# ============================================================

audio = pyaudio.PyAudio()

stream = audio.open(
    format=FORMATO,
    channels=CANAIS,
    rate=TAXA_AMOSTRAGEM,
    input=True,
    frames_per_buffer=1024
)


# ============================================================
# FUNÇÕES
# ============================================================

def gravar_bloco():
    """
    Grava um bloco de áudio de DURACAO_BLOCO segundos.
    """

    frames = []

    quantidade_blocos = int(
        TAXA_AMOSTRAGEM / 1024 * DURACAO_BLOCO
    )

    for _ in range(quantidade_blocos):
        dados = stream.read(
            1024,
            exception_on_overflow=False
        )
        frames.append(dados)

    return b"".join(frames)


def salvar_audio(dados, numero_bloco):
    """
    Salva o bloco de áudio como WAV.
    """

    nome_arquivo = f"bloco_{numero_bloco:03d}.wav"

    caminho = os.path.join(
        PASTA_BLOCOS_AUDIO,
        nome_arquivo
    )

    arquivo = wave.open(caminho, "wb")

    arquivo.setnchannels(CANAIS)
    arquivo.setsampwidth(audio.get_sample_size(FORMATO))
    arquivo.setframerate(TAXA_AMOSTRAGEM)
    arquivo.writeframes(dados)

    arquivo.close()

    return caminho


def classificar_audio(caminho_audio):
    """
    Classifica o mesmo trecho de áudio usando:

    1. Modelo de pessoa
    2. Modelo de emoção
    """

    caracteristicas = extrair_caracteristicas(
        caminho_audio
    )

    X = np.array(
        [caracteristicas],
        dtype=np.float32
    )

    # --------------------------------------------------------
    # PESSOA
    # --------------------------------------------------------

    probabilidades_pessoa = (
        modelo_pessoas.predict_proba(X)[0]
    )

    indice_pessoa = np.argmax(
        probabilidades_pessoa
    )

    pessoa = modelo_pessoas.classes_[
        indice_pessoa
    ]

    confianca_pessoa = float(
        probabilidades_pessoa[indice_pessoa]
    )

    # --------------------------------------------------------
    # EMOÇÃO
    # --------------------------------------------------------

    probabilidades_emocao = (
        modelo_emocoes.predict_proba(X)[0]
    )

    indice_emocao = np.argmax(
        probabilidades_emocao
    )

    emocao = modelo_emocoes.classes_[
        indice_emocao
    ]

    confianca_emocao = float(
        probabilidades_emocao[indice_emocao]
    )

    # --------------------------------------------------------
    # DESCONHECIDO
    # --------------------------------------------------------

    if confianca_pessoa < LIMITE_CONFIANCA_PESSOA:
        pessoa = "desconhecido"

    return (
        pessoa,
        confianca_pessoa,
        emocao,
        confianca_emocao
    )


def preparar_csv():
    """
    Cria o CSV com o cabeçalho exigido pelo projeto,
    caso ele ainda não exista.
    """

    if not os.path.exists(ARQUIVO_CSV):

        with open(
            ARQUIVO_CSV,
            "w",
            newline="",
            encoding="utf-8"
        ) as arquivo:

            escritor = csv.writer(arquivo)

            escritor.writerow([
                "inicio",
                "fim",
                "pessoa",
                "conf_pessoa",
                "emocao",
                "conf_emocao",
                "arquivo"
            ])


def registrar_resultado(
    inicio,
    fim,
    pessoa,
    confianca_pessoa,
    emocao,
    confianca_emocao,
    caminho_audio
):
    """
    Adiciona o resultado de um trecho ao CSV.
    """

    with open(
        ARQUIVO_CSV,
        "a",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        escritor = csv.writer(arquivo)

        escritor.writerow([
            inicio,
            fim,
            pessoa,
            round(confianca_pessoa, 4),
            emocao,
            round(confianca_emocao, 4),
            caminho_audio
        ])


# ============================================================
# INÍCIO DA REUNIÃO
# ============================================================

preparar_csv()

print()
print("=" * 60)
print("        SALA DE REUNIÃO INTELIGENTE")
print("=" * 60)
print()
print(f"Cada trecho terá {DURACAO_BLOCO} segundos.")
print()
print("Pressione ENTER para iniciar a análise.")
print("Durante a reunião, pressione CTRL+C para finalizar.")
print()

input("ENTER para começar...")

print()
print("Análise iniciada!")
print("Fale normalmente, uma pessoa por vez.")
print("Pressione CTRL+C quando quiser encerrar.")
print()


numero_bloco = 1
inicio_reuniao = time.time()


# ============================================================
# LOOP DA REUNIÃO
# ============================================================

try:

    while True:

        print(
            f"[Trecho {numero_bloco:04d}] "
            f"Gravando..."
        )

        inicio = time.time()

        dados_audio = gravar_bloco()

        fim = time.time()

        caminho_audio = salvar_audio(
            dados_audio,
            numero_bloco
        )

        print("  Áudio gravado.")
        print("  Analisando...")

        try:

            (
                pessoa,
                confianca_pessoa,
                emocao,
                confianca_emocao
            ) = classificar_audio(
                caminho_audio
            )

            inicio_formatado = (
                inicio - inicio_reuniao
            )

            fim_formatado = (
                fim - inicio_reuniao
            )

            registrar_resultado(
                round(inicio_formatado, 2),
                round(fim_formatado, 2),
                pessoa,
                confianca_pessoa,
                emocao,
                confianca_emocao,
                caminho_audio
            )

            print(
                f"  Pessoa: {pessoa} "
                f"({confianca_pessoa:.1%})"
            )

            print(
                f"  Emoção estimada: {emocao} "
                f"({confianca_emocao:.1%})"
            )

            if pessoa == "desconhecido":
                print(
                    "  ⚠️ Confiança da identificação "
                    "abaixo do limite."
                )

            print(
                f"  Resultado salvo em: {ARQUIVO_CSV}"
            )

        except Exception as erro:

            print(
                "  ERRO ao analisar o trecho:"
            )

            print(f"  {erro}")

        print()

        numero_bloco += 1


# ============================================================
# FINALIZAÇÃO
# ============================================================

except KeyboardInterrupt:

    print()
    print("=" * 60)
    print("Reunião encerrada.")
    print("=" * 60)

finally:

    stream.stop_stream()
    stream.close()

    audio.terminate()

    print()
    print(
        f"Resultados salvos em:"
        f"\n{ARQUIVO_CSV}"
    )

    print(
        f"Áudios salvos em:"
        f"\n{PASTA_BLOCOS_AUDIO}"
    )

    print()
    print(
        "Próxima etapa: geração automática do relatório."
    )
