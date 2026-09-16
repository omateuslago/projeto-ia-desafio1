import csv
import os
import re
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
TAMANHO_BUFFER = 1024
DURACAO_BLOCO = 4

LIMITE_CONFIANCA_PESSOA = 0.45

PASTA_REUNIOES = os.path.join(PASTA_RAIZ, "reunioes")
PASTA_MODELOS = os.path.join(PASTA_RAIZ, "modelos")

MODELO_PESSOAS = os.path.join(
    PASTA_MODELOS,
    "modelo_pessoas.pkl"
)

MODELO_EMOCOES = os.path.join(
    PASTA_MODELOS,
    "modelo_emocoes.pkl"
)


def obter_proximo_nome_reuniao():
    """Retorna um nome sequencial sem reutilizar reuniões anteriores."""

    if not os.path.exists(PASTA_REUNIOES):
        return "reuniao_01"

    numeros = []

    for nome in os.listdir(PASTA_REUNIOES):

        correspondencia = re.fullmatch(r"reuniao_(\d+)", nome)

        if correspondencia:
            numeros.append(int(correspondencia.group(1)))

    proximo_numero = max(numeros, default=0) + 1

    return f"reuniao_{proximo_numero:02d}"


def carregar_modelos():
    """Verifica e carrega os dois modelos exigidos pelo projeto."""

    if not os.path.exists(MODELO_PESSOAS):
        raise FileNotFoundError(
            "Modelo de pessoas não encontrado:\n"
            f"{MODELO_PESSOAS}"
        )

    if not os.path.exists(MODELO_EMOCOES):
        raise FileNotFoundError(
            "Modelo de emoções não encontrado:\n"
            f"{MODELO_EMOCOES}"
        )

    print("Carregando modelos...")

    modelo_pessoas = joblib.load(MODELO_PESSOAS)
    modelo_emocoes = joblib.load(MODELO_EMOCOES)

    print("Modelos carregados com sucesso.")

    return modelo_pessoas, modelo_emocoes


def preparar_reuniao(nome_reuniao):
    """Cria a estrutura obrigatória da reunião e o CSV."""

    pasta_reuniao = os.path.join(PASTA_REUNIOES, nome_reuniao)
    pasta_blocos_audio = os.path.join(pasta_reuniao, "blocos_audio")
    arquivo_csv = os.path.join(pasta_reuniao, "registros.csv")

    os.makedirs(pasta_blocos_audio, exist_ok=False)

    with open(
        arquivo_csv,
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

    return pasta_reuniao, pasta_blocos_audio, arquivo_csv


def gravar_bloco(stream):
    """Grava um bloco de áudio com a duração configurada."""

    frames = []

    quantidade_blocos = int(
        TAXA_AMOSTRAGEM / TAMANHO_BUFFER * DURACAO_BLOCO
    )

    for _ in range(quantidade_blocos):

        dados = stream.read(
            TAMANHO_BUFFER,
            exception_on_overflow=False
        )

        frames.append(dados)

    return b"".join(frames)


def salvar_audio(
    dados,
    numero_bloco,
    pasta_blocos_audio,
    tamanho_amostra
):
    """Salva um bloco em WAV dentro da reunião atual."""

    nome_arquivo = f"bloco_{numero_bloco:03d}.wav"
    caminho = os.path.join(pasta_blocos_audio, nome_arquivo)

    with wave.open(caminho, "wb") as arquivo:

        arquivo.setnchannels(CANAIS)
        arquivo.setsampwidth(tamanho_amostra)
        arquivo.setframerate(TAXA_AMOSTRAGEM)
        arquivo.writeframes(dados)

    return caminho


def classificar_audio(
    caminho_audio,
    modelo_pessoas,
    modelo_emocoes
):
    """Identifica a pessoa e estima a emoção vocal."""

    caracteristicas = extrair_caracteristicas(caminho_audio)

    entrada = np.array(
        [caracteristicas],
        dtype=np.float32
    )

    probabilidades_pessoa = modelo_pessoas.predict_proba(entrada)[0]
    indice_pessoa = np.argmax(probabilidades_pessoa)
    pessoa = modelo_pessoas.classes_[indice_pessoa]
    confianca_pessoa = float(probabilidades_pessoa[indice_pessoa])

    probabilidades_emocao = modelo_emocoes.predict_proba(entrada)[0]
    indice_emocao = np.argmax(probabilidades_emocao)
    emocao = modelo_emocoes.classes_[indice_emocao]
    confianca_emocao = float(probabilidades_emocao[indice_emocao])

    if confianca_pessoa < LIMITE_CONFIANCA_PESSOA:
        pessoa = "desconhecido"

    return pessoa, confianca_pessoa, emocao, confianca_emocao


def registrar_resultado(
    arquivo_csv,
    inicio,
    fim,
    pessoa,
    confianca_pessoa,
    emocao,
    confianca_emocao,
    caminho_audio
):
    """Registra todo bloco gravado, inclusive uma análise incerta."""

    caminho_relativo = os.path.relpath(caminho_audio, PASTA_RAIZ)

    with open(
        arquivo_csv,
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
            caminho_relativo
        ])


def main():

    try:

        modelo_pessoas, modelo_emocoes = carregar_modelos()

    except Exception as erro:

        print()
        print("ERRO AO CARREGAR OS MODELOS.")
        print(erro)

        return 1

    audio = pyaudio.PyAudio()

    try:

        tamanho_amostra = audio.get_sample_size(FORMATO)

        stream = audio.open(
            format=FORMATO,
            channels=CANAIS,
            rate=TAXA_AMOSTRAGEM,
            input=True,
            frames_per_buffer=TAMANHO_BUFFER
        )

    except Exception as erro:

        print()
        print("ERRO AO ACESSAR O MICROFONE.")
        print(erro)

        audio.terminate()

        return 1

    nome_reuniao = obter_proximo_nome_reuniao()

    try:

        (
            pasta_reuniao,
            pasta_blocos_audio,
            arquivo_csv
        ) = preparar_reuniao(nome_reuniao)

    except Exception as erro:

        print()
        print("ERRO AO PREPARAR A PASTA DA REUNIÃO.")
        print(erro)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        return 1

    print()
    print("=" * 60)
    print("        SALA DE REUNIÃO INTELIGENTE")
    print("=" * 60)
    print()
    print(f"Reunião: {nome_reuniao}")
    print(f"Cada bloco terá {DURACAO_BLOCO} segundos.")
    print()
    print("Pressione ENTER para iniciar a análise.")
    print("Durante a reunião, pressione CTRL+C para finalizar.")
    print()

    numero_bloco = 1

    try:

        input("ENTER para começar...")

        print()
        print("Análise iniciada!")
        print("Fale normalmente, uma pessoa por vez.")
        print()

        inicio_reuniao = time.time()

        while True:

            print(f"[Bloco {numero_bloco:03d}] Gravando...")

            inicio = time.time()
            dados_audio = gravar_bloco(stream)
            fim = time.time()

            caminho_audio = salvar_audio(
                dados_audio,
                numero_bloco,
                pasta_blocos_audio,
                tamanho_amostra
            )

            inicio_formatado = round(inicio - inicio_reuniao, 2)
            fim_formatado = round(fim - inicio_reuniao, 2)

            print("  Áudio gravado.")
            print("  Analisando...")

            try:

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

            except Exception as erro:

                pessoa = "desconhecido"
                confianca_pessoa = 0.0
                emocao = "incerto"
                confianca_emocao = 0.0

                print(
                    "  A análise falhou; o bloco será "
                    "registrado como incerto."
                )
                print(f"  Motivo: {erro}")

            registrar_resultado(
                arquivo_csv,
                inicio_formatado,
                fim_formatado,
                pessoa,
                confianca_pessoa,
                emocao,
                confianca_emocao,
                caminho_audio
            )

            print(f"  Pessoa: {pessoa} ({confianca_pessoa:.1%})")
            print(
                f"  Emoção estimada: {emocao} "
                f"({confianca_emocao:.1%})"
            )
            print(f"  Resultado salvo em: {arquivo_csv}")
            print()

            numero_bloco += 1

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
    print(f"Resultados salvos em:\n{arquivo_csv}")
    print(f"\nÁudios salvos em:\n{pasta_blocos_audio}")
    print()
    print("Próxima etapa:")
    print(
        "python src/07_gerar_relatorio.py "
        f"{nome_reuniao}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
