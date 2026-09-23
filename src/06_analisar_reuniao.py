import argparse
import csv
import os
import re
import sys
import queue
import signal
import threading
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


def gravar_bloco(stream, parar):
    """Grava um bloco de áudio com a duração configurada."""

    frames = []

    restantes = TAXA_AMOSTRAGEM * DURACAO_BLOCO
    while restantes > 0 and not parar.is_set():
        quantidade = min(TAMANHO_BUFFER, restantes)

        dados = stream.read(
            quantidade,
            exception_on_overflow=True
        )

        frames.append(dados)
        restantes -= quantidade

    return b"".join(frames)


def capturar_blocos(stream, parar, concluida, pendentes, erros,
                    pasta_blocos_audio, tamanho_amostra):
    """Salva WAVs sem aguardar os modelos; a fila contém somente caminhos."""
    amostras = 0
    numero = 1
    try:
        while not parar.is_set():
            dados = gravar_bloco(stream, parar)
            if not dados:
                break
            inicio = amostras / TAXA_AMOSTRAGEM
            amostras += len(dados) // (tamanho_amostra * CANAIS)
            fim = amostras / TAXA_AMOSTRAGEM
            caminho = salvar_audio(dados, numero, pasta_blocos_audio, tamanho_amostra)
            pendentes.put((inicio, fim, caminho))
            numero += 1
    except Exception as erro:
        erros.append(erro)
        parar.set()
    finally:
        concluida.set()


def aguardar_encerramento(entrada, parar):
    """A API envia 'parar'; EOF encerra caso o processo pai saia."""
    for linha in entrada:
        if linha.strip().lower() == "parar":
            break
    parar.set()


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
    parser = argparse.ArgumentParser(description="Análise local de uma reunião")
    parser.add_argument("--automatico", action="store_true",
                        help="Inicia sem Enter; recebe parar pela entrada padrão")
    args = parser.parse_args()
    parar = threading.Event()
    concluida = threading.Event()
    pendentes = queue.Queue()
    erros = []
    audio = None
    stream = None
    captura = None
    sinais_anteriores = {}

    def solicitar_encerramento(signum, frame):
        parar.set()

    try:
        if args.automatico:
            threading.Thread(target=aguardar_encerramento,
                             args=(sys.stdin, parar), daemon=True).start()
        else:
            # O microfone só é aberto depois da confirmação do usuário.
            input("ENTER para começar (CTRL+C durante a captura para encerrar)...")

        for nome in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sinal = getattr(signal, nome, None)
            if sinal is not None:
                sinais_anteriores[sinal] = signal.signal(sinal, solicitar_encerramento)

        if parar.is_set():
            return 0
        modelo_pessoas, modelo_emocoes = carregar_modelos()
        if parar.is_set():
            return 0
        audio = pyaudio.PyAudio()
        tamanho_amostra = audio.get_sample_size(FORMATO)
        stream = audio.open(format=FORMATO, channels=CANAIS,
                            rate=TAXA_AMOSTRAGEM, input=True,
                            frames_per_buffer=TAMANHO_BUFFER, start=False)
        nome_reuniao = obter_proximo_nome_reuniao()
        _, pasta_blocos_audio, arquivo_csv = preparar_reuniao(nome_reuniao)
        print(f"Reunião: {nome_reuniao}")
        print("Análise iniciada. Ao parar, aguarde os blocos pendentes.")
        stream.start_stream()
        captura = threading.Thread(
            target=capturar_blocos,
            args=(stream, parar, concluida, pendentes, erros,
                  pasta_blocos_audio, tamanho_amostra),
        )
        captura.start()

        while not concluida.is_set() or not pendentes.empty():
            try:
                inicio, fim, caminho_audio = pendentes.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                pessoa, conf_pessoa, emocao, conf_emocao = classificar_audio(
                    caminho_audio, modelo_pessoas, modelo_emocoes
                )
            except Exception as erro:
                pessoa, conf_pessoa = "desconhecido", 0.0
                emocao, conf_emocao = "incerto", 0.0
                print(f"Falha ao analisar {caminho_audio}: {erro}")
            registrar_resultado(arquivo_csv, round(inicio, 4), round(fim, 4),
                                pessoa, conf_pessoa, emocao, conf_emocao, caminho_audio)
            print(f"{inicio:.2f}s–{fim:.2f}s: {pessoa} ({conf_pessoa:.1%}), "
                  f"{emocao} ({conf_emocao:.1%})")

        if erros:
            raise RuntimeError(f"Captura interrompida: {erros[0]}")
        print(f"Reunião encerrada. Resultados: {arquivo_csv}")
        print(f"python src/07_gerar_relatorio.py {nome_reuniao}")
        return 0
    except (Exception, KeyboardInterrupt) as erro:
        print(f"ERRO: {erro}")
        return 1
    finally:
        parar.set()
        if captura is not None and captura.ident is not None:
            captura.join()
        try:
            if stream is not None:
                try:
                    stream.stop_stream()
                finally:
                    stream.close()
        finally:
            if audio is not None:
                audio.terminate()
            for sinal, anterior in sinais_anteriores.items():
                signal.signal(sinal, anterior)


if __name__ == "__main__":
    sys.exit(main())
