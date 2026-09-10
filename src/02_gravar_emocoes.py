import os
import random
import time
import wave

import pyaudio


# ============================================================
# CONFIGURAÇÕES
# ============================================================

TAXA_AMOSTRAGEM = 16000
CANAIS = 1
FORMATO = pyaudio.paInt16
TAMANHO_BUFFER = 1024

DURACAO = 4

# O professor exige no mínimo 25 por emoção.
# Vamos usar 30 para ter uma margem.
NUM_GRAVACOES_POR_EMOCAO = 30

PASTA_DATASET = "dataset_emocoes"


EMOCOES = [
    "alegre",
    "neutro",
    "triste",
    "irritado"
]


# ============================================================
# FRASES
# ============================================================

FRASES = [
    "Hoje teremos uma reunião importante.",
    "Precisamos analisar os resultados.",
    "O projeto está avançando conforme planejado.",
    "Vamos conversar sobre os próximos passos.",
    "Precisamos resolver esse problema.",
    "A equipe terminou essa atividade.",
    "Podemos começar a reunião agora.",
    "Esse resultado precisa ser analisado.",
    "Temos algumas tarefas para concluir.",
    "O prazo está chegando ao fim.",
    "Precisamos melhorar esse processo.",
    "Vamos verificar os dados novamente.",
    "A reunião começa neste momento.",
    "Essa atividade será concluída hoje.",
    "Precisamos organizar as informações.",
    "O trabalho da equipe foi finalizado.",
    "Vamos discutir essa situação.",
    "Temos uma decisão importante para tomar.",
    "Precisamos encontrar uma solução.",
    "O resultado foi diferente do esperado.",
    "Vamos continuar com o planejamento.",
    "Essa questão precisa de atenção.",
    "Temos novos dados para analisar.",
    "Precisamos conversar sobre esse assunto.",
    "O sistema está funcionando corretamente.",
    "Vamos revisar o que foi feito.",
    "Essa tarefa precisa ser concluída.",
    "Podemos seguir para o próximo assunto.",
    "Precisamos avaliar os resultados.",
    "A equipe pode continuar o trabalho."
]


# ============================================================
# PREPARAÇÃO DAS PASTAS
# ============================================================

os.makedirs(PASTA_DATASET, exist_ok=True)

for emocao in EMOCOES:

    pasta_emocao = os.path.join(
        PASTA_DATASET,
        emocao
    )

    os.makedirs(
        pasta_emocao,
        exist_ok=True
    )


# ============================================================
# FUNÇÕES
# ============================================================

def contar_gravacoes(emocao):
    """
    Conta quantos arquivos WAV existem
    na pasta da emoção.
    """

    pasta = os.path.join(
        PASTA_DATASET,
        emocao
    )

    arquivos = [
        arquivo
        for arquivo in os.listdir(pasta)
        if arquivo.lower().endswith(".wav")
    ]

    return len(arquivos)


def proximo_numero(emocao):
    """
    Descobre o próximo número disponível
    para o arquivo da emoção.
    """

    pasta = os.path.join(
        PASTA_DATASET,
        emocao
    )

    numeros = []

    for arquivo in os.listdir(pasta):

        if not arquivo.lower().endswith(".wav"):
            continue

        nome = os.path.splitext(
            arquivo
        )[0]

        partes = nome.split("_")

        if len(partes) != 2:
            continue

        try:

            numero = int(partes[1])

            numeros.append(numero)

        except ValueError:

            continue

    if not numeros:
        return 1

    return max(numeros) + 1


def gravar_audio(audio, stream):
    """
    Grava um áudio de DURACAO segundos.
    """

    frames = []

    quantidade_blocos = int(
        TAXA_AMOSTRAGEM
        / TAMANHO_BUFFER
        * DURACAO
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
    caminho
):
    """
    Salva os dados gravados no formato WAV.
    """

    arquivo = wave.open(
        caminho,
        "wb"
    )

    arquivo.setnchannels(
        CANAIS
    )

    arquivo.setsampwidth(
        audio.get_sample_size(FORMATO)
    )

    arquivo.setframerate(
        TAXA_AMOSTRAGEM
    )

    arquivo.writeframes(
        dados
    )

    arquivo.close()


def mostrar_progresso():
    """
    Mostra a quantidade de gravações
    de cada emoção.
    """

    print()
    print("=" * 55)
    print("PROGRESSO DO DATASET DE EMOÇÕES")
    print("=" * 55)

    total = 0

    for emocao in EMOCOES:

        quantidade = contar_gravacoes(
            emocao
        )

        total += quantidade

        status = "OK" if quantidade >= 25 else "PENDENTE"

        print(
            f"{emocao:<10} "
            f"{quantidade:>3} gravações "
            f"[{status}]"
        )

    print("-" * 55)

    print(
        f"Total: {total} gravações"
    )

    print("=" * 55)
    print()


def gravar_emocao(emocao):
    """
    Realiza as gravações de uma emoção.
    """

    quantidade_atual = contar_gravacoes(
        emocao
    )

    if quantidade_atual >= NUM_GRAVACOES_POR_EMOCAO:

        print()
        print(
            f"A emoção '{emocao}' "
            f"já possui {quantidade_atual} gravações."
        )

        resposta = input(
            "Deseja gravar mais mesmo assim? "
            "(s/n): "
        ).strip().lower()

        if resposta != "s":
            return

    numero = proximo_numero(
        emocao
    )

    print()
    print("=" * 60)
    print(
        f"GRAVAÇÃO DE EMOÇÃO: {emocao.upper()}"
    )
    print("=" * 60)

    print()
    print(
        "IMPORTANTE:"
    )

    print(
        "• Fale naturalmente."
    )

    print(
        "• Mantenha uma distância semelhante do microfone."
    )

    print(
        "• Não deixe música ou outras pessoas falando ao fundo."
    )

    print(
        "• Tente representar a emoção indicada."
    )

    print(
        "• Não altere a frase durante a gravação."
    )

    print()

    input(
        "Pressione ENTER quando estiver pronto..."
    )

    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=FORMATO,
        channels=CANAIS,
        rate=TAXA_AMOSTRAGEM,
        input=True,
        frames_per_buffer=TAMANHO_BUFFER
    )

    try:

        while True:

            if numero > (
                quantidade_atual
                + NUM_GRAVACOES_POR_EMOCAO
            ):

                break

            frase = random.choice(
                FRASES
            )

            print()
            print("-" * 60)

            print(
                f"Gravação {numero:03d}"
            )

            print(
                f"Emoção: {emocao.upper()}"
            )

            print()
            print(
                f'FRASE: "{frase}"'
            )

            print()

            input(
                "Pressione ENTER para gravar..."
            )

            print()
            print(
                "GRAVANDO..."
            )

            dados = gravar_audio(
                audio,
                stream
            )

            nome_arquivo = (
                f"{emocao}_{numero:03d}.wav"
            )

            caminho = os.path.join(
                PASTA_DATASET,
                emocao,
                nome_arquivo
            )

            salvar_audio(
                dados,
                caminho
            )

            print(
                f"Salvo: {caminho}"
            )

            numero += 1

            quantidade_atual += 1

            print()

            if quantidade_atual >= (
                NUM_GRAVACOES_POR_EMOCAO
            ):

                print(
                    f"Meta de {NUM_GRAVACOES_POR_EMOCAO} "
                    f"gravações atingida."
                )

                break

            resposta = input(
                "Deseja continuar gravando essa emoção? "
                "(s/n): "
            ).strip().lower()

            if resposta != "s":

                break

    finally:

        stream.stop_stream()
        stream.close()

        audio.terminate()

    print()
    print(
        f"Gravações de '{emocao}' finalizadas."
    )


# ============================================================
# MENU PRINCIPAL
# ============================================================

while True:

    print()
    print("=" * 60)
    print("       DATASET DE EMOÇÕES")
    print("       SALA DE REUNIÃO INTELIGENTE")
    print("=" * 60)

    print()

    for indice, emocao in enumerate(
        EMOCOES,
        start=1
    ):

        quantidade = contar_gravacoes(
            emocao
        )

        print(
            f"{indice} - {emocao:<10} "
            f"({quantidade}/{NUM_GRAVACOES_POR_EMOCAO})"
        )

    print()
    print("5 - Mostrar progresso")
    print("0 - Sair")

    print()

    opcao = input(
        "Escolha uma opção: "
    ).strip()

    if opcao in ["1", "2", "3", "4"]:

        indice = int(opcao) - 1

        emocao_escolhida = EMOCOES[
            indice
        ]

        gravar_emocao(
            emocao_escolhida
        )

    elif opcao == "5":

        mostrar_progresso()

    elif opcao == "0":

        print()
        print(
            "Encerrando programa..."
        )

        break

    else:

        print()
        print(
            "Opção inválida."
        )


# ============================================================
# RESUMO FINAL
# ============================================================

print()
mostrar_progresso()

print(
    "Dataset salvo em:"
)

print(
    os.path.abspath(
        PASTA_DATASET
    )
)

print()
print(
    "Quando todas as emoções tiverem pelo menos"
)

print(
    "25 gravações, o dataset atende ao requisito mínimo."
)