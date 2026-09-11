import os
import time
import wave
import random
import pyaudio


# ============================================================
# CONFIGURAÇÕES
# ============================================================

TAXA_AMOSTRAGEM = 16000
CANAIS = 1
FORMATO = pyaudio.paInt16
TAMANHO_BUFFER = 1024

DURACAO_GRAVACAO = 4
GRAVACOES_POR_PESSOA = 30

PASTA_DATASET = "dataset_pessoas"


# ============================================================
# PARTICIPANTES
# ============================================================

PESSOAS = [
    "gabriel",
    "joao",
    "mateus",
    "caio",
    "arthur"
]


# ============================================================
# FRASES
# ============================================================

FRASES = [
    "Hoje vamos discutir o projeto da reunião.",
    "Precisamos analisar os resultados apresentados.",
    "O sistema deverá funcionar corretamente.",
    "A equipe está trabalhando em uma nova solução.",
    "Vamos verificar os dados antes de tomar uma decisão.",
    "O projeto precisa ser testado antes da apresentação.",
    "A reunião começa depois do intervalo.",
    "Precisamos melhorar o desempenho do sistema.",
    "Os resultados serão analisados pela equipe.",
    "O banco de dados possui várias informações.",
    "A tecnologia pode ajudar na resolução do problema.",
    "Vamos organizar as tarefas do projeto.",
    "O relatório será apresentado ao professor.",
    "Precisamos verificar se todos os testes foram realizados.",
    "A equipe terminou uma parte importante do trabalho.",
    "O programa precisa receber alguns ajustes.",
    "Vamos analisar cada etapa do desenvolvimento.",
    "Os dados foram armazenados corretamente.",
    "O treinamento do modelo será realizado hoje.",
    "Precisamos comparar os resultados dos modelos.",
    "A inteligência artificial será utilizada no projeto.",
    "O sistema consegue analisar os arquivos de áudio.",
    "Vamos verificar a qualidade das gravações.",
    "O modelo será avaliado utilizando dados de teste.",
    "A apresentação deverá explicar o funcionamento do projeto.",
    "Precisamos documentar todas as etapas realizadas.",
    "Os integrantes deverão conhecer o funcionamento do código.",
    "O projeto será executado durante a apresentação.",
    "Vamos corrigir os problemas encontrados nos testes.",
    "O resultado final será apresentado para a turma."
]


# ============================================================
# FUNÇÕES
# ============================================================

def criar_pastas():

    os.makedirs(
        PASTA_DATASET,
        exist_ok=True
    )

    for pessoa in PESSOAS:

        pasta_pessoa = os.path.join(
            PASTA_DATASET,
            pessoa
        )

        os.makedirs(
            pasta_pessoa,
            exist_ok=True
        )


def obter_proximo_numero(pessoa):

    pasta = os.path.join(
        PASTA_DATASET,
        pessoa
    )

    numeros = []

    for arquivo in os.listdir(pasta):

        if not arquivo.lower().endswith(".wav"):
            continue

        nome = os.path.splitext(
            arquivo
        )[0]

        partes = nome.split("_")

        if len(partes) >= 2:

            try:

                numero = int(
                    partes[-1]
                )

                numeros.append(numero)

            except ValueError:
                pass

    if not numeros:
        return 1

    return max(numeros) + 1


def escolher_frase():

    return random.choice(FRASES)


def gravar_audio(pessoa, numero, frase):

    arquivo_saida = os.path.join(
        PASTA_DATASET,
        pessoa,
        f"{pessoa}_{numero:03d}.wav"
    )

    print()
    print("=" * 60)
    print("PREPARAÇÃO PARA GRAVAÇÃO")
    print("=" * 60)

    print()
    print(f"Pessoa: {pessoa}")
    print(
        f"Gravação: {numero}/{GRAVACOES_POR_PESSOA}"
    )

    print()
    print("Frase:")
    print(f'"{frase}"')

    print()
    print("Pressione ENTER quando estiver pronto.")

    input()

    print()
    print("Prepare-se...")

    for contagem in range(3, 0, -1):

        print(contagem)

        time.sleep(1)

    print()
    print(">>> FALE AGORA <<<")
    print()

    audio = pyaudio.PyAudio()

    try:

        # Obtém antes de encerrar o PyAudio
        tamanho_amostra = audio.get_sample_size(
            FORMATO
        )

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
        print()
        print(f"Detalhes: {erro}")

        audio.terminate()

        return False

    frames = []

    quantidade_blocos = int(
        TAXA_AMOSTRAGEM
        / TAMANHO_BUFFER
        * DURACAO_GRAVACAO
    )

    try:

        for _ in range(quantidade_blocos):

            dados = stream.read(
                TAMANHO_BUFFER,
                exception_on_overflow=False
            )

            frames.append(dados)

    except Exception as erro:

        print()
        print(
            f"Erro durante a gravação: {erro}"
        )

        stream.stop_stream()
        stream.close()
        audio.terminate()

        return False

    print()
    print(">>> GRAVAÇÃO FINALIZADA <<<")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    try:

        with wave.open(
            arquivo_saida,
            "wb"
        ) as arquivo:

            arquivo.setnchannels(
                CANAIS
            )

            arquivo.setsampwidth(
                tamanho_amostra
            )

            arquivo.setframerate(
                TAXA_AMOSTRAGEM
            )

            arquivo.writeframes(
                b"".join(frames)
            )

    except Exception as erro:

        print()
        print("ERRO AO SALVAR O ARQUIVO.")
        print(
            f"Detalhes: {erro}"
        )

        return False

    print()
    print("✓ Áudio salvo com sucesso!")
    print(
        f"Arquivo: {arquivo_saida}"
    )

    return True


def mostrar_progresso():

    print()
    print("=" * 60)
    print("PROGRESSO DO DATASET")
    print("=" * 60)

    for pessoa in PESSOAS:

        pasta = os.path.join(
            PASTA_DATASET,
            pessoa
        )

        quantidade = 0

        if os.path.exists(pasta):

            quantidade = len([
                arquivo
                for arquivo in os.listdir(pasta)
                if arquivo.lower().endswith(".wav")
            ])

        print(
            f"{pessoa:<15} "
            f"{quantidade:02d}/{GRAVACOES_POR_PESSOA}"
        )

    print("=" * 60)


def gravar_pessoa(pessoa):

    numero_inicial = obter_proximo_numero(
        pessoa
    )

    if numero_inicial > GRAVACOES_POR_PESSOA:

        print()
        print(
            f"O dataset de {pessoa} "
            f"já possui {GRAVACOES_POR_PESSOA} gravações."
        )

        return

    print()
    print("=" * 60)
    print(
        f"INICIANDO DATASET: {pessoa.upper()}"
    )
    print("=" * 60)

    for numero in range(
        numero_inicial,
        GRAVACOES_POR_PESSOA + 1
    ):

        frase = escolher_frase()

        sucesso = gravar_audio(
            pessoa,
            numero,
            frase
        )

        if not sucesso:

            print()
            print(
                "A gravação não pôde ser concluída."
            )

            return

        time.sleep(1)

    print()
    print("=" * 60)
    print(
        f"DATASET DE {pessoa.upper()} FINALIZADO!"
    )
    print("=" * 60)


def menu():

    while True:

        print()
        print("=" * 60)
        print(
            "DATASET DE PESSOAS - RECONHECIMENTO DE VOZ"
        )
        print("=" * 60)

        print()
        print("Participantes:")

        for indice, pessoa in enumerate(
            PESSOAS,
            start=1
        ):

            print(
                f"{indice} - {pessoa}"
            )

        print()
        print("6 - Mostrar progresso")
        print("0 - Encerrar")
        print()

        opcao = input(
            "Escolha uma opção: "
        ).strip()

        if opcao == "0":

            print()
            print("Programa encerrado.")

            break

        elif opcao == "6":

            mostrar_progresso()

        elif opcao in [
            "1",
            "2",
            "3",
            "4",
            "5"
        ]:

            indice = int(opcao) - 1

            pessoa = PESSOAS[indice]

            gravar_pessoa(
                pessoa
            )

        else:

            print()
            print("Opção inválida.")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print(
        "SISTEMA DE GRAVAÇÃO DO DATASET DE PESSOAS"
    )
    print("=" * 60)

    print()
    print("Configurações:")
    print(
        f"- Taxa de amostragem: "
        f"{TAXA_AMOSTRAGEM} Hz"
    )

    print(
        f"- Canais: {CANAIS} (mono)"
    )

    print(
        f"- Duração: "
        f"{DURACAO_GRAVACAO} segundos"
    )

    print(
        f"- Gravações por pessoa: "
        f"{GRAVACOES_POR_PESSOA}"
    )

    print()

    criar_pastas()

    menu()
