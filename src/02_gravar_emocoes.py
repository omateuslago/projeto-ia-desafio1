import os
import random
import time
import wave

import pyaudio


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

DURACAO_GRAVACAO = 4

GRAVACOES_POR_PESSOA_EMOCAO = 6

PASTA_DATASET = os.path.join(
    PASTA_RAIZ,
    "dataset_emocoes"
)


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
# EMOÇÕES
# ============================================================

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
# PREPARAÇÃO
# ============================================================

def criar_pastas():

    os.makedirs(
        PASTA_DATASET,
        exist_ok=True
    )

    for emocao in EMOCOES:

        pasta = os.path.join(
            PASTA_DATASET,
            emocao
        )

        os.makedirs(
            pasta,
            exist_ok=True
        )


# ============================================================
# CONTAGEM
# ============================================================

def contar_gravacoes(pessoa, emocao):

    pasta = os.path.join(
        PASTA_DATASET,
        emocao
    )

    prefixo = f"{pessoa}_{emocao}_"

    if not os.path.exists(pasta):
        return 0

    arquivos = [
        arquivo
        for arquivo in os.listdir(pasta)
        if arquivo.lower().endswith(".wav")
        and arquivo.startswith(prefixo)
    ]

    return len(arquivos)


# ============================================================
# PRÓXIMO NÚMERO
# ============================================================

def proximo_numero(pessoa, emocao):

    pasta = os.path.join(
        PASTA_DATASET,
        emocao
    )

    prefixo = f"{pessoa}_{emocao}_"

    numeros = []

    for arquivo in os.listdir(pasta):

        if not arquivo.lower().endswith(".wav"):
            continue

        if not arquivo.startswith(prefixo):
            continue

        nome = os.path.splitext(
            arquivo
        )[0]

        partes = nome.split("_")

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


# ============================================================
# GRAVAÇÃO
# ============================================================

def gravar_audio():

    audio = pyaudio.PyAudio()

    try:

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
        print(
            f"Detalhes: {erro}"
        )

        audio.terminate()

        return None, None

    frames = []

    quantidade_blocos = int(
        TAXA_AMOSTRAGEM
        / TAMANHO_BUFFER
        * DURACAO_GRAVACAO
    )

    try:

        for _ in range(
            quantidade_blocos
        ):

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

        return None, None

    stream.stop_stream()
    stream.close()
    audio.terminate()

    return (
        b"".join(frames),
        tamanho_amostra
    )


# ============================================================
# SALVAR
# ============================================================

def salvar_audio(
    dados,
    tamanho_amostra,
    caminho
):

    with wave.open(
        caminho,
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
            dados
        )


# ============================================================
# GRAVAR UMA EMOÇÃO
# ============================================================

def gravar_emocao(
    pessoa,
    emocao
):

    quantidade_atual = contar_gravacoes(
        pessoa,
        emocao
    )

    if quantidade_atual >= (
        GRAVACOES_POR_PESSOA_EMOCAO
    ):

        print()
        print(
            f"{pessoa} já possui "
            f"{quantidade_atual} gravações "
            f"da emoção '{emocao}'."
        )

        return

    numero = proximo_numero(
        pessoa,
        emocao
    )

    faltam = (
        GRAVACOES_POR_PESSOA_EMOCAO
        - quantidade_atual
    )

    print()
    print("=" * 60)
    print(
        f"PARTICIPANTE: {pessoa.upper()}"
    )
    print(
        f"EMOÇÃO: {emocao.upper()}"
    )
    print("=" * 60)

    print()
    print(
        f"Serão feitas {faltam} gravações."
    )

    print()
    print(
        "IMPORTANTE:"
    )

    print(
        "• Mantenha distância semelhante do microfone."
    )

    print(
        "• Grave em ambiente silencioso."
    )

    print(
        "• Não deixe outras pessoas falando ao fundo."
    )

    print(
        "• Procure representar vocalmente a emoção."
    )

    print()

    for _ in range(faltam):

        frase = random.choice(
            FRASES
        )

        print()
        print("-" * 60)

        print(
            f"Gravação "
            f"{numero:03d}/{GRAVACOES_POR_PESSOA_EMOCAO}"
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
        print("Prepare-se...")

        for contagem in range(
            3,
            0,
            -1
        ):

            print(contagem)

            time.sleep(1)

        print()
        print(">>> GRAVANDO <<<")

        dados, tamanho_amostra = gravar_audio()

        if dados is None:

            print()
            print(
                "A gravação foi interrompida."
            )

            return

        nome_arquivo = (
            f"{pessoa}_{emocao}_{numero:03d}.wav"
        )

        caminho = os.path.join(
            PASTA_DATASET,
            emocao,
            nome_arquivo
        )

        salvar_audio(
            dados,
            tamanho_amostra,
            caminho
        )

        print()
        print(
            "✓ Áudio salvo:"
        )

        print(caminho)

        numero += 1

        time.sleep(1)

    print()
    print("=" * 60)
    print(
        f"✓ {pessoa.upper()} - "
        f"{emocao.upper()} FINALIZADO"
    )
    print("=" * 60)


# ============================================================
# PROGRESSO
# ============================================================

def mostrar_progresso():

    print()
    print("=" * 70)
    print("PROGRESSO DO DATASET DE EMOÇÕES")
    print("=" * 70)

    total = 0

    for pessoa in PESSOAS:

        print()
        print(
            pessoa.upper()
        )

        for emocao in EMOCOES:

            quantidade = contar_gravacoes(
                pessoa,
                emocao
            )

            total += quantidade

            print(
                f"  {emocao:<10} "
                f"{quantidade}/"
                f"{GRAVACOES_POR_PESSOA_EMOCAO}"
            )

    print()
    print("-" * 70)

    print(
        f"Total de gravações: {total}/120"
    )

    print("=" * 70)


# ============================================================
# MENU
# ============================================================

def menu():

    while True:

        print()
        print("=" * 60)
        print("       DATASET DE EMOÇÕES")
        print("       SALA DE REUNIÃO INTELIGENTE")
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
            print(
                "Programa encerrado."
            )

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

            pessoa = PESSOAS[
                int(opcao) - 1
            ]

            print()
            print(
                f"Participante selecionado: "
                f"{pessoa}"
            )

            print()
            print("Escolha a emoção:")

            for indice, emocao in enumerate(
                EMOCOES,
                start=1
            ):

                quantidade = contar_gravacoes(
                    pessoa,
                    emocao
                )

                print(
                    f"{indice} - "
                    f"{emocao} "
                    f"({quantidade}/"
                    f"{GRAVACOES_POR_PESSOA_EMOCAO})"
                )

            print("0 - Voltar")

            print()

            opcao_emocao = input(
                "Escolha uma emoção: "
            ).strip()

            if opcao_emocao == "0":
                continue

            if opcao_emocao in [
                "1",
                "2",
                "3",
                "4"
            ]:

                emocao = EMOCOES[
                    int(opcao_emocao) - 1
                ]

                gravar_emocao(
                    pessoa,
                    emocao
                )

            else:

                print()
                print(
                    "Opção de emoção inválida."
                )

        else:

            print()
            print(
                "Opção inválida."
            )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print(
        "SISTEMA DE GRAVAÇÃO DO DATASET DE EMOÇÕES"
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
        "- 5 participantes"
    )

    print(
        "- 4 emoções"
    )

    print(
        "- 6 gravações por pessoa/emoção"
    )

    print(
        "- 120 gravações no total"
    )

    print()

    criar_pastas()

    menu()