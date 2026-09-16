SALA DE REUNIAO INTELIGENTE
===========================

Projeto em Python para identificar a pessoa que esta falando,
estimar a emocao vocal de cada trecho e gerar um relatorio da
reuniao. As classificacoes emocionais sao estimativas acusticas
e nao comprovam o sentimento real dos participantes.


REQUISITOS
----------

- Python 3
- Microfone reconhecido pelo sistema
- PortAudio
- Bibliotecas listadas em requirements.txt

Preparacao do ambiente, a partir da raiz do projeto:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt


ORDEM DE EXECUCAO
-----------------

1. Gravar o dataset de pessoas:

    python src/01_gravar_pessoas.py

   O projeto possui sete participantes cadastrados. Cada pessoa
   deve possuir pelo menos 20 audios. A meta configurada e de
   30 audios por pessoa.

2. Gravar o dataset de emocoes:

    python src/02_gravar_emocoes.py

   As classes sao alegre, neutro, triste e irritado. Cada pessoa
   grava seis rodadas por emocao. O mesmo numero de rodada usa a
   mesma frase nas quatro emocoes.

3. Treinar o modelo de pessoas:

    python src/03_treinar_pessoas.py

4. Treinar o modelo de emocoes:

    python src/04_treinar_emocoes.py

5. Testar os dois modelos com um arquivo WAV:

    python src/05_testar_modelos.py caminho/para/audio.wav

6. Analisar uma reuniao:

    python src/06_analisar_reuniao.py

   Pressione ENTER para iniciar e CTRL+C para encerrar. Cada
   execucao cria uma pasta numerada em reunioes/, contendo
   blocos_audio/ e registros.csv.

7. Gerar o relatorio da reuniao mais recente:

    python src/07_gerar_relatorio.py

   Para escolher uma reuniao especifica:

    python src/07_gerar_relatorio.py reuniao_01


ARQUIVOS GERADOS
----------------

modelos/
    modelo_pessoas.pkl
    modelo_emocoes.pkl
    metadados.json
    metadados_pessoas.json
    metadados_emocoes.json

resultados/
    matriz_confusao_pessoas.png
    matriz_confusao_emocoes.png

reunioes/reuniao_XX/
    blocos_audio/
    registros.csv
    relatorio.txt
    relatorio_reuniao.html
    graficos/


ENTREGA
-------

Crie o arquivo ENTREGA_DO_GRUPO.zip com uma pasta principal
chamada sala_reuniao_ia. Essa pasta deve incluir src/,
dataset_pessoas/,
dataset_emocoes/, modelos/, resultados/, reunioes/,
README.txt e requirements.txt. Os datasets e resultados sao
ignorados pelo Git por privacidade e devem ser adicionados
manualmente ao ZIP entregue ao professor.

Grave somente participantes que autorizaram o uso da voz.
