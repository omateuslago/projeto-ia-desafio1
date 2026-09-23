# Sala de Reunião Inteligente

Projeto acadêmico em Python que usa processamento de áudio e aprendizado de máquina para analisar uma reunião presencial. O sistema grava a conversa em blocos de quatro segundos e tenta responder a duas perguntas para cada trecho:

1. **Quem está falando?**
2. **Qual emoção vocal o trecho aparenta expressar?**

Ao final, os resultados podem ser consolidados em um relatório HTML com participantes identificados, tempo estimado de fala, distribuição das emoções e trechos classificados com baixa confiança.

> **Importante:** o projeto estima padrões da voz. Ele não comprova o sentimento real de uma pessoa e não deve ser usado para tomar decisões sensíveis sobre participantes.

## Visão geral

O projeto implementa todo o ciclo de uma solução simples de IA para áudio:

```text
Gravação dos datasets
        ↓
Extração de características acústicas
        ↓
Treinamento de dois modelos Random Forest
        ↓
Teste em um arquivo WAV
        ↓
Análise contínua de uma reunião pelo microfone
        ↓
CSV + gráficos + relatório HTML
```

Existem dois modelos independentes:

- **Modelo de pessoas:** identifica o participante pela voz.
- **Modelo de emoções:** classifica a voz como `alegre`, `neutro`, `triste` ou `irritado`.

Os participantes cadastrados atualmente no código são `joao`, `gabriel`, `mateus`, `matheus`, `erisson`, `caio` e `arthur`.

## Tecnologias utilizadas

- Python 3.10 ou superior (a compatibilidade das bibliotecas deve ser verificada no ambiente utilizado)
- PyAudio e PortAudio para captura do microfone
- Librosa para processamento de áudio
- NumPy para manipulação numérica
- scikit-learn para os modelos Random Forest e sua avaliação
- Joblib para salvar e carregar os modelos
- FastAPI e Uvicorn para a API local
- React e Vite para a interface web; Node.js e npm para executar o frontend
- Pandas para agregação dos resultados
- Matplotlib para matrizes de confusão e gráficos do relatório

## Estrutura do projeto

```text
.
├── api/                  # API FastAPI
├── frontend/             # interface React + Vite
├── tests/                # testes isolados de regressão
├── src/
│   ├── 01_gravar_pessoas.py
│   ├── 02_gravar_emocoes.py
│   ├── 03_treinar_pessoas.py
│   ├── 04_treinar_emocoes.py
│   ├── 05_testar_modelos.py
│   ├── 06_analisar_reuniao.py
│   ├── 07_gerar_relatorio.py
│   └── features.py
├── dataset_pessoas/
├── dataset_emocoes/
├── modelos/
├── resultados/
├── reunioes/
├── requirements.txt
├── README.txt
└── readme.md
```

Os scripts do pipeline de áudio ficam em `src/`, e os diretórios de dados e resultados são criados na raiz do repositório.

### Responsabilidade de cada arquivo

| Arquivo | Função |
|---|---|
| `src/01_gravar_pessoas.py` | Cria o dataset de reconhecimento de pessoas, gravando 30 áudios de quatro segundos para cada participante. |
| `src/02_gravar_emocoes.py` | Cria o dataset emocional, com seis gravações por combinação de participante e emoção. |
| `src/features.py` | Converte cada WAV em um vetor numérico com 64 características acústicas. |
| `src/03_treinar_pessoas.py` | Treina, avalia e salva o modelo que identifica as pessoas. |
| `src/04_treinar_emocoes.py` | Treina, avalia e salva o modelo que estima as emoções vocais. |
| `src/05_testar_modelos.py` | Executa os dois modelos sobre um arquivo WAV informado na linha de comando. |
| `src/06_analisar_reuniao.py` | Grava continuamente blocos de quatro segundos, classifica-os e registra os resultados em CSV. |
| `src/07_gerar_relatorio.py` | Lê o CSV da reunião e gera gráficos, tabela de baixa confiança e relatório HTML. |

## Como a IA funciona

### 1. Padronização do áudio

Cada arquivo é carregado como áudio mono e reamostrado para **16 kHz**. As gravações produzidas pelo próprio sistema usam o formato WAV, um canal e amostras de 16 bits.

### 2. Extração de características

O `features.py` resume o áudio em um vetor de **64 valores**:

- 13 médias e 13 desvios-padrão de MFCC;
- 13 médias e 13 desvios-padrão dos deltas de MFCC;
- média e desvio-padrão de RMS (energia);
- média e desvio-padrão da taxa de cruzamento por zero;
- média e desvio-padrão do centroide espectral;
- média e desvio-padrão da largura de banda espectral;
- média e desvio-padrão do rolloff espectral;
- média e desvio-padrão do pitch.

Valores ausentes ou infinitos são substituídos por zero. Caso o cálculo de pitch falhe, o pitch também assume zero.

### 3. Treinamento

Os dois classificadores usam `RandomForestClassifier` com 300 árvores, balanceamento automático das classes e semente aleatória 42. Cada dataset é dividido de forma estratificada em:

- **80% para treinamento**;
- **20% para teste**.

O treinamento imprime a acurácia e o relatório de classificação, salva o modelo em formato PKL, registra metadados em JSON e gera uma matriz de confusão em PNG.

### 4. Análise da reunião

Durante a reunião, uma thread lê o microfone e salva blocos de quatro segundos enquanto o fluxo principal classifica os WAVs. Ao encerrar, o último bloco pode ser menor. A fila guarda caminhos de arquivos, e os trechos pendentes são analisados antes da saída. Os tempos do CSV são calculados pela quantidade de amostras capturadas. Cada bloco é salvo em WAV e enviado aos dois modelos. Quando a confiança do modelo de pessoa fica abaixo de **45%**, o participante é registrado como `desconhecido`.

Para a emoção, o sistema sempre mantém a classe mais provável no CSV. O limite de 45% é usado posteriormente para destacar resultados emocionais de baixa confiança no relatório.

## Instalação

Execute os comandos a partir da raiz deste repositório. Os scripts calculam os caminhos de saída a partir da raiz do projeto.

### 1. Pré-requisitos do sistema

- Python 3.10 ou superior com suporte a ambientes virtuais;
- para a interface web, Node.js compatível com o lockfile: `^20.19.0 || >=22.12.0`, além de npm;
- microfone reconhecido pelo sistema;
- PortAudio, necessário para instalar e executar o PyAudio.

No Ubuntu ou Debian, normalmente é possível instalar os componentes do sistema com:

```bash
sudo apt update
sudo apt install python3-venv portaudio19-dev
```

### 2. Ambiente Python no Windows (PowerShell)

Em computador corporativo, utilize somente runtimes e instalações permitidos pela empresa. Os comandos abaixo são instruções para execução manual; não exigem alterar a política de execução do PowerShell nem ativar a `.venv`.

A partir da raiz do projeto, com `python` disponível:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Nos exemplos seguintes, substitua `python` por `.\.venv\Scripts\python.exe` no Windows. Não instale pacotes no Python global. O PyAudio depende de PortAudio; se a instalação não encontrar um pacote compatível, verifique o ambiente permitido antes de instalar ferramentas adicionais.

### 3. Ambiente Python no Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

O arquivo `requirements.txt` lista as bibliotecas necessárias. As versões Python não estão fixadas: a instalação não é totalmente reproduzível e precisa ser validada no ambiente utilizado. Modelos PKL devem ser treinados e carregados com versões compatíveis das bibliotecas. O frontend utiliza `package-lock.json` e deve ser instalado com `npm ci`.

## Como executar

As etapas devem ser realizadas na ordem abaixo. Os datasets não são versionados pelo Git, mas podem já existir na sua cópia local. Se estiverem completos e válidos, comece pelo treinamento; não é necessário gravar tudo novamente.

### Etapa 1 — Gravar o dataset de pessoas

```bash
python src/01_gravar_pessoas.py
```

O menu permite escolher um participante e retomar gravações incompletas. A meta configurada é de **30 arquivos por pessoa**, totalizando **210 gravações** para os sete participantes cadastrados.

Estrutura criada:

```text
dataset_pessoas/
├── arthur/
├── caio/
├── erisson/
├── gabriel/
├── joao/
├── mateus/
└── matheus/
```

### Etapa 2 — Gravar o dataset de emoções

```bash
python src/02_gravar_emocoes.py
```

Cada uma das sete pessoas grava seis exemplos para cada uma das quatro emoções. A meta total é de **168 gravações**.

O número da gravação define a frase. Assim, cada participante usa a mesma frase na rodada `001` de `alegre`, `neutro`, `triste` e `irritado`, conforme a metodologia proposta nos slides.

```text
dataset_emocoes/
├── alegre/
├── irritado/
├── neutro/
└── triste/
```

Os nomes seguem o padrão `pessoa_emocao_numero.wav`, por exemplo: `gabriel_alegre_001.wav`.

### Etapa 3 — Treinar o modelo de pessoas

```bash
python src/03_treinar_pessoas.py
```

Arquivos gerados:

```text
modelos/modelo_pessoas.pkl
modelos/metadados_pessoas.json
modelos/metadados.json
resultados/matriz_confusao_pessoas.png
```

O script exige pelo menos cinco pessoas e 20 gravações válidas por pessoa. Cada pasta de pessoa presente no dataset precisa atingir o mínimo, inclusive após descartar arquivos inválidos. O gravador cria pastas para os sete participantes configurados. Os metadados contabilizam somente os áudios processados com sucesso.

### Etapa 4 — Treinar o modelo de emoções

```bash
python src/04_treinar_emocoes.py
```

Arquivos gerados:

```text
modelos/modelo_emocoes.pkl
modelos/metadados_emocoes.json
modelos/metadados.json
resultados/matriz_confusao_emocoes.png
```

O script exige as quatro emoções e pelo menos 25 gravações válidas por emoção.

### Etapa 5 — Testar os dois modelos

```bash
python src/05_testar_modelos.py caminho/para/audio.wav
```

O comando mostra a pessoa e a emoção previstas, acompanhadas da probabilidade atribuída pelo respectivo modelo.

### Etapa 6 — Analisar uma reunião

```bash
python src/06_analisar_reuniao.py
```

Pressione `Enter` para iniciar e `Ctrl+C` para solicitar encerramento. Aguarde a análise dos blocos pendentes antes de fechar o terminal. Cada execução cria automaticamente a próxima pasta disponível (`reuniao_01`, `reuniao_02` e assim por diante), sem sobrescrever reuniões anteriores:

```text
reunioes/
└── reuniao_01/
        ├── blocos_audio/
        │   ├── bloco_001.wav
        │   ├── bloco_002.wav
        │   └── ...
        └── registros.csv
```

Cada linha do CSV possui:

| Campo | Significado |
|---|---|
| `inicio` | Instante inicial do trecho na linha do áudio capturado, em segundos. |
| `fim` | Instante final do trecho. |
| `pessoa` | Participante previsto ou `desconhecido`. |
| `conf_pessoa` | Confiança da identificação. |
| `emocao` | Emoção vocal prevista. |
| `conf_emocao` | Confiança da classificação emocional. |
| `arquivo` | Caminho do WAV correspondente. |

### Etapa 7 — Gerar o relatório

```bash
python src/07_gerar_relatorio.py
```

Sem argumento, o programa usa a reunião numerada mais recente. Para escolher uma reunião específica:

```bash
python src/07_gerar_relatorio.py reuniao_01
```

Saídas geradas em `reunioes/reuniao_01/`:

```text
reunioes/reuniao_01/
├── relatorio.txt
├── relatorio_reuniao.html
└── graficos/
        ├── tempo_fala.png
        ├── emocoes_por_pessoa.png
        ├── emocoes_geral.png
        └── baixa_confianca.csv
```

Abra `relatorio_reuniao.html` em um navegador para consultar o resultado consolidado.

## Interface web local

A interface permite consultar reuniões, iniciar/parar a captura e gerar relatórios. A gravação dos datasets e o treinamento continuam sendo feitos pelo terminal. O microfone utilizado é o do computador que executa o Python, não o do navegador.

Depois de instalar as dependências e treinar os dois modelos, abra dois terminais.

**Terminal 1 — API, a partir da raiz do projeto (Windows):**

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

No Linux com a `.venv` ativada, use `python -m uvicorn api.main:app --host 127.0.0.1 --port 8000`.

Use apenas um processo da API, sem `--reload` e sem múltiplos workers durante a captura. O controle da análise fica na memória desse processo.

**Terminal 2 — frontend, a partir da raiz:**

```powershell
cd frontend
npm ci
npm run dev
```

Abra [o painel](http://127.0.0.1:5173). O Vite encaminha `/api` e `/arquivos` para a porta 8000. A documentação da API fica em [Swagger](http://127.0.0.1:8000/docs).

1. Em **Reunião ao Vivo**, clique em **Iniciar**. Os dois modelos precisam existir.
2. Clique em **Parar** e aguarde a finalização dos trechos pendentes; esse processamento pode demorar mais que a captura.
3. Em **Relatório**, selecione uma reunião com registros e clique em **Gerar**.
4. Antes de fechar os terminais, pare a análise e aguarde o estado de análise parada.

A API rejeita CSV vazio e geração de relatório durante uma análise ativa. O projeto é destinado ao uso local: não há autenticação. O build (`npm run build`) gera arquivos estáticos, mas não substitui a API nem configura uma implantação em produção.

### Problemas comuns

- `python`, `node` ou `npm` não encontrado: o runtime não está acessível no terminal. Utilize uma instalação autorizada e reabra o terminal; não altere configurações corporativas para contornar restrições.
- Falha ao conectar ao painel: confira se API e Vite estão rodando nas portas indicadas.
- Modelo ausente: execute os scripts 03 e 04 e confira a pasta `modelos/`.
- Áudio inválido no treinamento: corrija ou regrave os arquivos indicados até atingir o mínimo de amostras válidas.
- Falha de microfone ou overflow: confira o dispositivo padrão e suas permissões. A captura sinaliza erro em vez de ocultar perda de áudio; os blocos já concluídos são preservados.
- CSV vazio: só é possível gerar relatório depois que ao menos um trecho for registrado.

### Verificações locais

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Os testes isolam bibliotecas externas e usam dados simulados. Não acessam o microfone, não iniciam servidor e não comprovam qualidade dos modelos. São necessários NumPy e Pandas para executá-los. A validação real de treinamento, build e captura deve ocorrer no ambiente autorizado com as dependências instaladas.

## Estrutura completa gerada em uso

Depois de executar todo o pipeline, a raiz do projeto tende a ficar assim:

```text
.
├── dataset_pessoas/       # gravações rotuladas por participante
├── dataset_emocoes/      # gravações rotuladas por emoção
├── modelos/              # modelos PKL e metadados JSON
├── resultados/           # matrizes de confusão do treinamento
├── reunioes/             # trechos, CSV e relatório da reunião
└── src/                  # código-fonte
```

## Limitações conhecidas

- **Não há separação de falantes simultâneos.** O uso esperado é uma pessoa falando por vez.
- **Não existe detecção de silêncio.** Blocos silenciosos ou com ruído ainda são enviados aos modelos.
- **A troca de falante pode ocorrer no meio de um bloco.** Nesse caso, uma única classe será atribuída aos quatro segundos inteiros.
- **O tempo de fala é aproximado.** Ele corresponde à soma dos blocos classificados, não ao tempo exato de fala detectada.
- **Os modelos reconhecem apenas os padrões vistos no treinamento.** Vozes, microfones e ambientes diferentes podem reduzir muito a acurácia.
- **A confiança é a probabilidade fornecida pela Random Forest**, não uma garantia de que a previsão esteja correta.
- **Cada execução cria uma reunião numerada.** Os WAVs e o CSV permanecem separados nas pastas `reuniao_01`, `reuniao_02` e seguintes.
- **A interface web é local.** O relatório gerado também pode ser aberto diretamente como HTML estático.
- **Datasets e modelos não são versionados.** Podem existir na cópia local; um clone novo precisa recebê-los ou executar gravação e treinamento.
- **A classificação pode ficar atrasada em relação à captura.** Os WAVs ficam em disco e a fila de caminhos cresce enquanto houver trechos pendentes. É necessário espaço em disco e aguardar a finalização.
- **Os testes automatizados são isolados.** Captura real, qualidade dos modelos e integração completa dependem de validação no ambiente local.

## Privacidade e uso responsável

O sistema grava e armazena voz, que é um dado pessoal e pode funcionar como dado biométrico. Antes de gravar uma reunião:

- obtenha autorização clara de todos os participantes;
- informe a finalidade da gravação e por quanto tempo os arquivos serão guardados;
- proteja os datasets, trechos WAV, modelos e relatórios contra acesso indevido;
- apague os dados quando deixarem de ser necessários;
- não trate a classificação emocional como diagnóstico psicológico ou verdade objetiva.

## Estado atual do projeto

O repositório oferece um protótipo funcional e didático do pipeline completo. O código-fonte possui sintaxe Python válida, mas a execução ponta a ponta ainda depende da instalação das bibliotecas, de um microfone compatível, da criação dos datasets e do treinamento local dos modelos.
