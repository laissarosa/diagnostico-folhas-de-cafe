# Diagnóstico de Doenças em Folhas de Café com Deep Learning

Classificador de imagens baseado em **ResNet-50 (transfer learning + fine-tuning)** que identifica cinco condições em folhas de café a partir de uma foto, com **Grad-CAM++** para mostrar onde o modelo olhou. O projeto cobre o ciclo completo: recuperação e consolidação de dados de fontes distintas, deduplicação, split sem vazamento, treinamento, avaliação por classe e interpretabilidade.

| Métrica (conjunto de teste, 234 imagens) | Valor |
| --- | --- |
| Acurácia | **92,3%** |
| Macro-F1 | **91,1%** |

## Problema

Doenças e pragas do cafeeiro (ferrugem, bicho-mineiro, cercosporiose e phoma) reduzem a produtividade, e o diagnóstico depende de um especialista olhando a folha. O objetivo é apoiar esse diagnóstico com um modelo que classifique a condição da folha a partir de uma foto e indique visualmente as regiões que sustentam a decisão, para que o resultado possa ser auditado por uma pessoa.

**Classes:** saudável, ferrugem (*rust*), bicho-mineiro (*miner*), cercosporiose (*cercospora*) e mancha de phoma (*phoma*).

## Destaques técnicos

- **Engenharia de dados real:** o download oficial do BRACOL (Mendeley Data) vinha com o zip truncado; criei um script de recuperação que extraiu 1.402 arquivos e consolidei essa fonte com uma segunda (BRACOL para detecção, Roboflow) para obter a classe saudável.
- **Deduplicação entre fontes com hash perceptual (dHash):** 893 imagens da segunda fonte já existiam na primeira e foram descartadas, evitando vazamento entre treino, validação e teste.
- **Treinamento em duas etapas:** só a cabeça classificadora (3 épocas) e depois fine-tuning de `layer3` e `layer4` (12 épocas), com aumento de dados e pesos por classe para o desbalanceamento.
- **Avaliação por classe**, não só acurácia global, com identificação do ponto fraco do modelo (cercosporiose).
- **Interpretabilidade com Grad-CAM++** sobre `layer3[-1]`.
- Tudo treinado em **CPU**, sem GPU.

## Dados

### Fontes

| Fonte | Conteúdo | Observação |
| --- | --- | --- |
| BRACOL (Mendeley Data) | `leaf/dataset.csv` (1.747 linhas) e imagens em `leaf/images` | Zip truncado; 1.401 imagens recuperadas em disco. O CSV inclui folhas saudáveis (`predominant_stress` = 0) |
| BRACOL para detecção (Roboflow, `bracol-original-detect`) | 1.141 imagens, 4 classes (Cercospora, Miner, Phoma, Rust), pastas train/valid/test | Todas com alguma lesão anotada e nenhuma saudável |

As imagens das duas fontes têm o mesmo tamanho (2048x1024, JPEG).

### Pipeline de preparação (`prepare_data.py`)

1. Lê os rótulos do `dataset.csv` do Mendeley e cruza com as imagens que de fato existem em disco (nem toda linha do CSV tem imagem).
2. Incorpora as imagens da fonte Roboflow para ampliar as classes com lesão.
3. Calcula o **dHash** de cada imagem e remove duplicatas entre as fontes (rótulos coincidiram em 851 das 893 duplicatas).
4. Gera os conjuntos de treino, validação e teste em `data/splits`, sem repetição de imagens entre eles.

### Dataset final

| Classe | Imagens |
| --- | --- |
| Ferrugem (*rust*) | 530 |
| Bicho-mineiro (*miner*) | 378 |
| Phoma | 347 |
| Cercosporiose (*cercospora*) | 157 |
| Saudável (*healthy*) | 142 |
| **Total** | **1.554** (Mendeley 1.342 + Roboflow 212) |

| Split | Imagens |
| --- | --- |
| Treino | 1.086 |
| Validação | 234 |
| Teste | 234 |

## Modelo e treinamento

- **Arquitetura:** ResNet-50 pré-treinada no ImageNet, escolhida por ter sido o melhor modelo no artigo de referência dos criadores do BRACOL.
- **Etapa 1:** treino apenas da camada classificadora (3 épocas).
- **Etapa 2:** fine-tuning de `layer3` e `layer4` (12 épocas, cerca de 5 minutos por época em CPU).
- **Desbalanceamento:** pesos por classe e aumento de dados.
- **Seleção do modelo:** melhor Macro-F1 na validação (0,901; acurácia 0,915), salvo em `models/best.pt`.

## Resultados

Avaliação no conjunto de teste (234 imagens nunca vistas no treino), script `evaluate.py`:

| Classe | F1 |
| --- | --- |
| Saudável | 1,000 |
| Ferrugem | 0,949 |
| Phoma | 0,935 |
| Bicho-mineiro | 0,897 |
| Cercosporiose | 0,773 |
| **Macro-F1** | **0,911** |
| **Acurácia** | **0,923** |

A cercosporiose é a classe com pior desempenho, e também a menor entre as doenças (157 imagens). É o principal ponto de melhoria identificado.

## Interpretabilidade

`src/gradcam.py` implementa a classe `Explainer`, que gera mapas de calor **Grad-CAM++** sobre a camada `layer3[-1]`. Essa camada, em vez do Grad-CAM tradicional sobre a última camada convolucional, melhorou bastante a localização das lesões nas folhas.

## Stack

| Área | Tecnologias |
| --- | --- |
| Linguagem | Python |
| Deep Learning | PyTorch, torchvision (ResNet-50, transfer learning) |
| Interpretabilidade | Grad-CAM++ |
| Dados | Hash perceptual (dHash), splits reprodutíveis em `data/splits` |
| Ambiente | Treinamento em CPU |

## Estrutura do repositório

```
.
├── data/
│   ├── raw/
│   │   ├── bracol/            # recuperação do zip do Mendeley
│   │   └── bracol_manual/     # leaf/dataset.csv e leaf/images
│   └── splits/                # treino / validação / teste
├── models/
│   └── best.pt                # melhor modelo (Macro-F1 na validação)
├── src/
│   └── gradcam.py             # Explainer (Grad-CAM++)
├── prepare_data.py            # consolidação, deduplicação e split
└── evaluate.py                # avaliação no conjunto de teste
```

## Como reproduzir

```bash
# 1. Preparar os dados (consolida as fontes, remove duplicatas com dHash e gera os splits)
python prepare_data.py

# 2. Treinar a ResNet-50 em duas etapas
<!-- comando do script de treino -->

# 3. Avaliar no conjunto de teste
python evaluate.py
```

## Limitações

- O conjunto de teste tem 234 imagens, então as métricas por classe têm margem de incerteza, em especial a cercosporiose (a menor classe).
- Todas as imagens vêm do BRACOL; o desempenho em fotos reais de campo, com outras câmeras, iluminação e fundos, ainda não foi medido.
- A classe saudável (142 imagens) é pequena e veio apenas da fonte do Mendeley.

## Próximos trabalhos

- [ ] **API de inferência em Python** com a ResNet-50 e o Grad-CAM++ no servidor.
- [ ] **Checagem de nitidez** da foto antes da classificação.
- [ ] **Limiar de recusa:** quando a confiança for baixa ou a foto fora do padrão, encaminhar o caso para um agrônomo; medir a taxa de recusa e se as recusas coincidem com os erros do modelo.
- [ ] **Aplicativo Android (Apache Cordova)** como cliente: câmera, envio da foto e exibição do diagnóstico com a confiança.
- [ ] **Validação com fotos reais fora do dataset** para medir a generalização.

## Referências

- BRACOL — conjunto de imagens de folhas de café (Mendeley Data).
- BRACOL para detecção, versão original (Roboflow, `bracol-original-detect`).