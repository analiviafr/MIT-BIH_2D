# Arrhythmia Classifier

Implementação do trabalho de conclusão de curso **"Classificação de arritmias cardíacas por meio de redes neurais convolucionais unidimensionais e bidimensionais"**. O presente trabalho está em processo de desenvolvimento.

## DATASETS 
Para a realização dos experimentos unidimensionais foi utilizado o dataset [ECG Heartbeat Classification](www.kaggle.com/shayanfazeli/heartbeat). A base de dados é composta por 104446 registros de ECG referentes às cinco superclasses de arritmias cardíacas recomendadas pelo padrão ANSI/AAMI/ISO EC57. As superclasses recomendadas por esse padrão são resultado do agrupamento de 15 classes do dataset [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) e estão apresentadas na tabela abaixo.

Superclasses | Descrição | Classes agrupadas
--------|----------|------
N | Batimento Normal | N, L, R, e, j
S | Batimento Ectópico Supraventricular | A, a, J, S
V | Batimento Ectópico Ventricular |  V, E
F | Batimento de Fusão | F
Q | Batimento Desconhecido | /, f, Q

Já para a aquisição das imagens bidimensionais, os registros de ECG oriudos do datase [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) foram pré-processados.

## PREPROCESS 
Foram selecionadas para trabalho as 5 superclasses recomendadas pela **AAMI**, sendo essas: Batimento Normal (N); Batimento Ectópico Supraventricular (S); Batimento Ectópico Ventricular (V); Batimento de Fusão (F); Batimento Desconhecido (Q).

Para gerar as imagens é necessário executar o arquivo ***preprocess.py***. Desse modo, serão geradas 109446 imagens com resolução de 256x256.

Classes | Imagens geradas
--------|----------------
N | 90.589
S | 2.779
V | 7.236
F | 803
Q | 8.039

## SPLIT DATASET
O script ***split_dataset.py*** seleciona 80% das imagens geradas para o conjunto de treinamento e 20% para o conjunto de teste.
