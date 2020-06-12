# arrhythmia_classifier_CNN2D

Implementação do trabalho de conclusão de curso **"Classificação de arritmias cardíacas por meio de redes neurais convolucionais bidimensionais"**. O presente trabalho está em processo de desenvolvimento.

## DATASET
O dataset utilizado no presente trabalho foi o [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/). 

## PREPROCESS 
São selecionadas para trabalho as 5 superclasses recomendadas pela **AAMI**, sendo essas: Batimento Normal (N); Batimento Ectópico Supraventricular (S); Batimento Ectópico Ventricular (V); Batimento de Fusão (F); Batimento Desconhecido (Q).

Para gerar as imagens é necessário executar o arquivo ***preprocess.py***. Desse modo, serão geradas 109.445 imagens com resolução de 256x256.

Classes | Imagens geradas
--------|----------------
N | 90.589
S | 2.779
V | 7.236
F | 803
Q | 8.039

O dataset de imagens criado também está disponível no [Kaggle](https://www.kaggle.com/analiviafr/ecg-images).

## SPLIT DATASET
O script ***split_dataset.py*** divide o dataset de imagens geradas em treinamento, teste e validação.
