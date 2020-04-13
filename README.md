# ECG_Classification_CNN2D

Implementação do trabalho de conclusão de curso **"Classificação de arritmias cardíacas por meio de redes neurais convolucionais bidimensionais"**. O presente trabalho está em processo de desenvolvimento.

## DATASET
O dataset utilizado foi o [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)

## PREPROCESS 
Para gerar as imagens é necessário executar o script ***preprocess.py***. Desse modo, serão geradas 109.445 imagens com resolução de 196x128.

Classes | Imagens geradas
--------|----------------
N | 90.589
S | 2.779
V | 7.236
F | 803
Q | 8.038

## SPLIT DATASET
O script ***split_dataset.py*** divide o dataset de imagens geradas em treinamento, teste e validação.
