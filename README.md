# Arrhythmia Classifier

Implementação do trabalho de iniciação científica **"Classificação de arritmias cardíacas por meio de redes neurais convolucionais unidimensionais e bidimensionais"**.

## DATASETS 
Para a realização dos experimentos unidimensionais foi utilizado o dataset [ECG Heartbeat Classification](www.kaggle.com/shayanfazeli/heartbeat). A base de dados é composta por 104446 registros de ECG referentes às cinco superclasses de arritmias cardíacas recomendadas pelo padrão ANSI/AAMI/ISO EC57. As superclasses recomendadas por esse padrão são resultado do agrupamento de 15 classes do dataset [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) e estão apresentadas na tabela abaixo.

Superclasses | Descrição | Classes agrupadas
--------|----------|------
N | Batimento Normal | N, L, R, e, j
S | Batimento Ectópico Supraventricular | A, a, J, S
V | Batimento Ectópico Ventricular |  V, E
F | Batimento de Fusão | F
Q | Batimento Desconhecido | /, f, Q

Já para a aquisição das imagens bidimensionais, foi realizado o pré-processamento do dataset [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/).

## PREPROCESS 
Foram selecionadas para trabalho as 5 superclasses recomendadas pela **AAMI**, sendo essas: Batimento Normal (N); Batimento Ectópico Supraventricular (S); Batimento Ectópico Ventricular (V); Batimento de Fusão (F); Batimento Desconhecido (Q).

Classes | Imagens geradas
--------|----------------
N | 90.589
S | 2.779
V | 7.236
F | 803
Q | 8.039

Para gerar as imagens é necessário executar o arquivo ***preprocess.py***. Desse modo, serão geradas 109446 imagens com resolução de 256x256, conforme ilustrado pela figura abaixo.
<p align="center">
  <img src="/docs/imagesECG2.png" >
</p>

## SPLIT DATASET
O script ***split_dataset.py*** seleciona 80% das imagens geradas para o conjunto de treinamento e 20% para o conjunto de teste.

## TRAINING
Foi desenvolvido uma arquitetura de CNN bidimensional, representada pela figura abaixo, que pode ser treinada por meio da execução do script **Training_2D.py**. Também foi implementada uma representação unidimensional dessa arquitetura, a qual pode ser treinada a partir de **Training_1D**.

<p align="center">
  <img src="/docs/proposed_model.png" >
</p>

## VALIDATION
Os algoritmos foram validados segundo a recomendação da AAMI, que afirma que as métricas de acurácia, sensibilidade, precisão e especificidade são as mais indicadas para a avaliação de sistemas automáticos para a classificação de arritmias cardíacas. Também utilizou-se a métrica *F1-score* e a estatística de Youden para melhor comparação entre os resultados alcançados. Além disso, a fim de melhor observar o comportamento do modelo proposto na classificação por classes, foi gerada a matriz de confusão.

Em sequência, com o intuito de verificar o desempenho da CNN 2D proposta, seus resultados foram comparados com arquiteturas disponíveis na literatura, tais como AlexNet, EfficientNet50, InceptionV3, VGG16 e VGG19. Para tanto, considerou-se os mesmos conjuntos de treinamento e de teste utilizados no modelo proposto, de modo a realizar comparações justas entre as arquiteturas. Por fim, os resultados alcançados foram equiparados ao desempenho de uma CNN 1D.

## RESULTS
A partir das imagens de ECG, foram realizados experimentos com as arquiteturas da literatura e com o modelo proposto. Todos os experimentos foram executados cinco vezes, a fim de obter a média e o desvio padrão dos resultados. A tabela abaixo apresenta as arquiteturas que obtiveram os melhores desempenhos, bem como os respectivos valores de *learning rate*.

*Learning Rate* | Arquitetura | Acurácia (%)| Precisão (%)| Sensibilidade (%)| Especificidade (%)| *F1-Score* (%)| Estatística de Youden
----------------|-------------|-----------|----------|---------------|-----------------|-------------|----------------------
0,001 | proposed_model | 82,86±0,16 | 82,86±0,16  | 82,86±0,16 | 95,73±0,08 | 82,86±0,16 | 0,7858818617
0,005 | EfficientNet | 82,79±0,00 | 82,79±0,00 | 82,79±0,00 | 95,70±0,00 | 82,79±0,00 | 0,7849155637
0,005 | VGG16 | 82,79±0,00 | 82,79±0,00 | 82,79±0,00 | 95,70±0,00 | 82,79±0,00 | 0,7849155637
0,01 | VGG16 | 82,79±0,05 | 82,79±0,05 | 82,79±0,05 | 95,70±0,05 | 82,79±0,05 | 0,7849155999
0,05 | EfficientNet | 82,79±0,00 | 82,79±0,00 | 82,79±0,00 | 95,70±0,00 | 82,79±0,00 | 0,7849155637
0,05 | InceptionV3 | 82,79±0,01 | 82,79±0,01 | 82,79±0,01 | 95,70±0,01 | 82,79±0,01 | 0,7849155637
0,05 | ResNet50 | 82,79±0,00 | 82,79±0,00 | 82,79±0,00 | 95,70±0,00 | 82,79±0,00 | 0,7849155637

O modelo proposto também se destacou em todas as métricas, o que indica que a classificação ocorreu de forma satisfatória. Além disso, a fim de verificar o desempenho da proposta na classificação por classes, foi gerada a matriz de confusão, ilustrada pela figura abaixo, de modo que a diagonal principal indica os verdadeiros positivos, ou seja, os acertos do classificador.

<p align="center">
  <img src="/docs/confusion_matrix_2D.PNG" >
</p>

Por fim, o modelo proposto foi adaptado para uma CNN 1D, a qual recebeu como entrada sinais de ECG oriundo da base de dados [ECG Heartbeat Classification](www.kaggle.com/shayanfazeli/heartbeat). Os resultados alcançados são expressos na tabela abaixo.

*Learning Rate* | Acurácia (%)| Precisão (%)| Sensibilidade (%)| Especificidade (%)| *F1-Score* (%)
----------------|-------------|-------------|------------------|-------------------|----------------
0,001 | 98,55±0,04 | 98,55±0,04  | 98,55±0,04 | 99,69±0,04 | 98,55±0,04 
0,005 | 97,94±0,03 | 97,94±0,03 | 97,94±0,03 | 99,49±0,03 | 97,94±0,03
0,01 | 96,80±0,39 | 96,80±0,39 | 96,80±0,39 | 98,93±0,39 | 96,80±0,39
0,05 | 96,36±0,62 | 96,36±0,62 | 96,36±0,62 | 99,09±0,62 | 96,36±0,62

Além disso, o melhor desempenho foi dado com 0,001 de *learning rate*, atingindo um índice de 0,9824299896 pela estatística de Youden, o que denota a existência de poucos falsos positivos e falsos negativos. Sua eficiência na classificação também é comprovada pela matriz de confusão, apresentada pela abaixo, uma vez que houve poucos erros na previsão dos rótulos.

<p align="center">
  <img src="/docs/confusion_matrix_1D.PNG" >
</p>

### Ana Lívia Franco
<a target="_blank" href="https://www.linkedin.com/in/analiviafr"><img src="https://img.shields.io/badge/-LinkedIn-0077B5?style=for-the-badge&logo=Linkedin&logoColor=white"></img></a> <a target="_blank" href="mailto:analiviafr@gmail.com"><img src="https://img.shields.io/badge/-Gmail-D14836?style=for-the-badge&logo=Gmail&logoColor=white"></img></a>
