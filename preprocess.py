from glob import glob
import wfdb
import cv2
import os
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_records():
    #serão utilizados os arquivos .atr
    records = glob('./mit_arrythmia_dat/*.atr')

    #retirando as extensões
    records = [path[:-4] for path in records]
    records.sort()
    return records

def segmentation(records, type, output_dir=''):
    #T(Qpeak(n) - 96) ≤ T(n) ≤ T(Qpeak(n) + 96)
    os.makedirs(output_dir, exist_ok=True)
    results = []
    kernel = np.ones((4, 4), np.uint8)
    cont = 1

    mean_values = []
    
    for e in tqdm(records):
        signals, fields = wfdb.rdsamp(e, channels=[0]) #retorna os sinais físicos
        mean_values.append(np.mean(signals))
        
    mean_values = np.mean(np.array(mean_values))
    std_values = 0
    cont = 0
    
    for e in tqdm(records):
        signals, fields = wfdb.rdsamp(e, channels=[0])
        cont += len(signals)
        for i in signals:
            std_values += (i[0] - mean_values)**2
            
    std_values = np.sqrt(std_values/cont)
    
    min_value = mean_values - 3*std_values
    max_value = mean_values + 3*std_values

    for e in tqdm(records):
        signals, fields = wfdb.rdsamp(e, channels = [0])

        #extração do tipo da batida
        ann = wfdb.rdann(e, 'atr') #lê uma anotação e retorna um retorne um objeto Annotation
        good = [type]
        ids = np.in1d(ann.symbol, good) #matriz booleana
        imp_beats = ann.sample[ids]
        beats = (ann.sample)
        for i in tqdm(imp_beats):
            beats = list(beats)
            j = beats.index(i)  #remove indice
            if(j!=0 and j!=(len(beats)-1)): #exclui primeiro e ultimo
                #considerando 96 pontos anteriores e posteriores, para que haja 192 pontos de amostragem
                data = (signals[beats[j]-96:beats[j]+96, 0])

                results.append(data)

                plt.axis([0, 192, min_value, max_value])
                plt.plot(data, linewidth=0.5)
                plt.xticks([]), plt.yticks([])
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)

                filename = output_dir+'fig_{}'.format(cont)+'.png'
                plt.savefig(filename)
                plt.cla()
                plt.clf()
                plt.close()
                
                #escala cinza
                im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                im_gray = cv2.erode(im_gray, kernel, iterations=1)
                im_gray = cv2.resize(im_gray, (192, 128), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(filename, im_gray)
                print('imagem {}'.format(filename))
                cont += 1

    return results

if __name__ == "__main__":
    records = get_records()
    
    #SUPERCLASSE N
    labelsN = ['N', 'L', 'R', 'e', 'j']
    outputN = ['N', 'L', 'R', 'e', 'j']

    for type, output_dir in zip(labelsN, outputN):
        seg = segmentation(records, type,'./MITBIH_2D/N/')

    #SUPERCLASSE S
    labelsS = ['A', 'a', 'J', 'S']
    outputS = ['A', 'a', 'J', 'S']

    for type, output_dir in zip(labelsS, outputS):
        seg = segmentation(records, type,'./MITBIH_2D/S/')

    #SUPERCLASSE V
    labelsV = ['V', 'E']
    outputV = ['V', 'E']

    for type, output_dir in zip(labelsV, outputV):
        seg = segmentation(records, type,'./MITBIH_2D/V/')

    #SUPERCLASSE F
    labelsF = ['F']
    outputF = ['F']

    for type, output_dir in zip(labelsF, outputF):
        seg = segmentation(records, type,'./MITBIH_2D/F/')

    #SUPERCLASSE Q
    labelsQ = ['/', 'f', 'Q']
    outputQ = ['p', 'f', 'Q']

    for type, output_dir in zip(labelsQ, outputQ):
        seg = segmentation(records, type,'./MITBIH_2D/Q/')
