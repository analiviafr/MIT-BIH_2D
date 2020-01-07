from glob import glob
import os
import random
    
test = []
train = []
val = []
    
dataset = '/home/analiviafr/MITBIH_2D/MITBIH_IMG'
output_dirs = ['NOR/', 'LBBB/', 'RBBB/', 'APC/', 'PVC/', 'PAB/', 'VEB/', 'VFW/']

cont = 0
records_type = {} #por tipo
    
for type in output_dirs:
    dir = os.path.join(dataset, type, '*')
    records = glob(dir)
    records_type[type] = records
    cont += len(records)
    
for type in output_dirs:
    conj = records_type[type]
    if len(conj) == 0:
        continue #pula o 0
    random.shuffle(conj) #aleatoriza a ordem das amostras

        #60% para treinamento
    for i in range(int(len(conj)*0.6)): #série numérica para o intervalo
        temp = conj[i].split('/')
        train.append('{} {}'.format(os.path.join(temp[-2], temp[-1]), output_dirs.index(type)))
        conj[i] = None

        #20% para validação
    for i in range(int(len(conj)*0.6), int(len(conj)*0.8)):
        if conj[i] == None:
            continue
        else:
            temp = conj[i].split('/')
            val.append('{} {}'.format(os.path.join(temp[-2], temp[-1]), output_dirs.index(type)))
            conj[i] = None

        #20% para teste
    for i in range(int(len(conj) * 0.8), len(conj)):
        if conj[i] == None:
            continue
        else:
            temp = conj[i].split('/')
            test.append('{} {}'.format(os.path.join(temp[-2], temp[-1]), output_dirs.index(type)))
            conj[i] = None

    with open('Validation.txt', 'w') as val_t:
        for v in val:
            val_t.write(v+'\n')
            
    with open('Train.txt', 'w') as train_t:
        for r in train:
            train_t.write(r+'\n')

    with open('Test.txt', 'w') as test_t:
        for t in test:
            test_t.write(t+'\n')

print('Train: {}'.format(len(train)))
print('Validation: {}'.format(len(val)))
print('Test: {}'.format(len(test)))
print('Total: {}'.format(cont))