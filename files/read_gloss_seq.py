# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 10:49:41 2019

@author: papastrat

"""
files=['phoenix2014T.train.gloss','phoenix2014T.dev.gloss','phoenix2014T.test.gloss']
       
        
a='MORGEN IX BISSCHEN WOLKE NORD DEUTSCH WOLKE SONNE MOEGLICH NORDOSTRAUM BISSCHEN SONNE SONNE BISSCHEN TEIL REGEN NORDWESTRAUM REGEN REGEN MOEGLICH AUCH BISSCHEN MEHR STARK REGEN REGEN WOLKE STARK BEWOELKT __HOLD__ WOLKE WOLKE WOLKE WOLKE'
import numpy as np
classes=[]
with open('phoenix2014T.vocab.gloss','r') as f:
    lines=f.readlines()
    for item in lines:
        classes.append(item.split('\n')[0])
print(len(classes))
b='DIESE SOMMER DIESE AKTUELL negalp-STIMMT SOMMER __PU__ ABER IM-VERLAUF TAGSUEBER SUEDWESTRAUM cl-KOMMEN cl-KOMMEN cl-KOMMEN MEHR WARM NAECHSTE SONNTAG DEUTSCH LAND UNGEFAEHR ZWANZIG BIS NEUN ZWANZIG GRAD NORD SPUEREN KUEHL ALS SUED MEHR WARM'

maxlen=0
maxseq=''
data=[]
for i in range(3):
    data=[]
    with open(files[i],'r') as f:
        for line in f:
            data.append(line.strip('\n'))
        for item in data:
            ## split seq in space
            item.strip('\n')
            temp=item.split(' ')
            if len(temp)>maxlen:
                maxlen=len(temp)
                maxseq=item
                index=data.index(item)
        
                
        
        print('LENGTH of file {} = {} max_len of gloss sequence = {} \n  max seq {} index {} '.format(files[i],len(data),maxlen,maxseq,index))
    data=[]
#with 
#print(data)
print('Classes {} '.format(classes))
print("index of glosses")


print("Replace gloss with ints")
for item in a.split(' '):
    print(item)
    print(classes.index(item))     

intdata=[]

oov=0
print("Replace gloss sdsdfsds with ints")
'''
for i in range(1):
    data=[]
    intdata=[]
    with open(files[1],'r') as f:
        for line in f:
            data.append(line.strip('\n'))
        for item in data:
            glosses=item.split(' ')
            intseq=''
            #print("seq = {} {}  ".format(data.index(item),item))
            for item in glosses:
                if(item not in classes):
                    print(" !!!!!!!!!! OUT OF VOCABULARY  !!!!!!  {} ".format(item))
                    oov+=1
                else :
                    intseq+=str(classes.index(item))+','
            intdata.append(intseq[0:-2])
    print(" converted {} file  to int , length= {} ".format(files[1],len(intdata)))
    with open(files[1]+'_indexes.txt','w') as wf:
        for item in intdata:
            wf.write("%s \n" % item)
                
'''                
for i in range(1):
    data=[]
    intdata=[]
    with open(files[2],'r') as f:
        for line in f:
            data.append(line.strip('\n'))
        for item in data:
            glosses=item.split(' ')
            intseq=[]
            #print("seq = {} {}  ".format(data.index(item),item))
            for item in glosses:
                if(item not in classes):
                    print(" !!!!!!!!!! OUT OF VOCABULARY  !!!!!!  {} ".format(item))
                    oov+=1
                else :
                    intseq.append(classes.index(item))
            intdata.append(intseq)
    print(" converted {} file  to int , length= {} ".format(files[2],len(intdata)))
    with open(files[2]+'_indexes.txt','w') as wf:
        for item in intdata:
            wf.write("%s \n" % item)
a=np.array(intdata)
import csv 
with open(files[2]+'_indexes.csv','w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for item in intdata:
        writer.writerow(item)
    
                
                
        
    
        
 