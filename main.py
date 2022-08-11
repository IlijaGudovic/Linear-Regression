from re import X
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

#Učitavanje skupa
data_frame = pd.read_csv("/content/net_national_incom.csv")
print(data_frame.shape)

#Funkcija koja vraća okvir podataka prema nazivu određene zemlje
def Select(name):
  for index, row in data_frame.iterrows(): # Prolazimo korz sve redove glavnog skupa
    #print(index, row)
    if  row['Country Name'] == name: 
      dohodak = row.iloc[1:].values

  #Inicijalizacija prazanog niza
  godina = []
  i = 1970
  while i < 2022:
    godina.append(i) # Dodavanje vrednosti od 1970 do 2022
    i += 1 

  #Inicijalizacija praznoog skupa podataka
  new_df = pd.DataFrame() 
  new_df['Godina'] = godina
  new_df['Dohodak'] = dohodak
  new_df = new_df.dropna()

  new_df = new_df.reset_index(drop = True)

  #Cuvanje novog skupa
  new_df.to_csv(name.lower() + '_net_national_incom.csv' , index = False)

  return(new_df)

df = Select('Serbia')
print(df.shape)
print(df)

x= df.iloc[:,0].values # skup features-a
y= df.iloc[:,1].values # skup target-a

#Iscrtavanje podataka 
plt.scatter(x,y)
plt.xlabel('Godine')
plt.ylabel('Dohodak')
plt.title('Podaci')
plt.show()

#Ponovono učitavamo skup 
c = pd.read_csv("/content/serbia_net_national_incom.csv")
print(c.corr()) # koeficijent korelacije
print('\n')

#Podela glavnog skupa na obucavajuci i testirajuci
X1_obucavajuci, X1_testirajuci, Y1_obucavajuci, Y1_testirajuci = train_test_split(x, y, test_size=0.20, random_state=1)
reg = LinearRegression() #

print(X1_obucavajuci.shape,X1_testirajuci.shape,Y1_obucavajuci.shape,Y1_testirajuci.shape) #Provera dimenzija

#Preoblikovanje skupova u dvodimenzionalne skupove
X1_obucavajuci=np.reshape(X1_obucavajuci,(-1,1))
Y1_obucavajuci=np.reshape(Y1_obucavajuci,(-1,1))
X1_testirajuci=np.reshape(X1_testirajuci,(-1,1))
Y1_testirajuci=np.reshape(Y1_testirajuci,(-1,1))

print(X1_obucavajuci.shape,X1_testirajuci.shape,Y1_obucavajuci.shape,Y1_testirajuci.shape ) #Ponovna provera dimenzija

#Obucavanje modela
reg.fit(X1_obucavajuci, Y1_obucavajuci)
print(reg.coef_, reg.intercept_) # y = reg.coef * x  reg.intercept

#Iscrtavanje regresione prave nad obucavajucim podacima 
plt.scatter(X1_obucavajuci,Y1_obucavajuci , color = 'orangered')
plt.plot(X1_obucavajuci, reg.predict(X1_obucavajuci),color='red')
plt.xlabel('Godine')
plt.ylabel('Dohodak')
plt.title('Regresija nad obucavajucem skupu')
plt.show()

#Iscrtavanje regresione prave nad testirajucim podacima 
plt.scatter(X1_testirajuci,Y1_testirajuci,color='orangered')
plt.plot(X1_testirajuci, reg.predict(X1_testirajuci),linewidth=2, color='red')
plt.xlabel('Godine')
plt.ylabel('Dohodak')
plt.title('regresija nad testirajucem skupu')
plt.show()

#Iscrtavanje grafikona neto nacionalnog dohodka po godinama
plt.plot(x,y, marker='.', mfc='black')
plt.xlabel('Godine')
plt.ylabel('Dohodak')
plt.title('neto nacionalni dohodak po glavi stanovnika')
plt.show()

Y_predikcija = reg.predict(X1_testirajuci) # Predikcija nad testirajucim podacima
print(metrics.mean_squared_error(Y1_testirajuci/max(Y1_testirajuci),Y_predikcija/max(Y_predikcija))) # Proracun error-a
print(metrics.r2_score(Y1_testirajuci/max(Y1_testirajuci),Y_predikcija/max(Y_predikcija))) # Proracun koeficijenta poklapanja

#Funkcija ispisuje procenu godisnjeg rasta neto nacionalnog dohodka po glavi stanovnika
def Calc_Rate():
  lastY = y[len(y) - 1] # Uzima poslednji segment skupa
  growth = reg.coef_
  rate  = growth / lastY * 100
  print('Prema obradjenim podacima godisnji rast neto nacionalnog dohodaka je %.0fEUR %.1f%% \n' % (growth, rate))

Calc_Rate()

#Redukcija skupa
x= df.iloc[8:,0].values # skup features-a
y= df.iloc[8:,1].values # skup target-a

#Ponovno treniranje modela
reg = LinearRegression()
reg.fit(np.reshape(x,(-1,1)), np.reshape(y,(-1,1)))

#Iscrtavanje novog grafikona neto nacionalnog dohodka sa regresijom
plt.plot(x,y, marker='.', mfc='black')
plt.plot(x, reg.predict(np.reshape(x,(-1,1))),color='red', alpha=0.4)
plt.xlabel('Godine')
plt.ylabel('Dohodak')
plt.title('neto nacionalni dohodak po glavi stanovnika')
plt.show()

#Predikcija neto nacionalnog dohodka u narednih 8 godina
godina = []
i = 2022
while i <= 2030:
  godina.append(i) # Dodavanje vrednosti u intervalu od 2022 do 2030
  i += 1 

#Inicijalizacija praznoog skupa podataka
predction_df = pd.DataFrame() 
predction_df['Godina'] = godina

godina = np.reshape(godina,(-1,1)) # Preoblikovanje u 2D skup
dohodak = reg.predict(godina) # Predikcija dohodka za godine[]
predction_df['Dohodak'] = dohodak # Definisanje kolone 'Dohodak'

plata_EUR = dohodak / 12
predction_df['Mesecni Dohodak EUR'] = plata_EUR

plata_DIN = plata_EUR * 117.55
predction_df['Mesecni Dohodak DIN'] = plata_DIN

#Pretvaranje okvira podataka u tekstualnu datoteku 
predction_df.to_csv('predction_net_national_incom.csv' , index = False)

print(predction_df)

#Kalkulacija rasta nad novim podacima
Calc_Rate()

#Funkcija kao argument uzima ime drzave za poredjenje
def Compare(target):
  country_target_df = Select(target)
  target_dohodak = country_target_df.iloc[-1 , -1] # Uzima poslednji segment skupa 

  god_pred = (np.reshape(target_dohodak,(-1,1))  - reg.intercept_) / reg.coef_

  print(target + ' net nacional incom is %.0f EUR in 2020 year' % target_dohodak)
  print('Srbija dostize dohodak od %.0f EUR %.0f. godine \n' % (target_dohodak, god_pred))


Compare("Madagascar")
Compare("Japan")
Compare("Brazil")
Compare("Turkey")
Compare("Costa Rica")
Compare("Kazakhstan")

