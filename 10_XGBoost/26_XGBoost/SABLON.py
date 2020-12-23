#################################################################
####################### Data Preprocesing #######################

"""**Kutuphaneler**"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""**Veri yukleme**"""

veriler = pd.read_csv('/content/eksikveriler.csv')

"""**DATA FRAME (SLICE) DILIMLEME**"""

X = veriler.iloc[:, [1]]
Y = veriler.iloc[:, [2]]
# NUMPY ARRAY DIZI DONUSUMU
x = X.values
y = Y.values

"""**Eksik verileri duzenleme**"""

from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

sayisalVeriler = veriler.iloc[:, 1:4].values

sayisalVeriler = imp.fit_transform(sayisalVeriler)

sayisalVeriler

ulke = veriler.iloc[:,0:1].values

ulke

"""**Kategorikden sayisal veriye gecis**"""

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:, 0] = le.fit_transform(veriler.iloc[:, 0:1])

ulke

ohe = preprocessing.OneHotEncoder()

ulke = ohe.fit_transform(ulke).toarray()

ulke

"""**Numpy dizisi DataFrame Donusumu**"""

sonuc1 = pd.DataFrame(data=ulke, index=range(len(ulke)), columns=['fr', 'tr', 'us'])

sonuc1

sonuc2 = pd.DataFrame(data=sayisalVeriler, index=range(len(sayisalVeriler)), columns= ['boy', 'kilo', 'yas'])

sonuc2


"""**DataFramelerin birlestirlime islemi**"""

s1 = pd.concat([sonuc1, sonuc2], axis=1) # satir satira birlestirmek icin yani yatay boyutta birlestirmek icin axis = 1

s2 = pd.concat([s1, sonuc3], axis=1)

s2

cinsiyet = veriler.iloc[:,-1].values

sonuc3 = pd.DataFrame(data=cinsiyet, index=range(len(cinsiyet)), columns=['cinsiyet'])

sonuc3

"""**Verilerin test ve egitim icin bolunmesi**"""

from sklearn.model_selection import train_test_split

# verinin yuzde 66 si antrenman icin kullanilsin kalan yuzde 33'u test edilsin diye ayrdik
# random_state rastsal ayirma icin kullaniliyor ayni degeri alan her kod ayni ayrimi yapar
x_train, x_test, y_train, y_test = train_test_split(s1, sonuc3, test_size = 0.33, random_state = 0)

"""**Verilerin normalize edilme islemi**"""

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

# burasi bir tur normalizasyon islemi anlatiyor (x-mean)/standar_deviation denklemini kullanir
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

####################### Data Preprocesing #######################
#################################################################

#################################################################
##################### Prediction Algorithms #####################
"""**LINEER REGRESSION**"""

X = veriler.iloc[:, 2:3]
Y = veriler.iloc[:, 5:]
# NUMPY ARRAY DIZI DONUSUMU
x = X.values
y = Y.values

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x, y)

import statsmodels.api as sm
model1 = sm.OLS(lin_reg.predict(x),x).fit()
model1.summary()

"""**POLYNOMIAL REGRESSION**"""

X = veriler.iloc[:, 2:3]
Y = veriler.iloc[:, 5:]
# NUMPY ARRAY DIZI DONUSUMU
x = X.values
y = Y.values

from sklearn.preprocessing import PolynomialFeatures

# verileri derecesine gore olusturduk
poly_reg = PolynomialFeatures(degree = 2)

x_poly = poly_reg.fit_transform(x)

# polynomial predict
lin_regP = LinearRegression()
lin_regP.fit(x_poly, Y) # x_poly derecesine gore yeni X leri tanimlayan dizi

model2 = sm.OLS(lin_regP.predict(x_poly),x).fit()
model2.summary()

"""**SUPPORT VECTOR MACHINE**"""

X = veriler.iloc[:, 2:5]
Y = veriler.iloc[:, 5:]
# NUMPY ARRAY DIZI DONUSUMU
x = X.values
y = Y.values

from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR

svrReg = SVR(kernel = 'rbf')
svrReg.fit(x_olcekli, y_olcekli)

model3 = sm.OLS(svrReg.predict(x_olcekli),x_olcekli).fit()
model3.summary()

"""**DECISION TREE**"""

X = veriler.iloc[:, 2:3]
Y = veriler.iloc[:, 5:]
# NUMPY ARRAY DIZI DONUSUMU
x = X.values
y = Y.values

from sklearn.tree import DecisionTreeRegressor

dT = DecisionTreeRegressor(random_state = 0)
dT.fit(x,y)

model4 = sm.OLS(dT.predict(x),x).fit()
model4.summary()

"""**RANDOM FOREST**"""

X = veriler.iloc[:, 2:3]
Y = veriler.iloc[:, 5:]
# NUMPY ARRAY DIZI DONUSUMU
x = X.values
y = Y.values

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(random_state = 0, n_estimators=10)
rfr.fit(x, y)

model5 = sm.OLS(rfr.predict(x),x).fit()
model5.summary()

print(f'LINEER REGRESSION       R-squared:   {model1.rsquared}')
print(f'POLYNOMIAL REGRESSION   R-squared:   {model2.rsquared}')
print(f'SUPPORT VECTOR MACHINE  R-squared:   {model3.rsquared}')
print(f'DECISION TREE           R-squared:   {model4.rsquared}')
print(f'RANDOM FOREST           R-squared:   {model5.rsquared}')

##################### Prediction Algorithms #####################
#################################################################

#################################################################
############## Evaluating of Method of Predictions ##############

"""**R2 SCORE**"""
from sklearn.metrics import r2_score
r2_score(y, y_pred)

"""**P VALUE**"""
import statsmodels.api as sm
model1 = sm.OLS(lin_reg.predict(x),x).fit()
model1.summary()

############## Evaluating of Method of Predictions ##############
#################################################################
#-------------------------------------------------------------------------------

#################################################################
#################### Clasification Algorithms ###################

"""**Logistic Regression**"""
from sklearn.linear_model import LogisticRegression

logReg = LogisticRegression(random_state=0)

logReg.fit(X_train, y_train)

y_pred = logReg.predict(X_test)

"""**K Nearest Neighborhood Classification**"""
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

"""**SVM Classification**"""
from sklearn.svm import SVC

svc = SVC(kernel = 'rbf')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

"""**Naive Bayes Classification**"""

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

mnb = MultinomialNB() # negatif data geldigi icin normalize edilmemis degerler ile kullandim
mnb.fit(x_train, y_train)

bnb = BernoulliNB()
bnb.fit(X_train, y_train)

y_predg = gnb.predict(X_test)
y_predm = mnb.predict(x_test)
y_predb = bnb.predict(X_test)

"""**Decision Tree Clasification**"""

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

"""**Random Forest Classification**"""
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
#################### Clasification Algorithms ###################
#################################################################

#################################################################
############ Evaluating of Method of Clasifications #############

"""**Confision Matrix**"""
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm

"""**ROC CURVE**"""
from sklearn.metrics import roc_curve
y_prob = rfc.predict_proba(X_test)
fpr, tpr, thold = roc_curve(y_test, y_prob[:, 0], pos_label = 'e' )

############ Evaluating of Method of Clasifications #############
#################################################################

#################################################################
#################### Clustring Algorithms ###################

"""**K-MEANS**"""

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, init = 'k-means++')
kmeans.fit(X)
kmeans.cluster_centers_ # sectigi noktalar

##WCSS hesabi
sonuclar = []
for i in range(1, 10):
  kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 245)
  kmeans.fit(X)
  sonuclar.append(kmeans.inertia_)


"""**Hierarchical Clustering**"""

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
y_pred = ac.fit_predict(x)
print(y_pred)

##Dendrogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))

#################### Clustring Algorithms ###################
#################################################################

#################################################################
#################### Assocication Rule Mining ###################

""" Apriori Algorithm """

# i used a special folder from github about apriori algorithm
from apyori import apriori

kural = apriori(dataM, min_support = 0.01, min_confidence = 0.2, min_lift = 3, max_length = 2)

#################### Assocication Rule Mining ###################
#################################################################


#################################################################
####################### Reinforced Learning #####################

"""**Random Selection**"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('/content/Ads_CTR_Optimisation.csv')

import random
random.seed(0)

N = 10000   # how much random numbers you will create
d = 10    # there are ten ads
toplam = 0
secilenler = []

for n in range(0, N):
  add = random.randrange(d)
  secilenler.append(add)
  odul = veriler.values[n, add] # will choice a column for each row
  toplam = toplam + odul

toplam

plt.hist(secilenler)
plt.show

"""**Upper Confidence Bound**"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('/content/Ads_CTR_Optimisation.csv')

N = 10000   # how much random numbers you will create
d = 10      # there are ten ads

import math
oduller = [0] * d
tiklamalar = [0] * d
toplam = 0
secilenler = []
for n in range(1, N):
  add = 0
  max_ucb = 0
  for i in range(0, d):
    if tiklamalar[i] > 0:
      ortalama = oduller[i] / tiklamalar[i]
      delta = math.sqrt(3/2 * math.log(n)/tiklamalar[i])
      ucb = ortalama + delta
    else:
      ucb = N*10 
    if max_ucb < ucb:
      max_ucb = ucb
      add = i     
  secilenler.append(add)
  tiklamalar[add] = tiklamalar[add] + 1
  odul = veriler.values[n, add]
  oduller[add] = oduller[add] + odul
  toplam = toplam + odul

toplam

plt.hist(secilenler)
plt.show()

"""**Thompson Sampling**"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('/content/Ads_CTR_Optimisation.csv')

N = 10000   # how much random numbers you will create
d = 10      # there are ten ads

import math
import random

toplam = 0
secilenler = []
sifirlar = [0] * d
birler = [0] * d
for n in range(1, N):
  add = 0
  max_th = 0
  for i in range(0, d):
    rasbeta = random.betavariate(birler[i] + 1, sifirlar[i] + 1)
    if rasbeta > max_th:
      max_th = rasbeta
      add = i
  secilenler.append(add)
  odul = veriler.values[n, add]
  if odul == 1:
    birler[add] = birler[add] + 1
  else:
    sifirlar[add] = sifirlar[add] + 1
  toplam = toplam + odul

print(toplam)

plt.hist(secilenler)
plt.show()

####################### Reinforced Learning #####################
#################################################################

#################################################################
####################### Reinforced Learning #####################

import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# ingilizce stopwords ler
dur = nltk.download('stopwords')

ps = PorterStemmer()

## PREPROCESSING
hepsi = []
for i in range(len(yorumlar)):
  # Noktalama isaretleri temizlenir (Punctiations are cleaned)
  yorum = re.sub('[^a-zA-Z]', ' ', yorumlar['Review'][i])
  # Butun harfker kucuk harf oldu (All letter became lowercase)
  yorum = yorum.lower()
  # Kelime kelime ayrildi (splited word by word)
  yorum = yorum.split()
  # Diyor ki eger kelime stopwords degilse o kelimenin kokunu bul
  # this line says if kelime is not a stopword then find the stem of the kelime
  # set icinde tekrar eden kelimleri bir kez barindiran bir kumedir | set is a set that include the repetitive words just one time 
  yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
  # join the yorum list but put a space between each 2 words
  yorum = ' '.join(yorum)
  hepsi.append(yorum)

## FEATURE EXTRACTION  
# BAG OF WORDS
from sklearn.feature_extraction.text import CountVectorizer
# max_feature : kac tane en cok kullanilan kelime alsin | how many most used words will get 
cv = CountVectorizer(max_features = 2000)
# sparse matrix 
X = cv.fit_transform(hepsi).toarray() # en cok kullanilan kelimelerin secildigi matris # y = f(x)
Y = yorumlar.iloc[:, 1].values # cikti kolonu

####################### Reinforced Learning #####################
#################################################################

#################################################################
########################## Deep Learning ########################

"""** Input, Hidden, Layer and parameters of Dense**"""
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(6, kernel_initializer = "uniform", activation = "relu", input_dim = 11, name = "gir"))
model.add(Dense(6, kernel_initializer = "uniform", activation = "relu", name = 'ara1'))
model.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid", name = 'cik'))

model.compile(optimizer = 'Adam',             # it is a form of stochastic gradient
              loss = 'binary_crossentropy',
              metrics = ['accuracy']) 
model.fit(X_train, y_train, epochs = 50)
y_pred = model.predict(X_test)
y_pred = y_pred > 0.5 

########################## Deep Learning ########################
#################################################################

#################################################################
####################### Dimension Reduction #####################

"""**PCA**"""

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

X_trainApca = pca.fit_transform(X_train)
X_testApca  = pca.transform(X_test)

"""**LDA**"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

lda = lda(n_components = 2)

# dikkat edilirse burada girdi degerlerinin ynainda ait olduklari siniflarida verdik (y boyutu)
X_trainAlda = lda.fit_transform(X_train, y_train)
X_testAlda = lda.fit_transform(X_test, y_test)

####################### Dimension Reduction #####################
#################################################################

#################################################################
####################### Dimension Reduction #####################

"""**k fold cross validation**"""

from sklearn.model_selection import  cross_val_score
'''
1. estimator : calssifier (bizim durum)
2. X
3. Y
4. cv : kac katmanli
'''
basari = cross_val_score(estimator = svc, X = X_train, y = y_train, cv = 4)

"""**Grid Search**"""
from sklearn.model_selection import GridSearchCV

'''
estimator  : neyi optimize etmke istedigim
param_grid : paramatlereler p dict i
scoring    : neye gore scorelanacak
cv         : kac katlamayla gidecek
n_jobs     : ayni anda calisacak is
'''

p = [{'C' : [1, 2, 3, 4, 5], 'kernel' : ['linear']},
     {'C' : [1, 10, 100, 1000], 'kernel' : ['rbf'], 'gamma' : [1, 0.5, 0.1, 0.01, 0.001]}]

gs = GridSearchCV(estimator = svc,
                  param_grid = p,
                  scoring = 'accuracy',
                  cv = 10,
                  n_jobs = -1)

grid_search = gs.fit(X_train, y_train)
eniyisonuc = grid_search.best_score_
eniyiparametre = grid_search.best_params_
print(eniyisonuc, '\n')
print(eniyiparametre)

####################### Dimension Reduction #####################
#################################################################

#################################################################
############################# XG Boost ##########################

"""** XGBoost **"""

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

############################# XG Boost ##########################
#################################################################
