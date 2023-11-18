#Exemplo do código em sala
import pandas as pd
#1. Avaliar a frequencia das classes
dados = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/2023/fertility_Diagnosis.txt', sep = ',')
# print(dados['Output'].value_counts())
#---------------------------------------
#Treinamento com os dados desbalanceados
#Segmentar os dados
dados_classes = dados['Output']
dados_atributos = dados.drop(columns = ['Output'])
# print(dados_atributos.columns)
#---------------------------------------
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier() #Constrói o meta estimador para treinamento
atr_train, atr_test, class_train, class_test = train_test_split(dados_atributos, dados_classes, test_size =0.3 )
# print(class_test)

#Treinar o modelo
fertility_tree = tree.fit(atr_train, class_train)
#Testar (avaliar) o modelo
Class_predict = fertility_tree.predict(atr_test)
# print(Class_predict)
#------------------------------------
#MATRIZ DE CONTINGÊNCIA
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm=confusion_matrix(class_test, Class_predict)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = fertility_tree.classes_)
disp.plot()
plt.show()