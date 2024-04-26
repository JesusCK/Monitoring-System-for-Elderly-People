from sklearn import metrics
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


class estimator:
  _estimator_type = ''
  classes_=[]
  def __init__(self, model, classes):
    self.model = model
    self._estimator_type = 'classifier'
    self.classes_ = classes
  def predict(self, X):
    y_prob= self.model.predict(X)
    y_pred = y_prob.argmax(axis=1)
    return y_pred
  
  

def loadData():
    X = np.load('Keypoints/X5.npy')
    y = np.load('Keypoints/y5.npy')

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=41)
    return X_test, y_test


def thereshold_vector(vector, threshold):
    return np.where(vector < threshold, 0, 1)


def LoadModel():
    model =  tf.keras.models.load_model('TrainedModel/ModeloBacano6.h5')
    model.summary()
    return model

def TensorToVector(res,y_test):
    pred =[]
    test = []
    for i in res:
        pred.append(np.argmax(i))
    for j in y_test:
        test.append(np.argmax(j))
    return pred, test

def ConfusionMatrix(pred, test, actions):
        # Ejemplo de una matriz de confusión
    # Reemplaza estas predicciones y etiquetas verdaderas con las tuyas


    # Calcular la matriz de confusión
    conf_matrix = confusion_matrix(pred, test)

    # Etiquetas de las clases


    # Crear la figura y el eje
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=0.8)  # Ajustar el tamaño de la fuente
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=actions, yticklabels=actions)

    # Añadir etiquetas y título
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta Verdadera')
    plt.title('Matriz de Confusión')

    # Mostrar la matriz de confusión
    plt.show()
def Recall(test,pred,actions):
   # Calcular el recall para cada clase
    recall_scores = recall_score(test, pred, average=None)

    # Etiquetas de las clases
    labels = ['Clase 0', 'Clase 1']

    # Crear la figura y el eje
    plt.figure(figsize=(8, 6))

    # Graficar los valores de recall
    plt.bar(actions, recall_scores, color=['blue', 'green'])

    plt.xticks(fontsize=8)  # Tamaño de letra de las etiquetas del eje x
    plt.yticks(fontsize=8)
    # Añadir etiquetas y título
    plt.xlabel('Clase')
    plt.ylabel('Recall')
    plt.title('Recall por Clase')

    # Mostrar la gráfica
    plt.show()

def F1Score(test, pred, actions):
    # Calculate the F1 score for each class
     f1_scores = f1_score(test, pred, average=None)

     # Class labels
     labels = ['Class 0', 'Class 1']

     # Create the figure and axis
     plt.figure(figsize=(8, 6))

     # Plot the F1 scores
     plt.bar(actions, f1_scores, color=['blue', 'green'])

     plt.xticks(fontsize=8)  # Font size of the x-axis labels
     plt.yticks(fontsize=8)
     # Add labels and title
     plt.xlabel('Class')
     plt.ylabel('F1 Score')
     plt.title('F1 Score per Class')

     # Show the plot
     plt.show()



if __name__=="__main__":
    actions = np.array(['Alerta de Caida',
                            'Normal',
                            'Normal',
                            'Sentandose',
                            'Levantandose',
                            'Sentado',
                            'Caminando'])
    X_test, y_test = loadData()
    model = LoadModel()
    y_pred = model.predict(X_test)
    y_pred_binary = thereshold_vector(y_pred, 0.5)
    pred, test = TensorToVector(y_pred_binary, y_test)
    print(pred, test)
    metric = classification_report(test, pred, target_names=actions)
    print(metric)
    classifier= estimator(model, actions)
    ConfusionMatrix(pred, test, actions)
    Recall(test,pred,actions)
    F1Score(test, pred, actions)

    



# Calcula las métricas
# accuracy = metrics.accuracy_score(y_true, y_pred)
# precision = metrics.precision_score(y_true, y_pred)
# recall = metrics.recall_score(y_true, y_pred)
# f1_score = metrics.f1_score(y_true, y_pred)

# print(f'Accuracy: {accuracy}')
# print(f'Precision: {precision}')
# print(f'Recall: {recall}')
# print(f'F1 Score: {f1_score}')