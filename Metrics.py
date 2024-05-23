from sklearn import metrics
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


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
    X = np.load('Keypoints/X_metrics.npy')
    y = np.load('Keypoints/y_metrics.npy')

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=40)
    return X_test, y_test


def thereshold_vector(vector, threshold):
    return np.where(vector < threshold, 0, 1)


def LoadModel():
    model =  tf.keras.models.load_model('TrainedModel/ModeloTest2.h5')
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
    print(conf_matrix)

    # Etiquetas de las clases


    # Crear la figura y el eje
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=np.arange(1, len(actions)+1))
    
    # Graficar la matriz de confusión
    fig, ax = plt.subplots(figsize=(13, 10))
    disp.plot(ax=ax)
    plt.title('Matriz de Confusión')
    plt.xticks(fontsize=15)  # Increase font size of x-axis labels
    plt.yticks(fontsize=15)  # Increase font size of y-axis labels
    
    # Añadir etiquetas y título
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta Verdadera')
    plt.title('Matriz de Confusión')
    
    
    # Crear leyenda
    legend_labels = ['Alerta de Caída', 'Sentándose', 'Levantándose', 'Sentado', 'Caminando']
    plt.legend(legend_labels, loc='upper right')
    plt.savefig('ConfusionMatrix.pdf')
    # Mostrar la matriz de confusión
    plt.show()
def metrics_plot(test, pred, actions):
    # Calculate the recall, F1 score and precision for each class
    recall_scores = recall_score(test, pred, average=None)
    f1_scores = f1_score(test, pred, average=None)
    precision_scores = metrics.precision_score(test, pred, average=None)

    # Create the figure and axis
    plt.figure(figsize=(10, 6))

    # Plot the recall, F1 score and precision scores
    barWidth = 0.25
    r1 = np.arange(len(actions))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    plt.bar(r1, recall_scores, color='blue', width=barWidth, edgecolor='grey', label='Recall')
    plt.bar(r2, f1_scores, color='green', width=barWidth, edgecolor='grey', label='F1 Score')
    plt.bar(r3, precision_scores, color='red', width=barWidth, edgecolor='grey', label='Precision')

    # Add xticks on the middle of the group bars
    plt.xlabel('Class', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(recall_scores))], actions)

    # Add labels and title
    plt.ylabel('Score')
    plt.title('Recall, F1 Score and Precision per Class')

    # Create legend & Show graphic
    plt.legend()
    plt.savefig('Metrics_model_uninorte.pdf')
    plt.show()

if __name__=="__main__":
    actions = np.array(['Alerta de Caída',
                        'Sentándose',
                        'Levantándose',
                        'Sentado',
                        'Caminando'])
    X_test, y_test = loadData()
    model = LoadModel()
    y_pred = model.predict(X_test)
    y_pred_binary = thereshold_vector(y_pred, 0.5)
    # Remove the 1st and 2nd components from each component of y_pred_binary and y_test
    y_pred_binary = np.delete(y_pred_binary, [1, 2], axis=1)
    y_test = np.delete(y_test, [1, 2], axis=1)
    print(y_pred_binary, y_test)
    pred, test = TensorToVector(y_pred_binary, y_test)
    print(pred, test)
    # metric = classification_report(test, pred, target_names=actions, output_dict=True)
    # df = pd.DataFrame(metric)
    # df
    # df.to_csv('metrics90T.csv', index=True)
    
    classifier= estimator(model, actions)
    ConfusionMatrix(pred, test, actions)
    metrics_plot(test, pred, actions)

    



# Calcula las métricas
# accuracy = metrics.accuracy_score(y_true, y_pred)
# precision = metrics.precision_score(y_true, y_pred)
# recall = metrics.recall_score(y_true, y_pred)
# f1_score = metrics.f1_score(y_true, y_pred)

# print(f'Accuracy: {accuracy}')
# print(f'Precision: {precision}')
# print(f'Recall: {recall}')
# print(f'F1 Score: {f1_score}')