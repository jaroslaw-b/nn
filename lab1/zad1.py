from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets


iris = datasets.load_iris()
data = iris.data
labels = iris.target
      
mlp = MLPClassifier()

data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.5, random_state=1)
scaler = StandardScaler()
scaler.fit(data)
data_train_std = scaler.transform(data_train)
data_test_std = scaler.transform(data_test)
#data_train_std = data_train
#data_test_std = data_test    
     
mlp.fit(data_train_std, labels_train)

probabilities = mlp.predict(data_test_std)
print('Dokładność: %.2f' % accuracy_score(labels_test, probabilities))