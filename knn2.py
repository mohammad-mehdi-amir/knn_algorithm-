from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = pd.read_csv("/Users/mohamad/Documents/exercise advance programming/mini project/Breast-cance1r.csv")

train_data, test_data = train_test_split(data, test_size=0.20,random_state=10)

x_train = train_data.iloc[:,:-1]
y_train = train_data.iloc[:, -1]

x_test = test_data.iloc[:,:-1]
y_test = test_data.iloc[:, -1]

dic = {}
for k in range(1, 40):
    classifier = KNeighborsClassifier(n_neighbors=k)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    classifier.fit(x_train, y_train)
    accuracy = classifier.score(x_test, y_test)
    dic[k] = accuracy

temp = 0.0
for accur in dic.values():
    if accur > temp:
        temp = accur




from sklearn.metrics import precision_recall_fscore_support
y_pred = classifier.predict(x_test)

precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='binary') 

print(f"Accuracy: {temp * 100:.2f}%")
print(f"F-score: {fscore*100:.2f}%")
