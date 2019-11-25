from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
iris=datasets.load_iris()
print("iris data set loaded...")
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target)
classifier=KNeighborsClassifier(3).fit(x_train,y_train)
y_pred=classifier.predict(x_test)
print('result of classification using k-nn with k=1')
for r in range(0,len(x_test)):
    print("sample:",str(x_test[r]),"actual-label:",str(y_test[r]),"predicted-label",str(y_pred[r]))
print("classification Accuracy:",classifier.score(x_test,y_test));
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
