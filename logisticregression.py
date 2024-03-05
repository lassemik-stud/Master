from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
for thing in X: 
    print(thing)
#clf = LogisticRegression(random_state=0,max_iter=10000).fit(X, y)