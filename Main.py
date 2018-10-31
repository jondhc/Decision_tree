#   Decision Tree
#   Main.py
#   Created by J. Daniel Herrejón C. (153548) & Albert Joseph Eapen (151534) on 30-10-18
#   Copyright © 2018 jondhc-AlbertoJoseF All rights reserved.

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz

X = [[1, 1, 1, 0], [1, 1, 1, 1], [2,3,2,0], [3,2,2,0], [1,1,1,0], [3,1,1,0], [1,1,1,0], [3,3,1,0], [3,1,2,0], [3,2,2,1]]
Y = [1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
play = clf.predict([[3, 1, 2, 0]])
if play == [1]:
    print("The show must go on.")
else:
    print("Ahi nos vidrios miss :)")

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("DecisionTreeGraphFinalDeluxeEdition")

'''
Reference:

rainy = 1
overcast = 2
sunny = 3

cool = 1
mild = 2
hot = 3

normal = 1
high = 2

true = 1
false = 0

----------------------------------------

Data:

D1 yes rainy cool normal false
 D2 no rainy cool normal true
 D3 yes overcast hot high false
 D4 no sunny mild high false
 D5 yes rainy cool normal false
 D6 yes sunny cool normal false
 D7 yes rainy cool normal false
 D8 yes sunny hot normal false
 D9 yes sunny cool high false
 D10 no sunny mild high true

D11 ? sunny cool high false

'''
