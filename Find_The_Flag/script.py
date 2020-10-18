import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
plt.style.use('seaborn-colorblind')

flags = pd.read_csv('flags.csv', header = 0)
print(flags.columns)
print(flags.head())

labels = flags[['Language']]
data = flags[["Red", "Green", "Blue", "Gold",
 "White", "Black", "Orange",
 "Circles",
"Crosses","Saltires","Quarters","Sunstars",
"Crescent","Triangle"]]

train_data, test_data, train_labels, test_labels = train_test_split(data,labels, random_state=1)


scores = []
for i in range(1,20):
  tree = DecisionTreeClassifier(random_state=1, max_depth =i)
  tree.fit(train_data,train_labels)
  score = tree.score(test_data, test_labels)
  scores.append(score)

plt.plot(range(1,20),scores)
plt.savefig('DecisionTree.png')
plt.show()
