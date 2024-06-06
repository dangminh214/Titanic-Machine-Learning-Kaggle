# Title: Machine Learning Practice in Kaggle:Titanic 
# Technical Stack: 
Panda 
Sk-Learn: Random Forrest Classifier 

# Visualize Data 
![grafik](https://github.com/dangminh214/Titanic-Machine-Learning-Kaggle/assets/51837721/2eb0198d-3d3f-4010-9f21-441230164c54)

# Take data from given CVS files: 
![grafik](https://github.com/dangminh214/Titanic-Machine-Learning-Kaggle/assets/51837721/f8e95024-2dc4-4c9d-823b-882d16fa7b5e)

# Using Random Forrest Classification 
```ruby 
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
```



