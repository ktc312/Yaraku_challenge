# __author__ = 'ktc312'
#  -*- coding: utf-8 -*-
# coding: utf-8
from sklearn.cross_validation import train_test_split
from sklearn import model_selection
from keras.wrappers import scikit_learn
from sklearn.preprocessing import Imputer
from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd


bcw_df = pd.read_pickle('bcw_df.pkl')


imputer = Imputer(strategy="median")
imputer.fit(bcw_df.loc[:, bcw_df.columns != 'malignant'])

X = imputer.transform(bcw_df[['clump_thickness', 'cell_size', 'cell_shape', 'MA', 'SE_cell_size', 'bare_nuclei',
                              'bland_chromatin', 'normal_nucleoli', 'mitoses']])
y = bcw_df['malignant']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)


def create_model():
    model = Sequential()
    model.add(Dense(units=7, kernel_initializer='RandomNormal', activation='sigmoid', input_dim=9))
    model.add(Dense(units=6, kernel_initializer='RandomNormal', activation='relu'))
    model.add(Dense(units=1, kernel_initializer='RandomNormal', activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model


sk_model = scikit_learn.KerasClassifier(build_fn=create_model)

epochs = [100, 120, 150, 170, 200]
batches = [3, 5, 7, 9]

param_grid = dict(epochs=epochs, batch_size=batches)

grid = model_selection.GridSearchCV(estimator=sk_model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Best: 0.971377 using {'batch_size': 7, 'epochs': 120}