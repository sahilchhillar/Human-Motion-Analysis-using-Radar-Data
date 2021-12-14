from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import pickle


#Finding the best parameters for Gaussian Naive Bayes Clasifier using GridSearchCV
def knn_raw_train(x_train, y_train):
    knn_params = {
        'n_neighbors': [5, 8, 10]
    }

    knn = KNeighborsClassifier()

    grid_search_knn = GridSearchCV(estimator=knn, param_grid=knn_params, cv=4)

    train_knn_raw = grid_search_knn.fit(x_train, y_train)
    print(f"Best parameters are: {train_knn_raw.best_params_}\nBest score for the parameter is: {train_knn_raw.best_score_}")

    name = "knn_raw"
    fname = open(file=name, mode="wb")
    pickle.dump(obj=train_knn_raw, file=fname)
    
    return train_knn_raw


#Testing the model
def knn_test(model, x_test):
    y_pred = model.predict(x_test)
    return y_pred



#Augmented data

#Finding the best parameters for Gaussian Naive Bayes Clasifier using GridSearchCV
def knn_augmented_train(x_train, y_train):
    knn_params = {
        'n_neighbors': [5, 8, 10]
    }

    knn = KNeighborsClassifier()

    grid_search_knn = GridSearchCV(estimator=knn, param_grid=knn_params, cv=4)

    train_knn_augmented = grid_search_knn.fit(x_train, y_train)
    print(f"Best parameters are: {train_knn_augmented.best_params_}\n \
            Best score for the parameter is: {train_knn_augmented.best_score_}")

    name = "knn_augmented"
    fname = open(file=name, mode="wb")
    pickle.dump(obj=train_knn_augmented, file=fname)
    
    return train_knn_augmented



#Feature extraction
def knn_feature_ext_train(x_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=5)
    train_knn_feature_ext = knn.fit(x_train, y_train)
    name = "knn_feature_ext"
    fname = open(file=name, mode="wb")
    pickle.dump(obj=train_knn_feature_ext, file=fname)
    
    return train_knn_feature_ext