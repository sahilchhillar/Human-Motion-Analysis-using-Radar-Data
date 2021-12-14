from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import pickle


#Finding the best parameters for Gaussian Naive Bayes Clasifier using GridSearchCV
def gaussian_nb_raw_train(x_train, y_train):
    gaussian_nb_params = {
        'var_smoothing': [1e-9, 1e-10, 1e-11]
    }

    gaussian_nb = GaussianNB()
    grid_search_gnb = GridSearchCV(estimator=gaussian_nb, param_grid=gaussian_nb_params, cv=4)

    train_gnb_raw = grid_search_gnb.fit(x_train, y_train)
    print(f"Best parameters are: {train_gnb_raw.best_params_}\nBest score for the parameter is: {train_gnb_raw.best_score_}")

    name = "gaussian_nb_raw"
    fname = open(file=name, mode="wb")
    pickle.dump(obj=train_gnb_raw, file=fname)
    
    return train_gnb_raw


#Testing the model
def gaussian_nb_test(model, x_test):
    y_pred = model.predict(x_test)
    return y_pred


#Augmented Data

#Finding the best parameters for Gaussian Naive Bayes Clasifier using GridSearchCV
def gaussian_nb_augmented_train(x_train, y_train):
    gaussian_nb_params = {
        'var_smoothing': [1e-9, 1e-10, 1e-11]
    }

    gaussian_nb = GaussianNB()
    grid_search_gnb = GridSearchCV(estimator=gaussian_nb, param_grid=gaussian_nb_params, cv=4)

    train_gnb_augmented = grid_search_gnb.fit(x_train, y_train)
    print(f"Best parameters are: {train_gnb_augmented.best_params_}\n \
            Best score for the parameter is: {train_gnb_augmented.best_score_}")

    name = "gaussian_nb_augmented"
    fname = open(file=name, mode="wb")
    pickle.dump(obj=train_gnb_augmented, file=fname)
    
    return train_gnb_augmented



#Feature extraction
def gaussian_nb_feature_ext_train(x_train, y_train):
    gaussian_nb = GaussianNB()
    train_gnb_feature_ext = gaussian_nb.fit(x_train, y_train)
    name = "gaussian_nb_feature_ext"
    fname = open(file=name, mode="wb")
    pickle.dump(obj=train_gnb_feature_ext, file=fname)
    
    return train_gnb_feature_ext