from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle


#Finding the best parameters for RandomForest Clasifier using GridSearchCV
def random_forest_raw_train(x_train, y_train):
    randomforest_params = {
        'n_estimators': [100, 150, 200],
        'bootstrap': [True, False]
    }

    random_forest_classifier = RandomForestClassifier(max_features='sqrt', class_weight='balanced')
    grid_search_random_forest = GridSearchCV(estimator=random_forest_classifier, param_grid=randomforest_params, cv=4)

    train_rf_raw = grid_search_random_forest.fit(x_train, y_train)
    print(f"Best parameters are: {train_rf_raw.best_params_}\nBest score for the parameter is: {train_rf_raw.best_score_}")

    name = "random_forest_raw"
    fname = open(file=name, mode="wb")
    pickle.dump(obj=train_rf_raw, file=fname)
    
    return train_rf_raw


#Testing the model
def random_forest_test(model, x_test):
    y_pred = model.predict(x_test)
    return y_pred


#Augmented Dataset

#Finding the best parameters for RandomForest Clasifier using GridSearchCV
def random_forest_augmented_train(x_train, y_train):
    randomforest_params = {
        'n_estimators': [100, 150, 200],
        'bootstrap': [True, False]
    }

    random_forest_classifier = RandomForestClassifier(max_features='sqrt', class_weight='balanced')
    grid_search_random_forest = GridSearchCV(estimator=random_forest_classifier, param_grid=randomforest_params, cv=4)

    train_rf_augmented = grid_search_random_forest.fit(x_train, y_train)
    print(f"Best parameters are: {train_rf_augmented.best_params_}\n \
                Best score for the parameter is: {train_rf_augmented.best_score_}")

    name = "random_forest_augmented"
    fname = open(file=name, mode="wb")
    pickle.dump(obj=train_rf_augmented, file=fname)
    
    return train_rf_augmented



def random_forest_feature_ext_train(x_train, y_train):
    random_forest_classifier = RandomForestClassifier(n_estimators=200, bootstrap=False,
                                                         max_features='sqrt', class_weight='balanced')

    train_rf_feature_ext = random_forest_classifier.fit(x_train, y_train)
    name = "random_forest_feature_ext"
    fname = open(file=name, mode="wb")
    pickle.dump(obj=train_rf_feature_ext, file=fname)

    return train_rf_feature_ext