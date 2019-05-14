import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
import os
import csv

from utils.io import create_dir

path1 = "/scr2/fabiof/repos/GNforInteraction/experiments/baseline_auto_predictor_3_object_dataset_bn_tflearn_lr_0_0001/latent_vectors_initial_image_of_full_test_set_5001_iterations_trained/latent_vectors_baseline_auto_predictor_dataset_tfrecords_5_objects_50_rollouts_padded.pkl"
#path1 = "/scr2/fabiof/repos/GNforInteraction/experiments/baseline_auto_predictor_3_object_dataset_bn_tflearn_lr_0_0001/latent_vectors_initial_image_of_full_test_set_5001_iterations_trained/latent_vectors_baseline_auto_predictor_dataset_tfrecords_5_objects_50_rollouts_padded_novel_shapes.pkl"

path2 = "/scr2/fabiof/repos/GNforInteraction/experiments/singulation22_latent256_3_objects_edges_send_s_e_g_m_edge_MLP/latent_vectors_initial_image_of_full_test_set_5226_iterations_trained/latent_vectors_gn_dataset_tfrecords_5_objects_50_rollouts_padded.pkl"
#path2 = "/scr2/fabiof/repos/GNforInteraction/experiments/singulation22_latent256_3_objects_edges_send_s_e_g_m_edge_MLP/latent_vectors_initial_image_of_full_test_set_5226_iterations_trained/latent_vectors_gn_dataset_tfrecords_5_objects_50_rollouts_padded_novel_shapes.pkl"


path = os.path.normpath(path1)
dataset_name = path.split(os.sep)[8][:-4]
csv_name = dataset_name.split("dataset_")[1] + "_regressions.csv"
df_name = dataset_name.split("dataset_")[1] + ".pkl"

df1 = pd.read_pickle(path1)
df2 = pd.read_pickle(path2)
data_pipeline = [(df1, "latent_vector_init_img", path1), (df2, "latent_vector_encoder_output_init_img", path2), (df2, "latent_vector_core_output_init_img", path2)]

""" for regression """
parameters_ridge = {
        "alpha": [0.1, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0],
        "fit_intercept": [True, False],
        "normalize": [True, False],
        "max_iter": [1000]
}

parameters_lasso = {
        "alpha": [0.1, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0],
        "fit_intercept": [True, False],
        "normalize": [True, False],
        "max_iter": [1000]
}

parameters_svr = {
        "kernel": ["rbf", "linear"]
}


""" for classification """
parameters_mlp = {
    "solver": ["lbfgs", "adam"],
    "max_iter": [5000],
    "learning_rate_init": [1e-3]
}

#parameters_logistic_regression = {
#    "penalty": ["l1", "l2"],
#    "fit_intercept": [True, False],
#    "C": [0.1, 0.5, 1.0, 1.5],
#    "max_iter": [5000]
#}

#parameters_svc = {
#        "kernel": ["linear", "rbf"],
#        "max_iter": [5000]
#}

#parameters_knc = {
#        "n_neighbors": [5, 10, 15],
#        "weights": ["uniform", "distance"]
#}

estimator_pipeline = [(Ridge(), parameters_ridge), (Lasso(), parameters_lasso), (SVR(), parameters_svr), (MLPRegressor(hidden_layer_sizes=(64,64)), parameters_mlp)]
#estimator_pipeline = [(SVC(), parameters_svc), (LogisticRegression(), parameters_logistic_regression),
#                      (KNeighborsClassifier(), parameters_knc), (MLPClassifier(), parameters_mlp)]

sub_dir_name = "latent_space_analysis"
dir_path, _ = create_dir("../experiments", sub_dir_name)


with open(os.path.join(dir_path, csv_name), 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter='\t', lineterminator='\n', )
    writer.writerow(["method name", "latent space data used", "estimator", "MSE over 25% test set (n_samples)", "precision (global metric / micro)", "recall (global metric / micro)", "best params after 5CV grid search"])

    df = pd.DataFrame(columns=["method name", "latent space data used", "estimator", "MSE", "y_true", "y_pred"])

    for tpl in data_pipeline:
        path = os.path.normpath(tpl[2])
        method_name = path.split(os.sep)[6]

        df = tpl[0]
        latent_space_data_used = tpl[1]
        X = df[latent_space_data_used]
        y = df["exp_len"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        X_train = [x.flatten() for x in X_train.values]
        y_train = [y for y in y_train.values]

        X_test = [x.flatten() for x in X_test.values]
        y_test = [y for y in y_test.values]

        for estimator_tpl in estimator_pipeline:
            estimator = estimator_tpl[0]
            parameters = estimator_tpl[1]
            estimator_name = estimator.__class__.__name__
            print("Running estimator: {} 5 CV grid search with params: {}".format(estimator_name, parameters))

            cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

            classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=parameters, n_jobs=-1)
            classifier.fit(X_train, y_train)
            estimator = estimator.set_params(**classifier.best_params_)
            best_params_dict = classifier.best_params_
            print("best params found are: {}".format(best_params_dict))

            y_test_predictions = classifier.predict(X_test)
            #acc = accuracy_score(y_true=y_test, y_pred=y_test_predictions)
            #precision = precision_score(y_true=y_test, y_pred=y_test_predictions, average="micro")
            #recall = recall_score(y_true=y_test, y_pred=y_test_predictions, average="micro")
            #report = classification_report(y_true=y_test, y_pred=y_test_predictions)
            error = mean_squared_error(y_true=y_test, y_pred=y_test_predictions)
            print("MSE on test set is: {} ({})".format(error, len(y_test)))
            #print("Accuracy on test set is: {} ({})".format(acc, len(y_test)))
            #writer.writerow([method_name, latent_space_data_used, estimator_name, str(acc) + " ({})".format(len(y_test)), precision, recall, best_params_dict])
            writer.writerow([method_name, latent_space_data_used, estimator_name, str(error) + " ({})".format(len(y_test)), "-", "-", best_params_dict])
            #df = df.append({"method name": method_name, "latent space data used": latent_space_data_used,
            #                "estimator name": estimator_name, "acc": acc, "classificaton report": report,
            #                "y_true": y_test, "y_pred": y_test_predictions, "estimator": classifier}, ignore_index=True)
            df = df.append({"method name": method_name, "latent space data used": latent_space_data_used,
                            "estimator name": estimator_name, "mse error": error, 
                            "y_true": y_test, "y_pred": y_test_predictions, "estimator": classifier}, ignore_index=True)
            df.to_pickle(df_name)

            csv_file.flush()


