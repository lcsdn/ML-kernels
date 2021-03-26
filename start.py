from kernel_code.kernel_classifiers import KernelSVM
from kernel_code.kernels import SpectrumKernel
from kernel_code.data import load_data
from kernel_code.prediction import predictions, save

## Load the data
path_data = "data"
data = load_data(path_data)
X_train_list = [data["Xtr%d.csv" % i].seq.values for i in range(3)]
X_test_list = [data["Xte%d.csv" % i].seq.values for i in range(3)]
y_train_list = [data["Ytr%d.csv" % i].Bound.values for i in range(3)]

## Specify the hyperparameters
kernel = SpectrumKernel(5)

kwargs_models = [
    {"kernel": kernel, "reg_param": 0.13468477737454598},
    {"kernel": kernel, "reg_param": 0.1489916516067451},
    {"kernel": kernel, "reg_param": 0.056035539068055064}
]

## Compute the predictions
predictions_df = predictions(
    ModelClass=KernelSVM,
    X_train_list=X_train_list,
    y_train_list=y_train_list,
    X_test_list=X_test_list,
    kwargs_models=kwargs_models,
    preprocesser=None
)

## Save the predictions
save(
    name_predictions="predictions",
    folder_predictions="",
    predictions_df=predictions_df
)
print("Done! Predictions saved to 'predictions.csv'")