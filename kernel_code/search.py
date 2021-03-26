import optuna

from .training import cross_validate
from .kernel_classifiers import KernelSVM

class OptunaObjective:
    preprocesser = None
    num_folds = 3
    
    def __init__(self, X, y, K=None):
        self.X = X
        self.y = y
        self.K = K
    
    def __call__(self, trial):
        kwargs_model = self.suggest_kwargs(trial)
        _, _, val_acc = cross_validate(
            ModelClass=self.ModelClass,
            X=self.X,
            y=self.y,
            k=self.num_folds,
            show_progression=False,
            preprocesser=self.preprocesser,
            K=self.K,
            **kwargs_model
        )
        return val_acc

def fixed_kernel_objective(kernel, num_folds=3):
    class Objective(OptunaObjective):
        ModelClass = KernelSVM

        def suggest_kwargs(self, trial):
            reg_param = trial.suggest_loguniform('reg_param', 1e-10, 1e20)
            kwargs_model = {"kernel": kernel, "reg_param":reg_param}
            return kwargs_model
    Objective.num_folds = num_folds
    return Objective

class HyperParamSearcher:
    def __init__(self, X_train_list, y_train_list, OptunaObjective, kernel=None):
        print("Searcher initialisation")
        self.Objective = OptunaObjective
        self._X_train_list = X_train_list
        self._y_train_list = y_train_list
        if kernel is not None:
            self._K_list = []
            for i, X_train in enumerate(X_train_list):
                print("Computing the kernel matrix on dataset %d" % i)
                self._K_list.append(kernel.pairwise_matrix(X_train))
        else:
            self._K_list = [None] * 3
        self.studies = [optuna.create_study(direction="maximize") for i in range(3)]
    
    def search(self, n_trials, *sets_to_search):
        if len(sets_to_search) == 0:
            sets_to_search = range(3)
        for i in sets_to_search:
            print("-" * 35)
            print("Optimising parameters for dataset %d" % i)
            print("%d trials" % n_trials)
            print("-" * 35)
            X_train, y_train, K = self._X_train_list[i], self._y_train_list[i], self._K_list[i]
            objective = self.Objective(X_train, y_train, K)
            self.studies[i].optimize(objective, n_trials=n_trials)
    
    def plot_slice(self):
        for i in range(3):
            optuna.visualization.matplotlib.plot_slice(self.studies[i])
    
    def get_best(self):
        best_hparams = [study.best_params if len(study.trials) else None for study in self.studies]
        best_losses = [study.best_trial.values[0] if len(study.trials) else None for study in self.studies]
        return best_hparams, best_losses
    
    def __repr__(self):
        best_hparams, best_losses = self.get_best()
        info = "Hyperparameters:\n" + \
        "\n".join(str(hparams) for hparams in best_hparams) + \
        "\n\nLosses:\n" + \
        "\n".join(str(loss) for loss in best_losses) + \
        "\n\nNumber of trials:\n" + \
        "\n".join(str(len(study.trials)) for study in self.studies)
        return info