"""Small utility to print the best params of an optuna database"""

import optuna
import os

study_name = "sol_optuna_64.db"
experiment_folder = os.path.abspath("/export/home/jalegria/Thesis/jf1uids/arena/data")
storage = f"sqlite:///{os.path.join(experiment_folder, study_name)}"

study = optuna.load_study(study_name=None, storage=storage)

best = study.best_trial

print("Best value: ", best.value)
print("Best params: ", best.params)
print(" User params: ", best.user_attrs)
