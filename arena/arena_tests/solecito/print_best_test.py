"""Small utility to print the best params of an optuna database"""

import optuna
from pathlib import Path

optuna_base = "arena/arena_tests/solecito/models/optuna"
Path(optuna_base).mkdir(parents=True, exist_ok=True)

# ---- Optuna study config ----
study_db = Path(optuna_base) / "optuna_study.db"
study_name = "solecito_optuna"
storage = f"sqlite:///{study_db}"

study = optuna.load_study(study_name=None, storage=storage)

best = study.best_trial

print("Best value: ", best.value)
print("Best params: ", best.params)
print(" User params: ", best.user_attrs)
