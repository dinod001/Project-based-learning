import os
import joblib
import logging
from sklearn.model_selection import GridSearchCV

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, param_grid=None, cv=5, scoring='accuracy', n_jobs=-1):
        """
        param_grid: dict
            Hyperparameter grid for GridSearchCV.
        cv: int
            Number of cross-validation folds.
        scoring: str
            Scoring metric.
        n_jobs: int
            Number of parallel jobs (-1 uses all cores).
        """
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs

    def train(self, X_train, Y_train, model):
        if self.param_grid:
            logger.info("Starting GridSearchCV for hyperparameter tuning...")
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=self.param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=1
            )
            grid_search.fit(X_train, Y_train)
            best_model = grid_search.best_estimator_
            best_score = grid_search.best_score_

            logger.info(f"Best Parameters: {grid_search.best_params_}")
            logger.info(f"Best CV Score: {best_score:.4f}")
            return best_model, best_score
        else:
            logger.info("No param_grid provided. Training model directly...")
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            return model, train_score

    def save_model(self, model, filePath):
        os.makedirs(os.path.dirname(filePath), exist_ok=True)
        joblib.dump(model, filePath)
        logger.info(f"Model saved to {filePath}")

    def load(self, filepath):
        if not os.path.exists(filepath):
            raise ValueError("Can't load. File not found.")
        logger.info(f"Loading model from {filepath}")
        return joblib.load(filepath)
