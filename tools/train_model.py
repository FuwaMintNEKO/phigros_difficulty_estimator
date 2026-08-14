import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os
import json
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print('[提示] xgboost 未安装，将仅使用 RandomForest 和 GradientBoosting')


class DifficultyModel:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        self.calibration_slope = 0.0
        self.calibration_intercept = 0.0

    def _compute_sample_weights(self, y, levels=None):
        y_min = np.min(y)
        y_max = np.max(y)
        y_range = y_max - y_min + 0.1

        weight_power = 1.5
        normalized = (y - y_min) / y_range
        weights = 1.0 + normalized ** weight_power * 2.0

        if levels is not None:
            for i, level in enumerate(levels):
                if level == 'AT':
                    weights[i] *= 30.0
                elif level in ('IN',):
                    weights[i] *= 2.0

        return weights

    def train(self, X, y, level_name='unknown', calibrate=True, levels=None):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        y_train_levels = None
        if levels is not None:
            _, y_train_levels, _, _ = train_test_split(
                range(len(levels)), levels, test_size=0.15, random_state=42
            )

        sample_weights = self._compute_sample_weights(y_train, y_train_levels)

        rf_model = RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
        )
        rf_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

        gb_model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        gb_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

        self.models['random_forest'] = rf_model
        self.models['gradient_boosting'] = gb_model

        if HAS_XGBOOST:
            xgb_model = XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            xgb_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            self.models['xgboost'] = xgb_model

        self.is_fitted = True

        if calibrate:
            y_train_pred = self.predict(X_train, 'ensemble', apply_calibration=False)
            residuals = y_train - y_train_pred
            A = np.vstack([y_train_pred, np.ones_like(y_train_pred)]).T
            coeffs, _, _, _ = np.linalg.lstsq(A, residuals, rcond=None)
            self.calibration_slope = coeffs[0]
            self.calibration_intercept = coeffs[1]

        return self._evaluate_models(X_test_scaled, y_test, level_name)

    def _evaluate_models(self, X_test, y_test, level_name):
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            within_05 = np.mean(np.abs(y_pred - y_test) <= 0.5) * 100
            within_10 = np.mean(np.abs(y_pred - y_test) <= 1.0) * 100

            results[name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'within_05': within_05,
                'within_10': within_10,
            }

        return results

    def predict(self, X, model_name='ensemble', apply_calibration=True):
        if not self.is_fitted:
            raise RuntimeError('模型尚未训练')

        X_scaled = self.scaler.transform(X)

        if model_name == 'ensemble':
            predictions = []
            weights = []
            if 'random_forest' in self.models:
                predictions.append(self.models['random_forest'].predict(X_scaled))
                weights.append(1.0)
            if 'gradient_boosting' in self.models:
                predictions.append(self.models['gradient_boosting'].predict(X_scaled))
                weights.append(3.0)
            if HAS_XGBOOST and 'xgboost' in self.models:
                predictions.append(self.models['xgboost'].predict(X_scaled))
                weights.append(1.0)

            if predictions:
                weight_sum = sum(weights)
                normal_weights = [w / weight_sum for w in weights]
                raw_pred = np.average(predictions, axis=0, weights=normal_weights)
                if apply_calibration:
                    return raw_pred + self.calibration_slope * raw_pred + self.calibration_intercept
                return raw_pred
            return None
        elif model_name in self.models:
            raw_pred = self.models[model_name].predict(X_scaled)
            if apply_calibration:
                return raw_pred + self.calibration_slope * raw_pred + self.calibration_intercept
            return raw_pred
        else:
            raise ValueError(f'未知模型: {model_name}')

    def get_feature_importance(self):
        if 'random_forest' not in self.models:
            return None

        rf = self.models['random_forest']
        if self.feature_names is None:
            return rf.feature_importances_

        importance = list(zip(self.feature_names, rf.feature_importances_))
        importance.sort(key=lambda x: x[1], reverse=True)
        return importance

    def save(self, filepath):
        save_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'calibration_slope': self.calibration_slope,
            'calibration_intercept': self.calibration_intercept,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        self.models = save_data['models']
        self.scaler = save_data['scaler']
        self.feature_names = save_data['feature_names']
        self.is_fitted = save_data['is_fitted']
        self.calibration_slope = save_data.get('calibration_slope', 0.0)
        self.calibration_intercept = save_data.get('calibration_intercept', 0.0)


def print_evaluation_results(results, level_name):
    print(f'\n{"="*50}')
    print(f'  [{level_name}] 模型评估结果')
    print(f'{"="*50}')
    for model_name, metrics in results.items():
        print(f'\n  {model_name}:')
        print(f'    MAE  (平均绝对误差): {metrics["mae"]:.4f}')
        print(f'    RMSE (均方根误差):   {metrics["rmse"]:.4f}')
        print(f'    R²   (决定系数):     {metrics["r2"]:.4f}')
        print(f'    偏差≤0.5 的占比:     {metrics["within_05"]:.1f}%')
        print(f'    偏差≤1.0 的占比:     {metrics["within_10"]:.1f}%')