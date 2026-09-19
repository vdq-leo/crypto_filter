def meta_sizing_cal(meta_probs):
    """Calibrate meta probabilities to sizing"""
    # return np.clip(2 * meta_probs - 1, 0, 1)
    return np.where(meta_probs > 0.5, meta_probs, 0)

def recursive_unwrap(model):
    """Recursively peel off model wrappers to find the base model for SHAP."""
    # Handle StatsModelsWrapper or similar that stores results in .results
    if hasattr(model, 'results'):
        return model.results
    if hasattr(model, 'model'):
        return recursive_unwrap(model.model)
    return model

class CombinedLongShortModelWrapper:
    def __init__(self, model_long, model_short):
        self.model_long = model_long
        self.model_short = model_short
        self.classes_ = np.array([-1, 0, 1])
        
    def predict_proba(self, X):
        p_long = self.model_long.predict_proba(X)
        p_short = self.model_short.predict_proba(X)
        
        c_l = {c: i for i, c in enumerate(self.model_long.classes_)}
        c_s = {c: i for i, c in enumerate(self.model_short.classes_)}
        
        prob_long = p_long[:, c_l[1]] if 1 in c_l else np.zeros(len(X))
        prob_short = p_short[:, c_s[-1]] if -1 in c_s else np.zeros(len(X))
        prob_neutral = 1.0 - (prob_long + prob_short)
        
        total = prob_long + prob_short + prob_neutral
        total[total == 0] = 1e-9
        return np.column_stack([prob_short/total, prob_neutral/total, prob_long/total])
        
    def predict(self, X):
        p_long = self.model_long.predict(X)
        p_short = self.model_short.predict(X)
        
        out = np.zeros(len(X))
        out[p_long == 1] = 1
        out[p_short == -1] = -1
        out[(p_long == 1) & (p_short == -1)] = 0
        return out

def get_shap_explainer(model, X_sample):
    """Select the most appropriate SHAP explainer for a given model."""
    raw_model = recursive_unwrap(model)
    
    # Try TreeExplainer for tree-based models (XGB, RF)
    try:
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from xgboost import XGBRegressor, XGBClassifier
        if isinstance(raw_model, (RandomForestRegressor, RandomForestClassifier, XGBRegressor, XGBClassifier)):
            return shap.TreeExplainer(raw_model)
    except ImportError:
        pass
        
    # Fallback to generic Explainer, but pass the prediction function for better compatibility
    try:
        if hasattr(raw_model, 'predict_proba'):
            # For classification, explain probabilities of the positive class or all classes
            return shap.Explainer(raw_model.predict_proba, X_sample)
        elif hasattr(raw_model, 'predict'):
            return shap.Explainer(raw_model.predict, X_sample)
    except Exception:
        pass
        
    # Last resort: let SHAP try to guess
    return shap.Explainer(raw_model)