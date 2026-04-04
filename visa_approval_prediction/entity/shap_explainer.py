import numpy as np
import pandas as pd
import shap
from visa_approval_prediction.logger import logging
from visa_approval_prediction.constants import LOCAL_DATA_FILE_PATH

TREE_MODELS = (
    "DecisionTreeClassifier", "RandomForestClassifier",
    "GradientBoostingClassifier", "XGBClassifier",
    "LGBMClassifier", "CatBoostClassifier", "ExtraTreesClassifier",
)


class VisaShapExplainer:
    """Generates SHAP-based explanations for visa approval predictions."""

    def __init__(self, model):
        self.preprocessor = model.preprocessing_object
        self.classifier = model.trained_model_object
        self.model_type = type(self.classifier).__name__

        if self.model_type in TREE_MODELS:
            self.explainer = shap.TreeExplainer(self.classifier)
        else:
            background = self._build_background(n_samples=50)
            self.explainer = shap.KernelExplainer(self.classifier.predict_proba, background)

        self.feature_mapping = self._build_feature_mapping()
        logging.info(
            f"SHAP explainer initialized "
            f"(model={self.model_type}, "
            f"explainer={type(self.explainer).__name__}, "
            f"transformed_features={len(self.feature_mapping)})"
        )

    def _build_background(self, n_samples=50):
        df = pd.read_csv(LOCAL_DATA_FILE_PATH).head(n_samples * 2)
        from datetime import date
        df['company_age'] = date.today().year - df['yr_of_estab']
        df.drop(columns=['case_status', 'yr_of_estab'], errors='ignore', inplace=True)
        transformed = self.preprocessor.transform(df)
        return shap.kmeans(transformed, min(n_samples, len(transformed)))

    def _build_feature_mapping(self):
        mapping = []
        for name, transformer, columns in self.preprocessor.transformers_:
            if name == 'remainder':
                continue
            if name == 'OneHotEncoder':
                for i, col in enumerate(columns):
                    n_cats = len(transformer.categories_[i])
                    mapping.extend([col] * n_cats)
            else:
                for col in columns:
                    mapping.append(col)
        return mapping

    def explain(self, input_df, input_data, prediction_result):
        transformed = self.preprocessor.transform(input_df)
        raw_shap = self.explainer.shap_values(transformed)

        # Extract SHAP values for the Denied class (label=1).
        # Certified=0, Denied=1. We want the "Denied" class SHAP values so that:
        #   positive SHAP = pushes toward Denied (weakness)
        #   negative SHAP = pushes toward Certified (strength)
        denied_class_idx = 1
        if hasattr(self.classifier, 'classes_'):
            classes = list(self.classifier.classes_)
            if 1 in classes:
                denied_class_idx = classes.index(1)

        if isinstance(raw_shap, list):
            shap_values = np.array(raw_shap[denied_class_idx][0])
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            shap_values = raw_shap[0, :, denied_class_idx]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 2:
            shap_values = raw_shap[0]
            if denied_class_idx == 0:
                shap_values = -shap_values
        else:
            shap_values = np.array(raw_shap[0])

        # Aggregate transformed-feature SHAP values back to original features
        feature_shap = {}
        for idx, feature_name in enumerate(self.feature_mapping):
            if idx < len(shap_values):
                feature_shap[feature_name] = (
                    feature_shap.get(feature_name, 0.0) + shap_values[idx]
                )

        sorted_features = sorted(
            feature_shap.items(), key=lambda x: abs(x[1]), reverse=True
        )

        strengths = []
        weaknesses = []
        suggestions = []

        for feature_name, shap_val in sorted_features:
            if abs(shap_val) < 0.01:
                continue

            msg = self._format_message(feature_name, shap_val, input_data)

            if shap_val < 0:
                strengths.append(msg)
            else:
                weaknesses.append(msg)
                suggestion = self._get_suggestion(feature_name, input_data)
                if suggestion:
                    suggestions.append(suggestion)

        return {
            'strengths': strengths,
            'weaknesses': weaknesses,
            'suggestions': suggestions,
        }

    def _get_suggestion(self, feature_name, input_data):
        value = input_data.get(feature_name, '')
        if feature_name == 'education_of_employee':
            if value in ("Master's", "Doctorate"):
                return None
            elif value == "Bachelor's":
                return "A Master's or Doctorate degree would strengthen the application"
            else:
                return "A Bachelor's or higher degree improves approval chances"
        elif feature_name == 'has_job_experience':
            return "Gaining relevant work experience strengthens applications"
        elif feature_name == 'requires_job_training':
            if value == "Y":
                return "Applicants not requiring training have higher approval rates"
            return None
        elif feature_name == 'prevailing_wage':
            return "Higher-paying positions correlate with better approval odds"
        elif feature_name == 'no_of_employees':
            return "Larger employers tend to have smoother PERM processes"
        elif feature_name == 'full_time_position':
            if value != "Y":
                return "Full-time positions demonstrate stronger employer commitment"
            return None
        elif feature_name == 'company_age':
            return "More established companies have stronger approval track records"
        return None

    def _format_message(self, feature_name, shap_val, input_data):
        intensity = self._get_intensity(abs(shap_val))
        value = input_data.get(feature_name, '')
        direction = 'favors' if shap_val < 0 else 'works against'

        if feature_name == 'education_of_employee':
            return f"{value} education {intensity} {direction} approval"
        elif feature_name == 'has_job_experience':
            exp = "Having" if value == "Y" else "Not having"
            return f"{exp} job experience {intensity} {direction} approval"
        elif feature_name == 'requires_job_training':
            trn = "Requiring" if value == "Y" else "Not requiring"
            return f"{trn} job training {intensity} {direction} approval"
        elif feature_name == 'full_time_position':
            pos = "Full-time" if value == "Y" else "Part-time"
            return f"{pos} position {intensity} {direction} approval"
        elif feature_name == 'no_of_employees':
            try:
                return f"Company size ({int(value):,} employees) {intensity} {direction} approval"
            except (ValueError, TypeError):
                return f"Company size {intensity} {direction} approval"
        elif feature_name == 'company_age':
            return f"Company age ({value} years) {intensity} {direction} approval"
        elif feature_name == 'prevailing_wage':
            try:
                return f"Prevailing wage (${float(value):,.0f}) {intensity} {direction} approval"
            except (ValueError, TypeError):
                return f"Prevailing wage {intensity} {direction} approval"
        elif feature_name == 'continent':
            return f"Applicant from {value} {intensity} {direction} approval"
        elif feature_name == 'region_of_employment':
            return f"Employment in {value} {intensity} {direction} approval"
        elif feature_name == 'unit_of_wage':
            return f"Wage unit ({value}) {intensity} {direction} approval"
        else:
            return f"{feature_name} ({value}) {intensity} {direction} approval"

    @staticmethod
    def _get_intensity(abs_shap):
        if abs_shap > 1.0:
            return 'strongly'
        elif abs_shap > 0.3:
            return 'moderately'
        return 'slightly'
