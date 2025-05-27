import pandas as pd
import xgboost as xgb
import joblib
import logging
import os
from pathlib import Path
# --- Remove plot_utils import if no longer needed elsewhere ---
# from .plot_utils import plot_confusion_matrix
from sklearn.metrics import accuracy_score, classification_report # Keep classification_report
from .config import settings, BASE_DIR
from .preprocessing import preprocess_and_split
from .database import load_data_from_mongo
from .prediction import reload_model_and_preprocessors
from datetime import datetime
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(task_id: str, hyperparams: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Loads data, trains, evaluates (returning metrics dict), saves artifacts, reloads,
    and returns training results.

    Args:
        task_id: Unique identifier for this training task.
        hyperparams: Optional dictionary of hyperparameters.

    Returns:
        A dictionary containing training results:
        {
            "status": "completed" or "failed",
            "message": "Success message or error description",
            "accuracy": float or None,
            "precision_macro": float or None,
            "recall_macro": float or None,
            "f1_score_macro": float or None,
            "classification_report_dict": dict or None,
            "params_used": dict,
            "start_time": datetime,
            "end_time": datetime or None
        }
    """
    start_time = datetime.now()
    if hyperparams is None:
        hyperparams = {}

    # Initialize results dictionary with new metric fields
    results = {
        "status": "failed",
        "message": "Training started but did not complete.",
        "accuracy": None,
        "precision_macro": None,
        "recall_macro": None,
        "f1_score_macro": None,
        "classification_report_dict": None,
        "params_used": {},
        "start_time": start_time,
        "end_time": None
    }

    logger.info(f"--- Starting Retraining Process (Task ID: {task_id}) ---")

    # --- Steps 1 & 2: Load and Prepare Data ---
    # (Keep this section as it was)
    try:
        logger.info("Loading training data from MongoDB...")
        df_train_raw = load_data_from_mongo()
        if df_train_raw is None or df_train_raw.empty:
            results["message"] = "No data loaded from MongoDB. Aborting training."
            logger.error(results["message"])
            results["end_time"] = datetime.now()
            return results
        logger.info(f"Loaded {len(df_train_raw)} records from MongoDB.")

        logger.info("Preparing data for training...")
        prep_results = preprocess_and_split(df_train_raw, fit_scalers_encoders=True)
        if prep_results is None or prep_results[0] is None:
            results["message"] = "Data preparation failed. Aborting training."
            logger.error(results["message"])
            results["end_time"] = datetime.now()
            return results

        X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, label_encoder = prep_results
        logger.info("Data preparation successful.")
        if not label_encoder or not hasattr(label_encoder, 'classes_'):
             results["message"] = "Label encoder not properly created during preprocessing."
             logger.error(results["message"])
             results["end_time"] = datetime.now()
             return results
    except Exception as e:
        results["message"] = f"Error during data loading/preparation: {e}"
        logger.exception(results["message"])
        results["end_time"] = datetime.now()
        return results

    # --- Step 3: Configure and Train Model ---
    # (Keep this section as it was)
    try:
        default_params = {
            'objective': 'multi:softmax', 'eval_metric': 'mlogloss', 'use_label_encoder': False,
            'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 8, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'gamma': 0, 'min_child_weight': 1, 'reg_lambda': 1,
            'reg_alpha': 0, 'booster': 'gbtree', 'sampling_method': 'uniform',
            'tree_method': 'auto', 'random_state': 42, 'n_jobs': -1
        }
        # ... (parameter handling logic remains the same) ...
        final_params = default_params.copy()
        valid_xgb_params = xgb.XGBClassifier().get_params().keys()
        if hyperparams:
            for key, value in hyperparams.items():
                actual_key = key
                if key == 'lambda': actual_key = 'reg_lambda'
                if key == 'alpha': actual_key = 'reg_alpha'
                if actual_key in valid_xgb_params and value is not None:
                    final_params[actual_key] = value
                else:
                    logger.warning(f"Ignoring invalid or None hyperparameter: {key}={value}")

        final_params['num_class'] = len(label_encoder.classes_)
        results["params_used"] = final_params # Store actual params used

        logger.info(f"Training XGBoost model (Task ID: {task_id}) with parameters: {final_params}")
        model = xgb.XGBClassifier(**final_params)

        model.fit(X_train_scaled, y_train_encoded,
                  eval_set=[(X_test_scaled, y_test_encoded)],
                  # early_stopping_rounds=15,
                  verbose=False)
        logger.info("Model training complete.")
    except Exception as e:
        results["message"] = f"Error during XGBoost model training: {e}"
        logger.exception(results["message"])
        results["end_time"] = datetime.now()
        return results


    # --- Step 4: Evaluate Model (Modified) ---
    try:
        logger.info("Evaluating model on test set...")
        y_pred_encoded = model.predict(X_test_scaled)
        target_names = label_encoder.classes_.tolist()

        # Get classification report as a dictionary
        report_dict = classification_report(
            y_test_encoded,
            y_pred_encoded,
            target_names=target_names,
            zero_division=0,
            output_dict=True # <-- Get dict output
        )

        # Extract overall accuracy and macro averages
        results["accuracy"] = report_dict.get('accuracy') # Overall accuracy
        if 'macro avg' in report_dict:
            results["precision_macro"] = report_dict['macro avg'].get('precision')
            results["recall_macro"] = report_dict['macro avg'].get('recall')
            results["f1_score_macro"] = report_dict['macro avg'].get('f1-score')

        results["classification_report_dict"] = report_dict # Store the whole dict

        logger.info(f"Test Set Accuracy: {results['accuracy']:.4f}" if results['accuracy'] is not None else "Accuracy not available")
        logger.info(f"Macro Avg Precision: {results['precision_macro']:.4f}" if results['precision_macro'] is not None else "Macro Precision not available")
        logger.info(f"Macro Avg Recall: {results['recall_macro']:.4f}" if results['recall_macro'] is not None else "Macro Recall not available")
        logger.info(f"Macro Avg F1-Score: {results['f1_score_macro']:.4f}" if results['f1_score_macro'] is not None else "Macro F1-Score not available")
        logger.debug(f"Full Classification Report Dict: {report_dict}")

        # --- Remove Confusion Matrix Plotting ---
        # cm_filename = f"cm_plot_{task_id}.png"
        # cm_output_path = CM_PLOT_DIR / cm_filename
        # try:
        #     y_test_labels = label_encoder.inverse_transform(y_test_encoded)
        #     y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
        #     plot_confusion_matrix(y_test_labels, y_pred_labels, classes=label_encoder.classes_,
        #                           normalize=True, title=f'Normalized Confusion Matrix (Task {task_id[:8]})',
        #                           output_path=cm_output_path)
        #     # results["confusion_matrix_filename"] = cm_filename # Store filename
        #     logger.info(f"Confusion matrix plot saved for task {task_id}.")
        # except Exception as plot_e:
        #      logger.warning(f"Could not generate/save confusion matrix plot: {plot_e}")
             # results["message"] += " (Warning: CM plot failed)"

    except Exception as e:
        results["message"] = f"Error during evaluation: {e}."
        logger.exception(results["message"])
        # Metrics will remain None

    # --- Step 5: Save Model ---
    # (Keep this section as it was)
    try:
        logger.info("Saving trained model...")
        temp_model_path = Path(str(settings.MODEL_PATH) + ".tmp")
        joblib.dump(model, temp_model_path)
        os.replace(temp_model_path, settings.MODEL_PATH)
        logger.info(f"Model saved successfully to {settings.MODEL_PATH}.")
    except Exception as e:
        results["status"] = "failed"
        results["message"] = f"Training completed, but failed to save model: {e}"
        logger.exception(results["message"])
        if temp_model_path.exists(): temp_model_path.unlink()
        results["end_time"] = datetime.now()
        return results

    # --- Step 6: Reload Model/Preprocessors ---
    # (Keep this section as it was)
    try:
        logger.info("Attempting to reload model and preprocessors for prediction service...")
        reload_model_and_preprocessors()
    except Exception as e:
         logger.warning(f"Retraining successful, but failed to hot-reload model/preprocessors: {e}")
         results["message"] += " (Warning: Hot-reload failed)"

    # --- Final Success ---
    # Only mark completed if saving succeeded
    if results["status"] != "failed":
         results["status"] = "completed"
         results["message"] = "Retraining process finished successfully."
         logger.info(f"--- Retraining Process Finished (Task ID: {task_id}) ---")

    results["end_time"] = datetime.now()
    return results

########################################################################################################
def incremental_train_model(today: datetime):
    """ Incremental training bằng dữ liệu mới MongoDB """
    logger.info(f"🔄 Incremental training với dữ liệu ngày {today.strftime('%Y-%m-%d')}...")

    df = load_data_from_mongo()
    if df.empty:
        logger.warning("⚠️ MongoDB không có dữ liệu.")
        return

    df['date_only'] = df['datetime'].dt.date
    today_date = today.date()
    df_today = df[df['date_only'] == today_date]

    if df_today.empty:
        logger.warning(f"⚠️ Không có dữ liệu cho ngày {today.strftime('%Y-%m-%d')}.")
        return

    results = preprocess_and_split(df_today, fit_scalers_encoders=False)
    if results is None or results[0] is None:
        logger.error("⚠️ Preprocessing thất bại.")
        return

    X_train_scaled, _, y_train_encoded, _, _ = results

    from core.prediction import model
    if model is None:
        logger.error("⚠️ Model chưa load.")
        return

    try:
        model.fit(X_train_scaled, y_train_encoded, xgb_model=model.get_booster())
        logger.info("✅ Incremental training thành công.")

        temp_model_path = Path(str(settings.MODEL_PATH) + ".tmp")
        joblib.dump(model, temp_model_path)
        os.replace(temp_model_path, settings.MODEL_PATH)
        logger.info(f"✅ Model incremental đã lưu vào {settings.MODEL_PATH}")

        reload_model_and_preprocessors()
    except Exception as e:
        logger.error(f"Lỗi incremental training: {e}")

def full_train_from_csv():
    """ Retrain toàn bộ model từ base_weather.csv """
    logger.info("🔄 Full retrain từ base_weather.csv...")

    base_csv_path = settings.BASE_DIR / "data" / "base_weather.csv"
    if not base_csv_path.exists():
        logger.error("⚠️ Base CSV không tồn tại.")
        return

    df = pd.read_csv(base_csv_path)
    if df.empty:
        logger.warning("⚠️ Base CSV trống.")
        return

    results = preprocess_and_split(df, fit_scalers_encoders=True)
    if results is None or results[0] is None:
        logger.error("⚠️ Preprocessing thất bại.")
        return

    X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, label_encoder = results

    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(label_encoder.classes_),
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_estimators=150,
        learning_rate=0.1,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train_encoded, eval_set=[(X_test_scaled, y_test_encoded)], verbose=False)

    temp_model_path = Path(str(settings.MODEL_PATH) + ".tmp")
    joblib.dump(model, temp_model_path)
    os.replace(temp_model_path, settings.MODEL_PATH)

    logger.info(f"✅ Full retrain model saved vào {settings.MODEL_PATH}")
    reload_model_and_preprocessors()
