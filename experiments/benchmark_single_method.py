import os
import sys
import io
import contextlib
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anomaly_detector import PseSCAnomalyDetector
from models.data_loader import load_ecg_for_anomaly_detection, convert_to_list


def main(method: str):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'Datasets', 'ECGFiveDays')
    train_path = os.path.join(data_dir, 'ECGFiveDays_TRAIN')
    test_path = os.path.join(data_dir, 'ECGFiveDays_TEST')

    X_normal, X_test, y_test = load_ecg_for_anomaly_detection(
        train_path, test_path, normal_label=1
    )
    X_normal_list = convert_to_list(X_normal)
    X_test_list = convert_to_list(X_test)

    detector = PseSCAnomalyDetector(
        n_shapelets_per_class=3,
        beta_mode='adaptive',
        contamination=0.1,
        method=method,
    )

    # Suppress verbose internal logs so terminal output is short and complete.
    with contextlib.redirect_stdout(io.StringIO()):
        detector.fit(X_normal_list)
        predictions = detector.predict(X_test_list)
        scores = detector.anomaly_score(X_test_list)

    pred_bin = (predictions == 1).astype(int)
    accuracy = float((pred_bin == y_test).mean())
    auc = float(roc_auc_score((1 - y_test), -scores))

    print(f"method={method}, accuracy={accuracy:.4f}, auc={auc:.4f}")


if __name__ == '__main__':
    m = sys.argv[1] if len(sys.argv) > 1 else 'iforest'
    main(m)
