import os
import sys
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anomaly_detector import PseSCAnomalyDetector
from models.data_loader import load_ecg_for_anomaly_detection, convert_to_list


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'Datasets', 'ECGFiveDays')
    train_path = os.path.join(data_dir, 'ECGFiveDays_TRAIN')
    test_path = os.path.join(data_dir, 'ECGFiveDays_TEST')

    X_normal, X_test, y_test = load_ecg_for_anomaly_detection(
        train_path, test_path, normal_label=1
    )
    X_normal_list = convert_to_list(X_normal)
    X_test_list = convert_to_list(X_test)

    methods = ['iforest', 'svm', 'gaussian']
    results = []

    for method in methods:
        print('\n' + '=' * 60)
        print(f'Running method: {method}')
        print('=' * 60)

        detector = PseSCAnomalyDetector(
            n_shapelets_per_class=3,
            beta_mode='adaptive',
            contamination=0.1,
            method=method,
        )
        detector.fit(X_normal_list)

        predictions = detector.predict(X_test_list)
        predictions_binary = (predictions == 1).astype(int)
        accuracy = float((predictions_binary == y_test).mean())

        scores = detector.anomaly_score(X_test_list)
        anomaly_auc = float(roc_auc_score((1 - y_test), -scores))

        row = {
            'method': method,
            'accuracy': accuracy,
            'auc': anomaly_auc,
        }
        results.append(row)
        print(f"Result -> accuracy: {accuracy:.4f}, auc: {anomaly_auc:.4f}")

    print('\nFINAL RESULTS')
    for row in results:
        print(f"{row['method']}: accuracy={row['accuracy']:.4f}, auc={row['auc']:.4f}")


if __name__ == '__main__':
    main()
