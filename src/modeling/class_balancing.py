from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, 
    confusion_matrix, roc_auc_score, f1_score
)
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import TomekLinks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from src import config as cfg

train_df = pd.read_csv(r"E:\Data_Science_Project\collision-severity-predictor\data\processed\train.csv")
val_df   = pd.read_csv(r"E:\Data_Science_Project\collision-severity-predictor\data\processed\val.csv")
TARGET = cfg.TARGET_COL
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test  = val_df.drop(columns=[TARGET])
y_test  = val_df[TARGET]
print("=" * 60)
print("STEP 1: Data Cleaning with Tomek Links")
print("=" * 60)

tomek = TomekLinks()
X_clean, y_clean = tomek.fit_resample(X_train, y_train)

print(f"\nOriginal training set distribution:")
print(pd.Series(y_train).value_counts().sort_index())
print(f"\nAfter Tomek Links:")
print(pd.Series(y_clean).value_counts().sort_index())


resampling_methods = {
    'No Resampling (Baseline)': None,
    'SMOTE': SMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42),
    'Borderline-SMOTE': BorderlineSMOTE(random_state=42, kind='borderline-1'),
    'SMOTE-ENN': SMOTEENN(random_state=42)
}


rf_params = {
    'n_estimators': 200,
    'max_depth': None,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}


results = []
confusion_matrices = {}
feature_importances = {}


print("\n" + "=" * 60)
print("STEP 2: Training and Evaluating Each Method")
print("=" * 60)

for method_name, resampler in resampling_methods.items():
    print(f"\n{'=' * 60}")
    print(f"Method: {method_name}")
    print(f"{'=' * 60}")
    
    start_time = time()
    
 
    if resampler is None:
        X_resampled, y_resampled = X_clean, y_clean
    else:
        try:
            X_resampled, y_resampled = resampler.fit_resample(X_clean, y_clean)
        except Exception as e:
            print(f"Error with {method_name}: {e}")
            continue
    
   
    print(f"\nClass distribution after resampling:")
    print(pd.Series(y_resampled).value_counts().sort_index())
    
    
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_resampled, y_resampled)
    

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)
    
    
    accuracy = accuracy_score(y_test, y_pred)
    
   
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
   
    f1_scores = {}
    for class_label in np.unique(y_train):
        if str(class_label) in class_report:
            f1_scores[f'f1_class_{class_label}'] = class_report[str(class_label)]['f1-score']
    
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
  
    n_classes = len(rf.classes_)
    if n_classes == 2:
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
    
  
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[method_name] = cm
    

    feature_importances[method_name] = pd.Series(
        rf.feature_importances_, 
        index=X_train.columns
    ).sort_values(ascending=False)
    
    training_time = time() - start_time
    

    result_dict = {
        'Method': method_name,
        'Accuracy': accuracy,
        'F1-Macro': f1_macro,
        'F1-Weighted': f1_weighted,
        'ROC-AUC': roc_auc,
        'Training Time (s)': training_time,
        **f1_scores
    }
    results.append(result_dict)
    
    # Print metrics
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1-Macro: {f1_macro:.4f}")
    print(f"F1-Weighted: {f1_weighted:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Training Time: {training_time:.2f}s")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))


print("\n" + "=" * 60)
print("STEP 3: Comparison Summary")
print("=" * 60)

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# Find best method for each metric
print("\n" + "=" * 60)
print("Best Method for Each Metric:")
print("=" * 60)
for col in ['Accuracy', 'F1-Macro', 'F1-Weighted', 'ROC-AUC']:
    best_idx = results_df[col].idxmax()
    best_method = results_df.loc[best_idx, 'Method']
    best_value = results_df.loc[best_idx, col]
    print(f"{col:15s}: {best_method:25s} ({best_value:.4f})")



fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Performance Comparison Across Resampling Methods', fontsize=16, fontweight='bold')

metrics = ['Accuracy', 'F1-Macro', 'F1-Weighted', 'ROC-AUC']
for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    
    bars = ax.barh(results_df['Method'], results_df[metric])
    
    # Color the best bar
    best_idx = results_df[metric].idxmax()
    bars[best_idx].set_color('green')
    bars[best_idx].set_alpha(0.7)
    
    ax.set_xlabel(metric, fontweight='bold')
    ax.set_title(f'{metric} by Method')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(results_df[metric]):
        ax.text(v, i, f' {v:.4f}', va='center')

plt.tight_layout()
plt.savefig(rf"{cfg.FIGURES_DIR}\SMOTE1")


f1_class_cols = [col for col in results_df.columns if col.startswith('f1_class_')]
if f1_class_cols:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(results_df['Method']))
    width = 0.8 / len(f1_class_cols)
    
    for i, col in enumerate(f1_class_cols):
        class_num = col.split('_')[-1]
        offset = (i - len(f1_class_cols)/2) * width + width/2
        ax.bar(x + offset, results_df[col], width, label=f'Class {class_num}')
    
    ax.set_xlabel('Resampling Method', fontweight='bold')
    ax.set_ylabel('F1-Score', fontweight='bold')
    ax.set_title('Class-wise F1-Score Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Method'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(rf"{cfg.FIGURES_DIR}\SMOTE2")


n_methods = len(confusion_matrices)
fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))

if n_methods == 1:
    axes = [axes]

for idx, (method_name, cm) in enumerate(confusion_matrices.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
    axes[idx].set_title(f'{method_name}', fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig(rf"{cfg.FIGURES_DIR}\SMOTE3")


fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(results_df['Method'], results_df['Training Time (s)'])
ax.set_xlabel('Training Time (seconds)', fontweight='bold')
ax.set_title('Training Time Comparison', fontweight='bold', fontsize=14)
ax.grid(axis='x', alpha=0.3)

for i, v in enumerate(results_df['Training Time (s)']):
    ax.text(v, i, f' {v:.2f}s', va='center')

plt.tight_layout()
plt.savefig(rf"{cfg.FIGURES_DIR}\SMOTE4")


print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)

best_f1_macro_idx = results_df['F1-Macro'].idxmax()
best_f1_macro_method = results_df.loc[best_f1_macro_idx, 'Method']

print(f"\n✓ Best overall method (F1-Macro): {best_f1_macro_method}")
print(f"  This method performs best across all classes on average.")

if f1_class_cols:
    print(f"\n✓ Class-specific performance:")
    for col in f1_class_cols:
        best_idx = results_df[col].idxmax()
        best_method = results_df.loc[best_idx, 'Method']
        best_value = results_df.loc[best_idx, col]
        class_num = col.split('_')[-1]
        print(f"  Class {class_num}: {best_method} (F1={best_value:.4f})")

print(f"\n✓ Save results to CSV:")
results_df.to_csv(rf'{cfg.REPORTS_DIR}\smote_comparison_results.csv', index=False)
print(f"  Results saved to: smote_comparison_results.csv")