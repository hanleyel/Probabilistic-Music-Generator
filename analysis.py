import pandas as pd

results_df = pd.read_csv('results.csv')
print(results_df['rms_random'].mean())
print(results_df['rms_median'].mean())
print(results_df['rms_corrected_random'].mean())
print(results_df['rms_corrected_median'].mean())
print(results_df['rms_corrected_preds'].mean())