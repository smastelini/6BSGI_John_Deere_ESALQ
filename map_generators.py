import pickle
import time

import pandas as pd

from feat_eng import MyPipeline


with open('models/pipeline_A.mdl', 'rb') as f:
    pipeline_A = pickle.load(f)

with open('models/pipeline_B.mdl', 'rb') as f:
    pipeline_B = pickle.load(f)

# Malha A
print('Malha A 1024x1024')
malha_a = pd.read_csv('malhas/mapa-A-1024x1024.csv')

start = time.time()
preds_l, preds_rf, preds_u = pipeline_A.run(malha_a, 'quantile')
print('MALHA A + Quantile (em segundos): ', time.time() - start)

malha_a['preds_gbt_05'] = preds_l
malha_a['preds_rf'] = preds_rf
malha_a['preds_gbt_95'] = preds_u

start = time.time()
preds_gpr, preds_std = pipeline_A.run(malha_a, 'gaussian_process')
print('MALHA A + GPR (em segundos): ', time.time() - start)

malha_a['preds_gpr_05'] = preds_gpr - 1.9600 * preds_std
malha_a['preds_gpr'] = preds_gpr
malha_a['preds_gpr_95'] = preds_gpr + 1.9600 * preds_std

malha_a.to_csv('predictions/malha_A_1024x1024.csv', index=False)


# Malha A
print('Malha B 1024x1024')
malha_b = pd.read_csv('malhas/mapa-B-1024x1024.csv')

start = time.time()
preds_l, preds_rf, preds_u = pipeline_B.run(malha_b, 'quantile')
print('MALHA B + Quantile (em segundos): ', time.time() - start)

malha_b['preds_gbt_05'] = preds_l
malha_b['preds_rf'] = preds_rf
malha_b['preds_gbt_95'] = preds_u

start = time.time()
preds_gpr, preds_std = pipeline_B.run(malha_b, 'gaussian_process')
print('MALHA B + GPR (em segundos): ', time.time() - start)

malha_a['preds_gpr_05'] = preds_gpr - 1.9600 * preds_std
malha_a['preds_gpr'] = preds_gpr
malha_a['preds_gpr_95'] = preds_gpr + 1.9600 * preds_std

malha_b.to_csv('predictions/malha_B_1024x1024.csv', index=False)
