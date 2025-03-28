import numpy as np
import pandas as pd
import os
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

# 定义不同的模型
models = {
    "SVM": make_pipeline(StandardScaler(), SVR(kernel='linear')),
    "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=123),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=123),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=123),
    "KNN": KNeighborsRegressor(n_neighbors=5),  # k=5
    "MLP": MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', solver='adam', max_iter=500, random_state=123)
}

def run_model(model, x, y, p_eids, kfold=5):
    kf = KFold(n_splits=kfold, shuffle=True, random_state=123)
    
    mae_scores = []
    pearson_scores = []
    predicted_y = []
    true_y = []
    samples = []
    
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        samples += list(p_eids[test_index])
        
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        
        predicted_y += list(predictions)
        true_y += list(y_test)
        
        mae = mean_absolute_error(y_test, predictions)
        r, _ = pearsonr(y_test, predictions)
        
        mae_scores.append(mae)
        pearson_scores.append(r)
    
    return {
        'true_y': true_y,
        'predicted_y': predicted_y,
        'mae_scores': mae_scores,
        'pearson_scores': pearson_scores,
        'eids': samples
    }

## 读入数据
#label = pd.read_csv('/share2/pub/zhangyr/zhangyr/myIdea/bioAge/data/UKB/label/self_icd_all_17diseases.csv',index_col=0)
label = pd.read_csv('/share2/pub/zhangyr/zhangyr/myIdea/bioAge/data/UKB/label/health_self_i0.csv')
sex = pd.read_csv('/share2/pub/zhangyr/zhangyr/myIdea/bioAge/data/UKB/body/502137_sex_20241226.csv',header=0,index_col=0)
age = pd.read_csv('/share2/pub/zhangyr/zhangyr/myIdea/bioAge/data/UKB/body/502137_age_20241226.csv',index_col=0)
baseline = pd.read_csv('/share2/pub/zhangyr/zhangyr/myIdea/bioAge/data/UKB/body/502137_baseline_78features_20241226.csv',header=0,index_col=0)

body = pd.read_csv('/share/pub/zhangyr/database/UKB-old//body_fileID_20241216.csv',header=0)
grouped_dict = body.groupby('Organ')['Filed ID'].apply(list).to_dict()
organ_dic = {}
for k in grouped_dict:
	if k != 'Body':
		organ_dic[k] = ['participant.p'+str(x)+'_i0' for x in grouped_dict[k]]

organ_dic['Pulmonary'] = organ_dic['Pulmonary']+['FEV1-FVC_ratio']
organ_dic['Musculoskeletal'] = organ_dic['Musculoskeletal']+['Waist-hip_ratio','BMD_avg',
                                                             'Ankle_spacing_width_avg',
                                                             'Hand_grip_strength_avg']
organ_results={}

for k in organ_dic:
	print(f"Running organ {k}...")
	#health_data = baseline.loc[label.index[label.sum(1) == 0]]
	health_data = baseline.loc[label['eid']]
	health_organ_data = health_data[set(health_data.columns) & set(organ_dic[k])]
	health_organ_data = health_organ_data.dropna()
	print(health_organ_data.shape)

	## organ features
	test_age = age.loc[health_organ_data.index]
	test_sex = sex.loc[health_organ_data.index]

	#female_x = health_organ_data[test_sex.values == 0].values
	#female_y = test_age[test_sex.values == 0].values.ravel()
	#female_eids = health_organ_data.index[test_sex.values.ravel() == 0]

	#male_x = health_organ_data[test_sex.values == 1].values
	#male_y = test_age[test_sex.values == 1].values.ravel()
	#male_eids = health_organ_data.index[test_sex.values.ravel() == 1]
	x = health_organ_data.values
	y = test_age.values.ravel()
	eids = health_organ_data.index

	# 存储所有模型的结果
	results = {}

	# 依次运行所有模型
	for model_name, model in models.items():
		print(f"Running {model_name}...")
		#female_res = run_model(model, female_x, female_y, female_eids)
		#male_res = run_model(model, male_x, male_y, male_eids)
		#results[model_name] = {'female':female_res,'male':male_res}
		results[model_name] = run_model(model, x, y, eids)
	organ_results[k] = results
	#np.save("/share2/pub/zhangyr/zhangyr/myIdea/bioAge/results/body/organ_"+str(k)+"_features_6models_results.npy", results)

	np.save("/share2/pub/zhangyr/zhangyr/myIdea/bioAge/results/body/202503_allgender_organ_"+str(k)+"_features_6models_results.npy", results)

# 保存结果到 .npy 文件
#np.save("/share2/pub/zhangyr/zhangyr/myIdea/bioAge/results/body/organ_features_6models_results.npy", organ_results)
np.save("/share2/pub/zhangyr/zhangyr/myIdea/bioAge/results/body/202503_allgender_organ_features_6models_results.npy", organ_results)
print("All results saved to organ_features_6models_results.npy")

