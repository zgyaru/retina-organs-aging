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
    "KNN": KNeighborsRegressor(n_neighbors=5),
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
brain = pd.read_csv('/share2/pub/zhangyr/zhangyr/myIdea/bioAge/data/UKB/brain/502137_baseline_803features_20241226.csv',index_col=0)
#health_brain = brain.loc[label.index[label.sum(1) == 0]].dropna()
health_brain = brain.loc[label['eid']].dropna()
print(health_brain.shape)
## new age
f = 'Xulab_11.csv'
folder = '/share/pub/zhangyr/database/UKB/output/'
data = pd.read_csv(os.path.join(folder, f),index_col=0)
pattern = 'participant\\.p21003_.+'
cols_bollen = data.columns.str.contains(pattern, regex=True)
age_time = data.loc[:,cols_bollen]


test_age = age_time['participant.p21003_i2'].loc[health_brain.index]
test_sex = sex.loc[health_brain.index]
print(test_age.head())
print(test_sex.head())

# female_x = health_brain[test_sex.values == 0].values
# female_y = test_age[test_sex.values.ravel() == 0].values.ravel()
# female_eids = health_brain.index[test_sex.values.ravel() == 0]

# male_x = health_brain[test_sex.values == 1].values
# male_y = test_age[test_sex.values.ravel() == 1].values.ravel()
# male_eids = health_brain.index[test_sex.values.ravel() == 1]

x = health_brain.values
y = test_age.values.ravel()
eids = health_brain.index


# 存储所有模型的结果
results = {}

# 依次运行所有模型
for model_name, model in models.items():
    print(f"Running {model_name}...")
    #female_res = run_model(model, female_x, female_y, female_eids)
    #male_res = run_model(model, male_x, male_y, male_eids)
    #results[model_name] = {'female':female_res,'male':male_res}
    results[model_name] = run_model(model, x, y, eids)

# 保存结果到 .npy 文件
np.save("/share2/pub/zhangyr/zhangyr/myIdea/bioAge/results/brain/202503_803features_6models_results.npy", results)
print("All results saved to model_results.npy")

