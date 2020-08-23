import pandas as pd
import xgboost as xgb
import wandb
from wandb import xgboost 

xg_param = {
    'eta' : 0.01,
    'max_depth': 30,
    'min_child_weight': 7,
    'subsample': 0.7,
    'colsample_bytree':0.8,
    'gamma': 3,
    # 'n_estimators': 10000,
    'learning_rate' : 0.01
}

xg_wandb = wandb.init(project='g-yes_xgboost_FBT', config= xg_param)
config = xg_wandb.config


data = pd.read_excel(r'C:\Users\신충현\Desktop\빅데이터 청년인재 파일\project g-yes\PP복합수지 레시피별 물성 종합_200729.xlsx',sheet_name = '수정용',encoding= 'cp949',header = 1)

data_input = data.drop(['No.(원)','No.(변환)', '비중','굴곡강도','HDT','IZOD','MI','인장강도'], axis=1)
data_input = data_input.dropna(how = 'any')

x_data_fbt = data_input.drop(['굴곡탄성률'],axis=1)
y_data_fbt= data_input['굴곡탄성률']

from sklearn.model_selection import train_test_split
X_train_fbt, X_test_fbt, y_train_fbt, y_test_fbt = train_test_split(x_data_fbt, y_data_fbt, test_size = 0.2, random_state=11)

xg_reg_fbt = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = config.colsample_bytree, learning_rate = config.learning_rate, eta= config.eta,
                max_depth = config.max_depth, gamma = config.gamma, n_estimators = 100000, subsample = config.subsample)
#  alpha = 10

xg_reg_fbt.fit(X_train_fbt,y_train_fbt, eval_set = [(X_test_fbt, y_test_fbt)], early_stopping_rounds = 5, callbacks=[wandb.xgboost.wandb_callback()], verbose = False)