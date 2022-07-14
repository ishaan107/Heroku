import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('/Users/ishaan/Downloads/MLProject-Optimal_Flight_Time-master/Backend/data.csv')
pd.set_option('display.max_columns', None)
data = data.drop_duplicates(ignore_index = True)
data.loc[data['Price']>100000].shape[0] /len(data) *100
percentile_995 = np.quantile(data['Price'],0.995)
data.loc[data['Price']>percentile_995,'Price'] = percentile_995
data['dept_airline'] = data['Airline'].str.strip().str.split(',').str[0]
data['arrival_airline'] = np.where(data['Airline'].str.strip().str.split(',').str[1].isnull(), data['Airline'].str.strip().str.split(',').str[0], data['Airline'].str.strip().str.split(',').str[1].str.strip())
data.drop(labels='Airline', axis=1, inplace=True)
labelencoder = LabelEncoder()
data['dept_airline'] = labelencoder.fit_transform(data['dept_airline'])
pd.DataFrame(zip(labelencoder.classes_,range(len(labelencoder.classes_)))).to_csv('dept_airline_map.csv')
data['arrival_airline'] = labelencoder.fit_transform(data['arrival_airline'])
pd.DataFrame(zip(labelencoder.classes_,range(len(labelencoder.classes_)))).to_csv('arrival_airline_map.csv')
labelencoder = LabelEncoder()
data['Dept_city'] = labelencoder.fit_transform(data['Dept_city'])
pd.DataFrame(zip(labelencoder.classes_,range(len(labelencoder.classes_)))).to_csv('dept_city_map.csv')
data['arrival_city'] = labelencoder.fit_transform(data['arrival_city'])
pd.DataFrame(zip(labelencoder.classes_,range(len(labelencoder.classes_)))).to_csv('arrival_city_map.csv')
# onehotencoding preferrable
# cabin , Dept_flights_time 
cabin = pd.get_dummies(data['Cabin'])
dept_flights_time  = pd.get_dummies(data['Dept_flights_time'])
data = pd.concat([data,cabin,dept_flights_time],axis=1)
data.drop(['Cabin','Dept_flights_time'],axis=1, inplace = True )
data['departure_minute'] = pd.to_datetime(data['departure_time']).dt.minute / 60
data['arrival_hour'] = pd.to_datetime(data['arrival_time']).dt.hour 
data['arrival_minute'] = pd.to_datetime(data['arrival_time']).dt.minute / 60
data.drop(['departure_time','arrival_time'],inplace =True ,axis=1)
x= data
x= x.drop(['optimal_hours'],axis=1)
y = data['optimal_hours']
# feature scaling
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
x = pd.DataFrame(std.fit_transform(x), columns=x.columns)
data = pd.concat([x,y] , axis= 1)

# spliting the data in train test split
from sklearn.model_selection import train_test_split
train,val  = train_test_split(data, test_size=0.20, random_state=42)
train_feature = train.drop(['optimal_hours'],axis=1)
train_target = train['optimal_hours']
val_feature = val.drop(['optimal_hours'],axis=1)
val_target = val['optimal_hours']
from sklearn.feature_selection import f_regression
f_score, p_value = f_regression(train_feature, train_target)
data_dict = {
    'f_score': f_score,
    'p_value': p_value
}
feature_imp_df = pd.DataFrame(data_dict, index = train_feature.columns)
feature_imp_df.sort_values(by='f_score', ascending=False)
feature_imp_list = feature_imp_df[feature_imp_df.sort_values(by='f_score', ascending=False)['f_score']>20].index.tolist()
feature_imp_list.append('Price')
train_feature = train_feature[feature_imp_list]
val_feature = val_feature[feature_imp_list]
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(min_samples_leaf=6)
model.fit(train_feature, train_target)
