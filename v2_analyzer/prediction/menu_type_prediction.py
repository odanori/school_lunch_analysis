import re
from datetime import date, datetime, time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
from pmdarima import arima, model_selection, utils
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

GRP = 'e'
AGG_D = {
    'amount_g': 'sum',
    'carolies_kcal': 'sum',
    'protein_g': 'sum',
    'fat_g': 'sum',
    'sodium_mg': 'sum',
    'menu': set,
}
STD_L = ['amount_g', 'carolies_kcal', 'protein_g', 'fat_g', 'sodium_mg']
SPECIFIED_WORD_LIST = [r'こぎつね|おこわ|たきこみ', r'カレー', r'丼|ライス|そぼろご', r'ごはん', r'パン|フランス', r'そば|うどん|スパゲ|めん|メン']


def prediction(data: pd.DataFrame) -> None:
    menu_type_prediction(data)


def prediction_preprocess(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

    def without_milk(text_set: Tuple[str]) -> List[str]:
        text_l = list(text_set)
        ex_text_l = [tx for tx in text_l if '乳' not in tx]
        return ex_text_l

    def extract_specified_contents(menu_list: List[str]) -> int:
        menu_type = 0
        for type_num, spesified_word in enumerate(SPECIFIED_WORD_LIST):
            ex_list = [m for m in menu_list if re.search(spesified_word, m)]
            if len(ex_list) > 0:
                menu_type = type_num
                break
            else:
                menu_type = len(SPECIFIED_WORD_LIST)
        return menu_type

    def standarization(data: List[float]) -> List[float]:
        n = len(data)
        data_mean = np.mean(data)
        var = sum([(d - data_mean) ** 2 for d in data]) / (n - 1)
        std = var ** 0.5
        z = [(d - data_mean) / std for d in data]
        return z

    cp_data = data.copy()
    # 年度、日にち、地域で集計
    agg_d = AGG_D.copy()
    data_agg = cp_data.groupby(['era', 'date', 'area_group']).agg(agg_d).reset_index()

    # 牛乳を除いたメニュー数計測とメニュー種の推定
    ex_data_agg = data_agg[(data_agg['area_group'] == GRP)]
    ex_data_agg['unix_time'] = [datetime.combine(x, time()).timestamp() for x in ex_data_agg['date'].values]
    ex_data_agg['without_milk'] = ex_data_agg['menu'].apply(without_milk)
    ex_data_agg['menu_counts'] = ex_data_agg['without_milk'].apply(len)
    ex_data_agg['menu_type'] = ex_data_agg['without_milk'].apply(extract_specified_contents)

    # 追加処理(適宜変更)
    # メニュー数のone-hot-encoding
    COL_N = 'menu_counts'
    def custom_combiner(feature, category):
        feature = COL_N
        return f'{feature}_{category}'
    encoder = OneHotEncoder(feature_name_combiner=custom_combiner, handle_unknown='ignore')
    encoder.fit(np.array(ex_data_agg[COL_N].values).reshape(-1, 1))
    # print(encoder.get_feature_names_out())
    encoder_data = encoder.transform(np.array(ex_data_agg[COL_N].values.reshape(-1, 1))).toarray()
    encoder_data = pd.DataFrame(encoder_data, columns=encoder.get_feature_names_out()).reset_index(drop=True)
    ex_data_agg.reset_index(drop=True, inplace=True)
    ex_data_agg = pd.concat([ex_data_agg, encoder_data], axis=1)

    # カレンダー特徴量(ランダムフォレスト検討時に効果なし)
    ex_data_agg['month'] = ex_data_agg['date'].apply(lambda x: x.month)
    ex_data_agg['day'] = ex_data_agg['date'].apply(lambda x: x.day)
    ex_data_agg['weekday'] = ex_data_agg['date'].apply(lambda x: x.weekday())
    COL_N = 'weekday'
    encoder = OneHotEncoder(feature_name_combiner=custom_combiner, handle_unknown='ignore')
    encoder.fit(np.array(ex_data_agg[COL_N].values).reshape(-1, 1))
    # print(encoder.get_feature_names_out())
    encoder_data = encoder.transform(np.array(ex_data_agg[COL_N].values.reshape(-1, 1))).toarray()
    encoder_data = pd.DataFrame(encoder_data, columns=encoder.get_feature_names_out()).reset_index(drop=True)
    ex_data_agg.reset_index(drop=True, inplace=True)
    ex_data_agg = pd.concat([ex_data_agg, encoder_data], axis=1)

    # ラグ特徴量、ローリング特徴量の追加(ランダムフォレスト検討時に効果なし)
    # std_l = STD_L
    # lag = 2
    # lag_cols = [f'{col}_lag{lag}' for col in std_l]
    # ex_data_agg[lag_cols] = ex_data_agg[std_l].shift(lag)
    # # rolling_cols = [f'{col}_rolling{lag}' for col in STD_L]
    # # ex_data_agg[rolling_cols] = ex_data_agg[STD_L].shift(lag).rolling(window=lag).mean()
    # # ex_data_agg.dropna(inplace=True)
    # lag = 3
    # lag_cols = [f'{col}_lag{lag}' for col in std_l]
    # ex_data_agg[lag_cols] = ex_data_agg[std_l].shift(lag)
    # # ex_data_agg.dropna(inplace=True)

    ex_data_agg.drop(columns=['menu', 'without_milk'], inplace=True)

    base_data = ex_data_agg[(ex_data_agg['date'] <= date(2023, 12, 31))].copy()
    predict_expect_data = ex_data_agg[(ex_data_agg['date'] > date(2023, 12, 31))].copy()

    # base_data[STD_L] = base_data[STD_L].apply(standarization, axis=0)
    # predict_expect_data[STD_L] = predict_expect_data[STD_L].apply(standarization, axis=0)

    return base_data, predict_expect_data


def prediction_menu(base_data: pd.DataFrame, predict_expect_data: pd.DataFrame) -> None:
    features = ['amount_g', 'carolies_kcal', 'protein_g', 'fat_g', 'sodium_mg', 'menu_counts',
                'amount_g_lag2', 'carolies_kcal_lag2', 'protein_g_lag2', 'fat_g_lag2', 'sodium_mg_lag2',
                'amount_g_lag3', 'carolies_kcal_lag3', 'protein_g_lag3', 'fat_g_lag3', 'sodium_mg_lag3',
                # 'amount_g_rolling2', 'carolies_kcal_rolling2', 'protein_g_rolling2', 'fat_g_rolling2', 'sodium_mg_rolling2',
                ]  # 'menu_counts', 'unix_time',
    target = 'menu_type'

    x = base_data[features]
    y = base_data[target]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2024)
    forest = RandomForestRegressor(n_estimators=1000, criterion='squared_error', random_state=100, n_jobs=-1)
    forest.fit(X_train, y_train)

    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    print(f'MAE train: {mse_train:.2f}, test: {mse_test:.2f}')

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    print(f'R2 SCORE train: {r2_train}, test: {r2_test}')


def prediction_menu_arima(base_data: pd.DataFrame, predict_expect_data: pd.DataFrame) -> None:
    # plt.plot(base_data['date'].values, base_data['menu_type'].values)
    # plt.plot(predict_expect_data['date'].values, predict_expect_data['menu_type'].values)

    # plt.show()
    # utils.plot_acf(base_data['menu_type'].values, alpha=0.05)
    # utils.plot_pacf(base_data['menu_type'].values, alpha=0.05)
    # print('d= ', arima.ndiffs(base_data['menu_type'].values))
    # print('D= ', arima.nsdiffs(base_data['menu_type'].values, m=2))
    train_size = len(base_data) * 7 // 10
    print(train_size, len(base_data))
    train, test = model_selection.train_test_split(base_data['menu_type'].values, train_size=train_size)
    arima_model = pm.auto_arima(train,
                                seasonal=True,
                                m=12,
                                trace=True,
                                n_jobs=-1,
                                maxiter=10)

    preds, conf_int = arima_model.predict(n_periods=test.shape[0],
                                          return_conf_int=True)
    print('MAE: ', mean_absolute_error(test, preds))
    # print('MAPE(%): ', np.mean(abs(test - preds)/test) * 100)






def menu_type_prediction(data: pd.DataFrame) -> None:
    base_data, predict_expect_data = prediction_preprocess(data)
    # prediction_menu(base_data, predict_expect_data)
    prediction_menu_arima(base_data, predict_expect_data)
