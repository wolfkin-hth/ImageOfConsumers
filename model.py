# -*- coding: utf-8 -*-
# model
# author = 'huangth'

import os
import time

import numpy as np
import pandas as pd

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error

BASE_PATH = '/home/wolfkin/ImageOfConsumers/'
ETL_DATA_PATH = os.path.join(BASE_PATH, "EtlData")


def get_feature(name):
    data_name = os.path.join(ETL_DATA_PATH, "{}.csv".format(name))
    df = pd.read_csv(data_name)
    return df


def lgb_mae_model(train_df, test_df, params):
    NFOLDS = 10
    train_label = train_df['信用分']
    kfold = KFold(n_splits=NFOLDS, shuffle=False, random_state=2019)
    kf = kfold.split(train_df, train_label)

    train = train_df.drop(['用户编码', '信用分'], axis=1)
    test = test_df.drop(['用户编码'], axis=1)

    cv_pred = np.zeros(test.shape[0])
    valid_best_l2_all = 0

    count = 0
    for i, (train_fold, validate) in enumerate(kf):
        print("model: lgb_mae. fold: ", i , "training...")
        X_train, label_train = train.iloc[train_fold], train_label.iloc[train_fold]
        X_validate, label_validate = train.iloc[validate], train_label.iloc[validate]

        dtrain = lgb.Dataset(X_train, label_train)
        dvalid = lgb.Dataset(X_validate, label_validate, reference=dtrain)

        bst = lgb.train(params, dtrain, num_boost_round=10000, valid_sets=dvalid, verbose_eval=-1, early_stopping_rounds=50)
        cv_pred += bst.predict(test, num_iteration=bst.best_iteration)
        valid_best_l2_all += bst.best_score['valid_0']['l1']

        count += 1

    cv_pred /= NFOLDS
    valid_best_l2_all /= NFOLDS
    print("lgb_mae cv score for valid is: ", 1/(1+valid_best_l2_all))

    print("----------------------------------------")
    print("----------------------------------------")
    print("lgb_mae  feature importance：")
    fea_importances = pd.DataFrame({
        'column': train.columns,
        'importance': bst.feature_importance(importance_type='split', iteration=bst.best_iteration)
    }).sort_values(by='importance', ascending=False)
    print(fea_importances)
    print("----------------------------------------")
    print("----------------------------------------")

    return cv_pred


def lgb_mse_model(train_df, test_df, params):
    NFOLDS = 10
    train_label = train_df['信用分']
    kfold = KFold(n_splits=NFOLDS, shuffle=False, random_state=2019)
    kf = kfold.split(train_df, train_label)

    train = train_df.drop(['用户编码', '信用分'], axis=1)
    test = test_df.drop(['用户编码'], axis=1)

    cv_pred = np.zeros(test.shape[0])
    valid_best_l2_all = 0

    count = 0
    for i, (train_fold, validate) in enumerate(kf):
        print("model:lgb_mse. fold: ", i , "training...")
        X_train, label_train = train.iloc[train_fold], train_label.iloc[train_fold]
        X_validate, label_validate = train.iloc[validate], train_label.iloc[validate]

        dtrain = lgb.Dataset(X_train, label_train)
        dvalid = lgb.Dataset(X_validate, label_validate, reference=dtrain)

        bst = lgb.train(params, dtrain, num_boost_round=10000, valid_sets=dvalid, verbose_eval=-1, early_stopping_rounds=50)
        cv_pred += bst.predict(test, num_iteration=bst.best_iteration)
        valid_best_l2_all += bst.best_score['valid_0']['l1']

        count += 1

    cv_pred /= NFOLDS
    valid_best_l2_all /= NFOLDS
    print("lgb_mse cv score for valid is: ", 1/(1+valid_best_l2_all))

    print("----------------------------------------")
    print("----------------------------------------")
    print("lgb_mse  feature importance：")
    fea_importances = pd.DataFrame({
        'column': train.columns,
        'importance': bst.feature_importance(importance_type='split', iteration=bst.best_iteration)
    }).sort_values(by='importance', ascending=False)
    print(fea_importances)
    print("----------------------------------------")
    print("----------------------------------------")

    return cv_pred


def xgb_mae_model(train_df, test_df, params):
    NFOLDS = 5
    train_label = train_df['信用分']
    kfold = KFold(n_splits=NFOLDS, shuffle=False, random_state=2019)
    kf = kfold.split(train_df, train_label)

    train = train_df.drop(['用户编码', '信用分'], axis=1)
    test = test_df.drop(['用户编码'], axis=1)

    cv_pred = np.zeros(test.shape[0])

    count = 0
    preds_list = list()
    oof = np.zeros(train_df.shape[0])
    for i, (train_fold, validate) in enumerate(kf):
        print("model: xgb_mae. fold: ", i , "training...")
        X_train, label_train = train.iloc[train_fold], train_label.iloc[train_fold]
        X_validate, label_validate = train.iloc[validate], train_label.iloc[validate]

        gbm = xgb.XGBRegressor(**params)
        bst = gbm.fit(X_train, label_train, eval_set=[(X_train, label_train), (X_validate, label_validate)],
                          early_stopping_rounds=200, verbose=500)

        k_pred = bst.predict(X_validate)
        oof[validate] = k_pred

        preds = gbm.predict(test)
        preds_list.append(preds)

        count += 1

    fold_mae_error = mean_absolute_error(train_label, oof)

    preds_columns = ['preds_{id}'.format(id=i) for i in range(NFOLDS)]
    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds_list = list(preds_df.mean(axis=1))
    cv_pred = preds_list

    print("xgb_mae cv score for valid is: ", 1/(1+fold_mae_error))

    # print("----------------------------------------")
    # print("----------------------------------------")
    # print("xgb_mae  feature importance：")
    # fea_importances = pd.DataFrame({
    #     'column': train.columns,
    #     'importance': bst.feature_importance(importance_type='split', iteration=bst.best_iteration)
    # }).sort_values(by='importance', ascending=False)
    # print(fea_importances)
    # print("----------------------------------------")
    # print("----------------------------------------")

    return cv_pred


def model_bagging(pred1, pred2):
    cv_pred = (pred1 + pred2 ) / 3
    return cv_pred


def model_main():
    train_data = get_feature(name="train_data")
    test_data = get_feature(name="test_data")

    print('Gen train shape: {}, test shape: {}'.format(train_data.shape, test_data.shape))
    print('features num: ', test_data.shape[1] - 1)

    # lgb_mae参数
    params_mae_lgb = {
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'regression_l1',
        'metric': 'mae',
        'feature_fraction': 0.66,
        'bagging_fraction': 0.8,
        'bagging_freq': 2,
        'num_leaves': 31,
        'verbose': -1,
        'max_depth': 5,
        'lambda_l2': 5, 'lambda_l1': 0, 'nthread': 8
    }
    # lgb_mse参数
    params_mse_lgb = {
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': 'mae',
        'feature_fraction': 0.66,
        'bagging_fraction': 0.8,
        'bagging_freq': 2,
        'num_leaves': 31,
        'verbose': -1,
        'max_depth': 5,
        'lambda_l2': 5, 'lambda_l1': 0, 'nthread': 8,
        'seed': 89
    }
    # xgb_mae参数
    params_mae_xgb = {
        'booster': 'gbtree',
        'learning_rate': 0.01,
        'max_depth': 5,
        'subsample': 0.66,
        'colsample_bytree': 0.8,
        'objective': 'reg:linear',
        'n_estimators': 10000,
        'min_child_weight': 3,
        'gamma': 0,
        'silent': True,
        'n_jobs': -1,
        'random_state': 2019,
        'reg_alpha': 2,
        'reg_lambda': 0.1,
        'alpha': 1,
        'verbose': 1
    }

    # xgb_mae_pred = xgb_mae_model(train_data, test_data, params_mae_xgb)
    lgb_mae_pred = lgb_mae_model(train_data, test_data, params_mae_lgb)
    lgb_mse_pred = lgb_mse_model(train_data, test_data, params_mse_lgb)

    bagging_pred = model_bagging(lgb_mae_pred, lgb_mse_pred)

    test_data_sub = test_data[['用户编码']]
    test_data_sub['score'] = bagging_pred
    test_data_sub.columns = ['id', 'score']
    test_data_sub['score'] = test_data_sub['score'].apply(lambda x: int(np.round(x)))
    test_data_sub[['id', 'score']].to_csv('0216_2143result.csv', index=False)


if __name__ == "__main__":
    t0 = time.time()
    model_main()
    print("Model has trained!")
    print("Cost {} s.".format(time.time() - t0))
