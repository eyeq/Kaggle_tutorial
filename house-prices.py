import collections

import numpy as np
import pandas as pd
from sklearn import preprocessing, decomposition, linear_model, svm, neighbors, tree, ensemble, model_selection, metrics
from keras import models, layers, losses, optimizers, wrappers, callbacks, utils
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    print("データ読み込み中")
    train = pd.read_csv("house-prices/train.csv")
    test = pd.read_csv("house-prices/test.csv")

    # print(train.info())
    # print(test.info())

    print("データ変換中")
    train_y = train["SalePrice"]
    train_x, train_y, test_x = convert_data(train.drop("SalePrice", axis=1), train_y, test)

    # print(train_x.info())
    # print(test_x.info())

    print("学習中")
    train_y = train_y - train["MiscVal"]
    test_y = predict(train_x, train_y, test_x)
    test_y = test_y + test["MiscVal"]

    print("結果出力中")
    csv = pd.DataFrame()
    csv["Id"] = test["Id"]
    csv["SalePrice"] = test_y
    csv.to_csv("house-prices/output.csv", index=False)

    print("出力完了")


def convert_data(train_data: pd.DataFrame, train_y: pd.Series, test_data: pd.DataFrame) -> (pd.DataFrame, pd.Series, pd.DataFrame):
    train_x = pd.DataFrame()
    test_x = pd.DataFrame()

    for column in train_data:
        # print(train_data[column].value_counts())
        types = pd.concat([train_data[column], train_data[column]]).dtypes

        if types == "int32" or types == "int64" or types == "float64":
            train_x[column] = train_data[column].fillna(0)
            test_x[column] = test_data[column].fillna(0)
        else:
            train_x[column] = train_data[column].fillna("NA")
            test_x[column] = test_data[column].fillna("NA")

            encoder = preprocessing.LabelEncoder()
            encoder.fit(pd.concat([train_x[column], test_x[column]]))
            train_x[column] = encoder.transform(train_x[column])
            test_x[column] = encoder.transform(test_x[column])

    # train_y = np.log(train_y)
    train_x['Remodel'] = train_x['YearBuilt'] != train_x['YearRemodAdd']
    test_x['Remodel'] = test_x['YearBuilt'] != test_x['YearRemodAdd']

    # train_x['NonHouseArea'] = train_x['LotArea'] - train_x['1stFlrSF']
    # test_x['NonHouseArea'] = test_x['LotArea'] - test_x['1stFlrSF']

    train_x['TotalHouseSF'] = train_x['TotalBsmtSF'] + train_x['GrLivArea']
    test_x['TotalHouseSF'] = test_x['TotalBsmtSF'] + test_x['GrLivArea']

    train_x['TotalPorchSF'] = train_x["WoodDeckSF"] + train_x["OpenPorchSF"] + train_x["EnclosedPorch"] + train_x["3SsnPorch"] + train_x["ScreenPorch"]
    test_x['TotalPorchSF'] = test_x["WoodDeckSF"] + test_x["OpenPorchSF"] + test_x["EnclosedPorch"] + test_x["3SsnPorch"] + test_x["ScreenPorch"]

    train_x['TotalSF'] = train_x['TotalHouseSF'] + train_x['TotalPorchSF']
    test_x['TotalSF'] = test_x['TotalHouseSF'] + test_x['TotalPorchSF']

    train_x["AllFullBath"] = train_x["BsmtFullBath"] + train_x["FullBath"]
    test_x["AllFullBath"] = test_x["BsmtFullBath"] + test_x["FullBath"]

    train_x["AllHalfBath"] = train_x["BsmtHalfBath"] + train_x["HalfBath"]
    test_x["AllHalfBath"] = test_x["BsmtHalfBath"] + test_x["HalfBath"]

    train_x["Toilet"] = train_x["AllFullBath"] + train_x["AllHalfBath"]
    test_x["Toilet"] = test_x["AllFullBath"] + test_x["AllHalfBath"]

    # train_x["GarageExt"] = (train_x["YearBuilt"] != train_x["GarageYrBlt"]) & (train_x["GarageYrBlt"] != 0)
    # test_x["GarageExt"] = (test_x["YearBuilt"] != test_x["GarageYrBlt"]) & (test_x["GarageYrBlt"] != 0)

    # 不要なデータを除外
    train_x = train_x.drop(["Id", "MiscFeature", "MiscVal"], axis=1)
    test_x = test_x.drop(["Id", "MiscFeature", "MiscVal"], axis=1)

    # データ数が少なすぎるものを除外
    train_x = train_x.drop(["Street", "Utilities", "PoolArea"], axis=1)
    test_x = test_x.drop(["Street", "Utilities", "PoolArea"], axis=1)

    '''
    rf = ensemble.RandomForestRegressor(n_estimators=100)
    rf.fit(train_x, train_y)
    view = pd.DataFrame()
    view["Feature"] = train_x.columns[0:]
    view["Importance"] = rf.feature_importances_
    print(view.sort_values("Importance", ascending=False))
    '''

    return train_x, train_y, test_x


def predict(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame) -> np.ndarray:
    models = collections.OrderedDict([
        # ("SGD", linear_model.SGDRegressor(max_iter=1000, random_state=0)),
        # ("Lasso", linear_model.Lasso(alpha=1.0, random_state=0)),
        # ("Ridge", linear_model.Ridge(alpha=1.0, random_state=0)),
        # ("Elastic Net", linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=0)),
        # ("線形SVM", svm.LinearSVR(C=0.01, epsilon=2.0)),
        # ("カーネルSVM", svm.SVR(kernel='rbf', C=0.01, gamma=0.1, epsilon=0.1)),
        # ("最近傍法", neighbors.KNeighborsRegressor(n_neighbors=1, weights='distance')),
        # ("K近傍法", neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance')),
        # ("決定木", tree.DecisionTreeRegressor()),
        ("ランダムフォレスト", ensemble.RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)),
        ('bagging', ensemble.BaggingRegressor(tree.DecisionTreeRegressor(random_state=0), n_estimators=100, n_jobs=-1, random_state=0)),
        ('AdaBoost', ensemble.AdaBoostRegressor(tree.DecisionTreeRegressor(random_state=0), n_estimators=100, random_state=0)),
        # ('Bagging & AdaBoost', ensemble.AdaBoostRegressor(ensemble.BaggingRegressor(tree.DecisionTreeRegressor(random_state=0), n_estimators=2000, random_state=0), random_state=0)),
        ('GradientBoost', ensemble.GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, random_state=0)),
        # ('XGBoost', xgb.XGBRegressor(n_estimators=100, random_state=0)),
        # ('XGBoostRF', xgb.XGBRFRegressor(n_estimators=100, random_state=0)),
    ])

    train_y = train_y
    for k, v in models.items():
        scores = model_selection.cross_validate(v, train_x, train_y, cv=5,
                                                scoring=metrics.make_scorer(lambda y_true, y_pred: np.sqrt(metrics.mean_squared_error(y_true, y_pred))))
        print("  ", k)
        print("    ", "RMSE     = ", scores["test_score"].mean())
        print("    ", "標準偏差 = ", scores["test_score"].std())

    model = models["GradientBoost"]
    model.fit(train_x, train_y)
    return model.predict(test_x)


if __name__ == '__main__':
    pd.set_option('display.max_rows', 100)
    main()
