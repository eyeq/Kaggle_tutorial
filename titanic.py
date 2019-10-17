import collections

import numpy as np
import pandas as pd
from sklearn import preprocessing, decomposition, linear_model, svm, neighbors, tree, ensemble, model_selection


def main():
    print("データ読み込み中")
    train = pd.read_csv("titanic/train.csv")
    test = pd.read_csv("titanic/test.csv")

    print("データ変換中")
    train_y = train["Survived"]
    train_x, test_x = convert_data(train.drop("Survived", axis=1), train_y, test)

    print("学習中")
    test_y = predict(train_x, train_y, test_x)

    print("結果出力中")
    csv = pd.DataFrame()
    csv["PassengerId"] = test["PassengerId"]
    csv["Survived"] = test_y
    csv.to_csv("titanic/output.csv", index=False)

    print("出力完了")


def convert_data(train_data: pd.DataFrame, train_y: pd.Series, test_data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    # 姓
    train_data["Family_Name"] = train_data["Name"].replace(", .*", "", regex=True)
    test_data["Family_Name"] = test_data["Name"].replace(", .*", "", regex=True)

    # 敬称
    train_data["Honorific"] = train_data["Name"].replace(".*, ", "", regex=True).replace(" .*", "", regex=True)
    test_data["Honorific"] = test_data["Name"].replace(".*, ", "", regex=True).replace(" .*", "", regex=True)

    # 家族
    train_data["Family"] = train_data["SibSp"] + train_data["Parch"] + 1
    test_data["Family"] = test_data["SibSp"] + test_data["Parch"] + 1

    # 家族の生存率
    alone = train_y[train_data["Family"] <= 1].mean()
    small = train_y[(1 < train_data["Family"]) & (train_data["Family"] <= 4)].mean()
    large = train_y[4 < train_data["Family"]].mean()

    family_survived = []
    for i, r in train_data.iterrows():
        temp = train_y[(train_data["PassengerId"] != r["PassengerId"]) & (train_data["Family_Name"] == r["Family_Name"]) & (train_data["SibSp"] == r["SibSp"]) & (train_data["Parch"] == r["Parch"])]
        family_survived.append(temp.mean())
    train_data["Family_Survived"] = family_survived
    train_data["Family_Survived"] = train_data.apply(lambda r: r["Family_Survived"] if not np.math.isnan(r["Family_Survived"]) else alone if r["Family"] <= 1 else small if r["Family"] <= 4 else large, axis=1)
    family_survived = []
    for i, r in test_data.iterrows():
        temp = train_y[(train_data["PassengerId"] != r["PassengerId"]) & (train_data["Family_Name"] == r["Family_Name"]) & (train_data["SibSp"] == r["SibSp"]) & (train_data["Parch"] == r["Parch"])]
        family_survived.append(temp.mean())
    test_data["Family_Survived"] = family_survived
    test_data["Family_Survived"] = test_data.apply(lambda r: r["Family_Survived"] if not np.math.isnan(r["Family_Survived"]) else alone if r["Family"] <= 1 else small if r["Family"] <= 4 else large, axis=1)

    total_data = pd.concat([train_data, test_data], ignore_index=True)

    # 家族内の子供の数
    train_data["Child"] = train_data.apply(lambda r: ((total_data["Family_Name"] == r["Family_Name"]) & (total_data["SibSp"] == r["SibSp"]) & (total_data["Parch"] == r["Parch"]) & (total_data["Age"] < 14)).count(), axis=1)
    test_data["Child"] = test_data.apply(lambda r: ((total_data["Family_Name"] == r["Family_Name"]) & (total_data["SibSp"] == r["SibSp"]) & (total_data["Parch"] == r["Parch"]) & (total_data["Age"] < 14)).count(), axis=1)

    # 同じチケット
    train_data["Ticket_Count"] = train_data["Ticket"].map(lambda d: (total_data["Ticket"] == d).sum())
    test_data["Ticket_Count"] = test_data["Ticket"].map(lambda d: (total_data["Ticket"] == d).sum())

    # 家族以外（親戚/友達）
    train_data["Friend"] = train_data.apply(lambda r: ((total_data["Ticket"] == r["Ticket"]) & ~((total_data["Family_Name"] == r["Family_Name"]) & (total_data["SibSp"] == r["SibSp"]) & (total_data["Parch"] == r["Parch"]))).sum(), axis=1)
    test_data["Friend"] = test_data.apply(lambda r: ((total_data["Ticket"] == r["Ticket"]) & ~((total_data["Family_Name"] == r["Family_Name"]) & (total_data["SibSp"] == r["SibSp"]) & (total_data["Parch"] == r["Parch"]))).sum(), axis=1)

    total_data = pd.concat([train_data, test_data], ignore_index=True)
    train_x = make_x(train_data, total_data)
    test_x = make_x(test_data, total_data)

    return train_x, test_x


def predict(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame) -> np.ndarray:
    # models = collections.OrderedDict([
    #     ("SGD", linear_model.SGDClassifier(loss="hinge", max_iter=5, tol=-np.infty, random_state=0)),
    #     ("パーセプトロン", linear_model.Perceptron(max_iter=5, tol=-np.infty, random_state=0)), # SGDClassifier(loss="perceptron")
    #     ("ロジスティック回帰", linear_model.LogisticRegression(solver='liblinear', multi_class='auto', C=1.0, penalty="l2", random_state=0)), # SGDClassifier(loss="log")
    #     ("線形SVM", svm.LinearSVC(loss="hinge", C=1.0, class_weight="balanced", random_state=0)),
    #     ("カーネルSVM", svm.SVC(kernel='rbf', gamma=1/2, C=1.0, class_weight='balanced', random_state=0)),
    #     ("最近傍法", neighbors.KNeighborsClassifier(n_neighbors=1, weights='distance')),
    #     ("K近傍法", neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance')),
    #     ("決定木", tree.DecisionTreeClassifier()),
    #     ("ランダムフォレスト", ensemble.RandomForestClassifier(criterion='entropy', n_estimators=100, class_weight="balanced", random_state=0)),
    # ])
    #
    # for k, v in models.items():
    #     scores = model_selection.cross_validate(v, train_x, train_y, cv=5)
    #     print("  ", k)
    #     print("    ", "正解率   = ", scores["test_score"].mean())
    #     print("    ", "標準偏差 = ", scores["test_score"].std())

    model = linear_model.LogisticRegression(solver='liblinear', multi_class='auto', C=1.0, penalty="l2", random_state=0)
    model.fit(train_x, train_y)
    return model.predict(test_x)


def make_x(data: pd.DataFrame, total: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame()

    # 旅客等級
    pclass = data["Pclass"].fillna(data["Pclass"].median())
    x["Pclass_1"] = pclass.map(lambda d: d == 1)
    x["Pclass_2"] = pclass.map(lambda d: d == 2)
    x["Pclass_3"] = pclass.map(lambda d: d == 3)
    # 氏名
    name = data["Name"]
    # 性別
    sex = data["Sex"].fillna("")
    x["Sex_M"] = sex.map(lambda d: d == "male")
    x["Sex_F"] = sex.map(lambda d: d == "female")
    # 年齢
    age = data.apply(lambda r: r["Age"] if not np.math.isnan(r["Age"]) else total[total["Honorific"] == r["Honorific"]]["Age"].median(), axis=1)
    x["Age_Y"] = age.map(lambda d: d < 14)
    x["Age_A"] = age.map(lambda d: 14 <= d < 50)
    x["Age_E"] = age.map(lambda d: 50 <= d)
    x["Age_NaN"] = data["Age"].map(lambda d: np.math.isnan(d))
    # 兄弟・配偶者
    sibsp = data["SibSp"]
    # 親・子供
    parch = data["Parch"]
    # 家族
    family = data["Family"]
    x["Alone"] = family.map(lambda d: d <= 1)
    x["Small"] = family.map(lambda d: 1 < d <= 4)
    x["Large"] = family.map(lambda d: 4 < d)
    # 家族の生存率
    family_survived = data["Family_Survived"]
    x["Family_Survived"] = family_survived

    # testデータの正答率には影響なし
    # 家族に子供が含まれるか
    child = data["Child"]
    x["Child_In_Family"] = child.map(lambda d: 0 < d)
    # チケット番号
    ticket = data["Ticket"]
    # 友達
    friend = data["Friend"]
    x["NoFriends"] = friend.map(lambda d: d == 0)
    # 運賃
    fare = data["Fare"].fillna(0)
    x["Fare_L"] = fare.map(lambda d: 0 < d < 10.5)  # Pclass3のみ 船底に近い？
    x["Fare_M"] = fare.map(lambda d: 25.5 <= d < 50.0)  # Pclass1/2/3混合
    x["Fare_H"] = fare.map(lambda d: 50.0 <= d)  # ほぼPclass1

    # 不要なデータ
    # 客室番号
    cabin = data["Cabin"].fillna("")
    # x["Cabin_A"] = cabin.map(lambda d: d.startswith("A"))
    # x["Cabin_B"] = cabin.map(lambda d: d.startswith("B"))
    # x["Cabin_C"] = cabin.map(lambda d: d.startswith("C"))
    # x["Cabin_D"] = cabin.map(lambda d: d.startswith("D"))
    # x["Cabin_E"] = cabin.map(lambda d: d.startswith("E"))
    # x["Cabin_F"] = cabin.map(lambda d: d.startswith("F"))
    # x["Cabin_G"] = cabin.map(lambda d: d.startswith("G"))
    # x["Cabin_S"] = cabin.map(lambda d: d.endswith(("1", "3", "5", "7", "9"))
    # x["Cabin_P"] = cabin.map(lambda d: d.endswith(("0", "2", "4", "6", "8"))
    # 出港地
    embarked = data["Embarked"].fillna("")
    # x["Embarked_C"] = embarked.map(lambda d: d == "C")
    # x["Embarked_Q"] = embarked.map(lambda d: d == "Q")
    # x["Embarked_S"] = embarked.map(lambda d: d == "S")
    return x


if __name__ == '__main__':
    main()
