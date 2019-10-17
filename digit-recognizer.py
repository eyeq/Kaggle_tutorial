import numpy as np
import pandas as pd
from sklearn import preprocessing, decomposition, linear_model, svm, neighbors, tree, ensemble, model_selection
from keras import models, layers, losses, optimizers, wrappers, callbacks, utils


def main():
    print("データ読み込み中")
    train = pd.read_csv("digit-recognizer/train.csv")
    test = pd.read_csv("digit-recognizer/test.csv")

    print("データ変換中")
    train_y = train["label"]
    train_x, train_y, test_x = convert_data(train.drop("label", axis=1), train_y, test)

    print("学習中")
    test_y = predict(train_x, train_y, test_x)

    print("結果出力中")
    csv = pd.DataFrame()
    csv["ImageId"] = range(1, len(test_y) + 1)
    csv["Label"] = test_y
    csv.to_csv("digit-recognizer/output.csv", index=False)

    print("出力完了")


def convert_data(train_data: pd.DataFrame, train_y: pd.Series, test_data: pd.DataFrame) -> (np.ndarray, np.ndarray, np.ndarray):
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    train_x = train_data.values.reshape(-1, 28, 28, 1)
    train_y = utils.np_utils.to_categorical(train_y, num_classes=10)
    test_x = test_data.values.reshape(-1, 28, 28, 1)

    return train_x, train_y, test_x


def generate_model():
    model = models.Sequential()

    model.add(layers.Conv2D(32, 5, activation="relu", input_shape=(28, 28, 1)))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(64, 5, activation="relu"))
    model.add(layers.MaxPool2D())

    model.add(layers.Flatten())

    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation="softmax"))

    model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.Adam(), metrics=["accuracy"])
    return model


def predict(train_x: np.ndarray, train_y: pd.Series, test_x: np.ndarray) -> np.ndarray:
    # clf = wrappers.scikit_learn.KerasClassifier(build_fn=generate_model, epochs=20, batch_size=64)

    # scores = model_selection.cross_validate(clf, train_x, train_y, cv=5)
    # print("    ", "正解率   = ", scores["test_score"].mean())
    # print("    ", "標準偏差 = ", scores["test_score"].std())

    epochs = 20
    lr_decay = callbacks.LearningRateScheduler(lambda e: 1e-3 if e < epochs * 0.5 else 1e-4 if e < epochs * 0.75 else 1e-5)

    model = generate_model()
    model.fit(train_x, train_y, batch_size=64, epochs=epochs, callbacks=[lr_decay])
    return np.argmax(model.predict(test_x), axis=1)


if __name__ == '__main__':
    main()
