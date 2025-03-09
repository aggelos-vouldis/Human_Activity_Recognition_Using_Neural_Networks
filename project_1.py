# Imports
import pandas as pd

import numpy as np

from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import KFold

# from tensorflow.keras.optimizers import Nadam
from keras.optimizers import Nadam

from keras.regularizers import L2
from keras.metrics import CategoricalCrossentropy, MeanSquaredError, CategoricalAccuracy
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.callbacks import EarlyStopping


# --------------------------------------------
# Woman = 1
# Man = 2
# x values = [-617,533]
#
# class values will be transformed to (1,2,3,4,5)
# sittingdown = 1
# standingup = 2
# standing = 3
# walking = 4
# sitting = 5
# --------------------------------------------


# Global Variables
INPUT_FILE = 'dataset-HAR-PUC-Rio.csv'
OUTPUT_FILE = 'A2, E.txt'
DELIM = ';'

HIDDEN_LAYER_NODES = 22
OUT_NODES = 5

EPOCHS = 500
BATCH_SIZE = 38

MOMENTUM = 0.2
LEARNING_RATE = 0.001
REGULATION = 0.1


# Basic Functions
def underSample(df):  # underSamples our dataset
    class_1 = df[df['class'] == 1]
    class_2 = df[df['class'] == 2]
    class_3 = df[df['class'] == 3]
    class_4 = df[df['class'] == 4]
    class_5 = df[df['class'] == 5]

    def getMinimunNumber(numbers):
        return min(numbers)

    min_count = getMinimunNumber([class_1['class'].count(), class_2['class'].count(
    ), class_3['class'].count(), class_4['class'].count(), class_5['class'].count()])

    # pick random samples from classes 1 - 5 (cause there is a big difference between them)
    class_1 = class_1.sample(min_count)
    class_2 = class_2.sample(min_count)
    class_3 = class_3.sample(min_count)
    class_4 = class_4.sample(min_count)
    class_5 = class_5.sample(min_count)

    return_df = pd.concat(
        [class_1, class_2, class_3, class_4, class_5], ignore_index=True)

    return return_df


def import_from_csv_and_change_values():
    imported_df = pd.read_csv(
        INPUT_FILE, delimiter=DELIM, low_memory=False)

    imported_df.replace({'gender': {"Woman": 1, "Man": 2}}, inplace=True)
    imported_df.replace({'class': {
        "sittingdown": 1,
        "standingup": 2,
        "standing": 3,
        "walking": 4,
        "sitting": 5
    }}, inplace=True)
    imported_df.replace(',', '.', inplace=True, regex=True)

    imported_df['how_tall_in_meters'] = imported_df['how_tall_in_meters'].astype(
        float)
    imported_df['body_mass_index'] = imported_df['body_mass_index'].astype(
        float)

    # remove unnecessary data from dataframe
    imported_df = imported_df.drop('user', axis='columns')
    imported_df = underSample(imported_df)
    return imported_df


# Retrieve Info and change it
def main():
    start_dataframe = import_from_csv_and_change_values()
    print("Data reached and ready!")
    print(start_dataframe)
    norm_dataset = start_dataframe.to_numpy()
    X = norm_dataset[:, :-1]
    y = norm_dataset[:, -1]

    X = QuantileTransformer().fit_transform(X)

    encoder = LabelEncoder().fit(y)
    encoded_Y = encoder.transform(y)
    # convert integers to dummy variables (i.e. one hot encoded)
    y = np_utils.to_categorical(encoded_Y)
    print("Y converted to categorical variables!")

    kfold = KFold(n_splits=5, shuffle=True)
    print("KFolds done! Starting with cross validation")

    # Validate model
    fold = 0

    scores_mse = []
    scores_accuracy = []
    scores_CE = []

    print(
        f"HIDDEN_LAYER_NODES={HIDDEN_LAYER_NODES}   OUT_NODES = {OUT_NODES}   EPOCHS={EPOCHS}   BATCH_SIZE={BATCH_SIZE}")

    for train, test in kfold.split(X):
        fold += 1

        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]

        model = Sequential()
        model.add(Dense(HIDDEN_LAYER_NODES, input_dim=17,
                        activation='relu', activity_regularizer=L2(REGULATION)))
        model.add(Dense(OUT_NODES, input_dim=HIDDEN_LAYER_NODES,
                        activation='softmax', activity_regularizer=L2(REGULATION)))

        metrics = [
            CategoricalAccuracy(name='ACC'),
            MeanSquaredError(name='MSE'),
            CategoricalCrossentropy(name='CE')
        ]

        res = model.compile(loss='categorical_crossentropy', optimizer=Nadam(
            learning_rate=LEARNING_RATE, use_ema=True, ema_momentum=MOMENTUM), metrics=metrics)

        earlyStop = EarlyStopping(
            monitor='val_loss', mode='min', verbose=1, patience=200)

        model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=0,
                  epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[earlyStop])

        pred = model.evaluate(
            X_test, y_test, use_multiprocessing=True, workers=4, verbose=2)

        score_mse = pred[2]
        scores_mse.append(score_mse)

        score_accuracy = pred[1]
        scores_accuracy.append(score_accuracy)

        score_CE = pred[3]
        scores_CE.append(score_CE)

        print(f"#Fold {fold}    (MSE): {round(score_mse, 5)}    (Accuracy): {round(score_accuracy, 5)}    (CE): {round(score_CE, 5)}")

    print(
        f"\nFinal, out of sample    (MSE): {round(np.mean(scores_mse), 5)}    (Accuracy): {round(np.mean(scores_accuracy), 5)}    (CE): {round(np.mean(scores_CE), 5)}")


if __name__ == '__main__':
    main()
