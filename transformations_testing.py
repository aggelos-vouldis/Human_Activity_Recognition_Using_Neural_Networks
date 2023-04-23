import pandas as pd

import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler

input_file = 'dataset-HAR-PUC-Rio.csv'
output_file = 'min_max_results.txt'
delim = ';'

xyzList = ["gender", "age", "how_tall_in_meters", "weight", "body_mass_index", "x1", "y1", "z1", "x2", "y2",
           "z2", "x3", "y3", "z3", "x4", "y4", "z4", "class"]


def returnXYZNumber(number):
    if number == 5:
        return 1
    elif number == 8:
        return 2
    elif number == 11:
        return 3
    else:
        return 4


def delete_file_contents():
    delete_file_contents = open(output_file, 'w')
    delete_file_contents.close()


def export_min_max_to_txt(dataset, transformation_name, i):
    with open(output_file, 'a') as file:
        if i == 0:
            file.write(f"{transformation_name}: \n")
        else:
            file.write(f"\n\n{transformation_name}: \n")
        for i in range(5, 16, 3):
            file.write(f"""X{returnXYZNumber(i)} min: {dataset[:, i].min()}   X{returnXYZNumber(i)} max: {dataset[:, i].max()}
Y{returnXYZNumber(i)} min: {dataset[:, i+1].min()}   Y{returnXYZNumber(i)} max: {dataset[:, i+1].max()}
Z{returnXYZNumber(i)} min: {dataset[:, i+2].min()}   Z{returnXYZNumber(i)} max: {dataset[:, i+2].max()}
""")

        file.write(f"""
gender min: {dataset[:, 0].min()}   gender max: {dataset[:, 0].max()}
age:{dataset[:, 1].min()}   age max: {dataset[:, 1].max()}
how_tall_in_meters: {dataset[:, 2].max()}   how_tall_in_meters: {dataset[:, 2].min()}
weight: {dataset[:, 3].min()}   weight max: {dataset[:, 3].max()}
body_mass_index: {dataset[:, 4].min()}   body_mass_index: {dataset[:, 4].max()}""")


def visualizeData(visualization_title, dataset):
    fig = plt.figure()

    ax = fig.add_subplot(projection='3d')
    ax.scatter(dataset[:, 5], dataset[:, 6], dataset[:, 7],
               marker=".", label='1')  # blue
    ax.scatter(dataset[:, 8], dataset[:, 9], dataset[:, 10],
               marker=".", label='2')  # orange
    ax.scatter(dataset[:, 11], dataset[:, 12], dataset[:, 13],
               marker=".", label='3')  # green

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.legend()
    if visualization_title != None:
        plt.title(visualization_title)
    plt.show()


def remove_data_from_dataframe(df):
    tempdf = df.drop('user', axis='columns')
    return tempdf


def import_from_csv_and_change_values():
    imported_df = pd.read_csv(input_file, delimiter=delim, low_memory=False)

    # --------------------------------------------
    # x values = [-617,533]
    # class values will be transformed to (1,2,3,4,5)
    # sittingdown = 1
    # standingup = 2
    # standing = 3
    # walking = 4
    # sitting = 5
    # --------------------------------------------
    imported_df.replace("sittingdown", 1, inplace=True)
    imported_df.replace("standingup", 2, inplace=True)
    imported_df.replace("standing", 3, inplace=True)
    imported_df.replace("walking", 4, inplace=True)
    imported_df.replace("sitting", 5, inplace=True)

    # --------------------------------------------
    # Woman = 1
    # Man = 2
    # --------------------------------------------
    imported_df.replace("Woman", 1, inplace=True)
    imported_df.replace("Man", 2, inplace=True)
    imported_df.replace(',', '.', inplace=True, regex=True)

    imported_df['how_tall_in_meters'] = imported_df['how_tall_in_meters'].astype(
        float)
    imported_df['body_mass_index'] = imported_df['body_mass_index'].astype(
        float)

    df_data_removed = remove_data_from_dataframe(imported_df)
    return df_data_removed


def main():
    start_dataframe = import_from_csv_and_change_values()
    print(pd.Series(start_dataframe['class']).value_counts())
    norm_dataset = start_dataframe.to_numpy()

    X = norm_dataset[:, :-1]
    y = norm_dataset[:, -1]

    types_of_transformation = [
        {"key": 0, "name": "Plain Data - No Transformation", "X": X},
        {"key": 1, "name": "Quantile Transformation",
            "X": QuantileTransformer().fit_transform(X)},
        {"key": 2, "name": "Standard Transformation",
            "X": StandardScaler().fit_transform(X)},  # Centering
        {"key": 3, "name": "Min-Max Transformation",
            "X": MinMaxScaler().fit_transform(X)}  # Normalization - Min Max Scaler
    ]

    delete_file_contents()
    for i, type_of_transformation in enumerate(types_of_transformation):
        export_min_max_to_txt(
            type_of_transformation["X"], type_of_transformation["name"], i)
        # visualizeData(
        #     type_of_transformation["name"], type_of_transformation["X"])

    print(f"""------------------------------------------------------------------------------
Transformations minimum and maximum values are exported to '{output_file}'
------------------------------------------------------------------------------""")


if __name__ == "__main__":
    main()
