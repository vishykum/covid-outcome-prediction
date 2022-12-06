import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show_train_dataset_pie_chart(train_dataset: pd.DataFrame, title: str):
    plt.figure()
    data = train_dataset.groupby("outcome_group").size()
    print("\n" + title + ",")
    print(data)
    data = [int(data[0]), int(data[1]), int(data[2])]
    labels = ["deceased", "hospitalized", "nonhospitalized"]
    colours = sns.color_palette('pastel')[0:4]
    plt.pie(x=data, labels=labels, colors=colours, autopct='%.0f%%')
    plt.title(title)
    plt.show()
