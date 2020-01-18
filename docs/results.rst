=======
Results
=======

.. plot::
   :context: close-figs
   :include-source:

    import os.path
    import numpy as np
    import pandas as pd

    import matplotlib.pyplot as plt
    import seaborn as sns

.. plot::
   :context: close-figs
   :include-source:

    n_splits = 3
    name = "default"
    summary_dir = "../logs"

    splits = range(n_splits)

    names = ["default", "adam", "adam-beta-0.5"]
    pretty_names = ["RMSProp", r"Adam ($\beta_1=0.9$)", r"Adam ($\beta_1=0.5$)"]
    pretty_name_mapping = dict(zip(names, pretty_names))

    def get_basename(name, split_num):

        return f"{name}.split{split_num:d}"

    def make_plot_data(names, splits):

        df_list = []

        for name in names:
            for split_num in splits:

                basename = get_basename(name, split_num)
                csv_path = os.path.join(summary_dir, f"{basename}.csv")

                df = pd.read_csv(csv_path).assign(name=name, split=split_num)
                df_list.append(df)

        data = pd.concat(df_list, axis="index", sort=True)

        return data

    data = make_plot_data(names, splits).rename(columns=dict(acc="train", val_acc="validation"))

    data = data.assign(name=data.name.replace(pretty_name_mapping))

    new_data = pd.melt(data,
                       id_vars=["name", "split", "epoch"],
                       value_vars=["train", "validation"],
                       value_name="accuracy", var_name="partition")

    g = sns.relplot(x="epoch", y="accuracy", hue="name",
    #                  units="split", estimator=None,
                    row="partition", ci="sd",
                    height=5, aspect=golden_ratio,
                    data=new_data, kind="line")
    g.set(yscale="log")