from graph_id.core.graph_id import GraphIDGenerator


# https://github.com/hackingmaterials/matminer_examples/blob/main/matminer_examples/machine_learning-nb/voronoi-ward-prb-2017.ipynb
import argparse
import os

import numpy as np
import pandas as pd
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import (
    ElementProperty,
    IonProperty,
    Stoichiometry,
    ValenceOrbital,
)
from matminer.featurizers.structure import (
    ChemicalOrdering,
    MaximumPackingEfficiency,
    SiteStatsFingerprint,
    StructuralHeterogeneity,
    StructureComposition,
)

# read json from a file
from monty.serialization import loadfn
from pymatgen.core import Structure
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from matminer.featurizers.conversions import ConversionFeaturizer


class CifToStructure(ConversionFeaturizer):
    def __init__(self, target_col_id="cif", overwrite_data=False):
        self._target_col_id = target_col_id
        self._overwrite_data = overwrite_data
        super().__init__(target_col_id, overwrite_data)

    def citations(self):
        return []

    def featurize(self, string):
        s = Structure.from_str(input_string=string, fmt="cif")
        return [s]

    def implementors(self):
        return ["Koki Muraoka"]


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=int, default=2019)
parser.add_argument("--debug", action="store_true")
parser.add_argument("-s", "--seed", type=int, default=0)


args = parser.parse_args()


random_seed = args.seed

featurizer = MultipleFeaturizer(
    [
        SiteStatsFingerprint.from_preset("CoordinationNumber_ward-prb-2017"),
        StructuralHeterogeneity(),
        ChemicalOrdering(),
        MaximumPackingEfficiency(),
        SiteStatsFingerprint.from_preset("LocalPropertyDifference_ward-prb-2017"),
        StructureComposition(Stoichiometry()),
        StructureComposition(ElementProperty.from_preset("magpie")),
        StructureComposition(ValenceOrbital(props=["frac"])),
        StructureComposition(IonProperty(fast=True)),
    ]
)


json_path = "data/mp.2019.04.01.json"

pickle_path = f"data/mp2019-debug{args.debug}.pickle"


if not os.path.exists(pickle_path):
    print(f"No {pickle_path}")
    print(f"Reading {json_path}")
    data = loadfn(json_path)
    print("The number of structures is {}".format(len(data)))
    print(
        "formation_energy_per_atom is {}".format(data[0]["formation_energy_per_atom"])
    )

    d1 = pd.DataFrame(data)
    if args.debug:
        d1 = d1.head(2000)

    c2s = CifToStructure()
    d2 = c2s.featurize_dataframe(d1, col_id="structure")

    d2["structure"] = d2["cif"]
    del d2["cif"]

    gidgen = GraphIDGenerator()

    print("Calculating Graph IDs...")

    d3 = featurizer.featurize_dataframe(d2, "structure", ignore_errors=True)
    d3["graph_id"] = gidgen.get_many_ids(d3["structure"].values, parallel=True)

    del d3["structure"]
    print("Saving to pickle...")
    d3.to_pickle(pickle_path)
else:
    print("Reading from pickle...")
    d3 = pd.read_pickle(pickle_path)


train_df, test_df = train_test_split(d3, train_size=0.5, random_state=random_seed)

X_train = train_df[featurizer.feature_labels()].values
X_test = test_df[featurizer.feature_labels()].values

y_train = train_df["formation_energy_per_atom"].values
y_test = test_df["formation_energy_per_atom"].values


model = Pipeline(
    [
        ("imputer", SimpleImputer()),  # For the failed structures
        ("model", RandomForestRegressor(n_estimators=150, n_jobs=-1)),
    ]
)

model.fit(X_train, y_train)

leaked_df = test_df[test_df.graph_id.isin(train_df.graph_id)]
unleaked_df = test_df[test_df.graph_id.isin(train_df.graph_id) is False]

X_leaked = leaked_df[featurizer.feature_labels()].values
X_unleaked = unleaked_df[featurizer.feature_labels()].values

y_leaked = leaked_df["formation_energy_per_atom"].values
y_unleaked = unleaked_df["formation_energy_per_atom"].values


y_leaked_pred = model.predict(X_leaked)
mae = mean_absolute_error(y_leaked, y_leaked_pred)
print(mae)


y_unleaked_pred = model.predict(X_unleaked)
mae = mean_absolute_error(y_unleaked, y_unleaked_pred)
print(mae)
