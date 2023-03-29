from pathlib import Path


parent = Path().cwd().parent
root = Path.joinpath(parent, 'observacions')
path_observacions_filtrat = Path.joinpath(root, 'observacions_filtrat')
path_observacions = Path.joinpath(root, 'observacions')
estacions_path = Path.joinpath(root, "estacions_cabal.csv")
split_train_test = Path.joinpath(root, "splits_train_test.csv")
