import pandas as pd
from os import listdir
from os.path import join, isdir, isfile, basename


def mapper(root_path: str, classes: pd.DataFrame) -> pd.DataFrame:
    # Initiates mapper
    mapper_dict = {
        "Chemin": [],
        "Bord": [],
        "Phyllotaxie": [],
        "Type_feuille": [],
        "Ligneux": []
    }

    # Retrieves subfolders
    folders_paths = [join(root_path, f) for f in listdir(root_path) if isdir(join(root_path, f))]
    print(folders_paths)

    # Retrieves files paths and its features
    for fold in folders_paths:
        # Features
        plant_name = basename(fold)
        print(plant_name)
        bord = classes.loc[classes['Nom'] == plant_name, 'Bord'].item()
        phyllotaxie = classes.loc[classes['Nom'] == plant_name, 'Phyllotaxie'].item()
        type_feuille = classes.loc[classes['Nom'] == plant_name, 'Type_feuille'].item()
        ligneux = classes.loc[classes['Nom'] == plant_name, 'Ligneux'].item()

        # Append to map
        for f in [join(fold, file) for file in listdir(fold) if isfile(join(fold, file))]:
            mapper_dict['Chemin'].append(f)
            mapper_dict['Bord'].append(bord)
            mapper_dict['Phyllotaxie'].append(phyllotaxie)
            mapper_dict['Type_feuille'].append(type_feuille)
            mapper_dict['Ligneux'].append(ligneux)

    return pd.DataFrame(mapper_dict)

if __name__ == "__main__":
    Bord = {0: "lisse", 1: "denté"}
    Phyllotaxie = {0: "alterné", 1: "opposé"}
    Type_feuille = {0: "simple", 1: "composée"}
    Ligneux = {0: "non", 1: "oui"}

    classes = {
        "Nom": ["convolvulaceae", "monimiaceae", "amborella", "castanea", "desmodium", "eugenia", "laurus", "litsea", "magnolia", "rubus", "ulmus"],
        "Bord": [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
        "Phyllotaxie": [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        "Type_feuille": [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        "Ligneux": [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    }
    classes_df = pd.DataFrame(classes)
    print(classes_df)

    dataset = "../dataset/Test"
    mapped = mapper(dataset, classes_df).set_index('Chemin').to_csv(dataset + ".csv")
    dataset = "../dataset/Train"
    mapped = mapper(dataset, classes_df).set_index('Chemin').to_csv(dataset + ".csv")

