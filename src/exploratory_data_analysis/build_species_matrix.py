import numpy as np
from pathlib import Path
import pandas as pd



if __name__ == '__main__':
    ROOT = Path(__file__).resolve().parent.parent.parent
    train_data = np.load(ROOT / 'data/species_train.npz')

    df = pd.DataFrame(
        {
            'ID':train_data['train_ids'],
            'Longitude':train_data['train_locs'][:,0],
            'Latitude':train_data['train_locs'][:,1]
        }
    )

    taxon_names_lookup = pd.DataFrame(
        {
            'ID':train_data['taxon_ids'],
            'Names':train_data['taxon_names']
        }
    )

    print(df)


    
    def build_species_matrix(data: pd.DataFrame, taxon_names_lookup: pd.DataFrame, grid_size: float = 1.0):
        # Define grid
        lon_values = np.arange(-180, 180 + grid_size, grid_size)
        lat_values = np.arange(-90,  90 + grid_size, grid_size)
        n_lon = int(360 / grid_size)
        n_lat = int(180 / grid_size)
        n_species = len(taxon_names_lookup)

        # Digitise coordinates
        lon_idx = np.digitize(data['Longitude'], lon_values, right=False) - 1
        lat_idx = np.digitize(data['Latitude'],  lat_values,  right=False) - 1
        lon_idx = np.clip(lon_idx, 0, n_lon - 1)
        lat_idx = np.clip(lat_idx, 0, n_lat - 1)

        # Compute flattened cell indices
        box_idx = (lat_idx * n_lon + lon_idx).astype(int)

        # Map species to matrix rows
        species_idx_lookup = {sid: i for i, sid in enumerate(taxon_names_lookup['ID'])}
        species_idx = data['ID'].map(species_idx_lookup).to_numpy()

        # Build and populate the matrix
        matrix = np.zeros((n_species, n_lat * n_lon), dtype=np.uint16)
        np.add.at(matrix, (species_idx, box_idx), 1)

        return matrix

    matrix = build_species_matrix(df, taxon_names_lookup, 0.25)



    