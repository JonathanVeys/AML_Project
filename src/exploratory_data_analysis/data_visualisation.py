import numpy as np
from pathlib import Path
from shapely.geometry import Point
import matplotlib.pyplot as plt
import geopandas as gpd
import geodatasets




if __name__ == '__main__':
    ROOT = Path(__file__).resolve().parent.parent.parent
    train_data = np.load(ROOT / 'data/species_train.npz')
    train_locations = train_data['train_locs']
    train_ids = train_data['train_ids']
    taxon_ids = train_data['taxon_ids']
    taxon_names = train_data['taxon_names']

    geometry = [Point(xy) for xy in zip(train_locations[:,1], train_locations[:,0])]
    gdf = gpd.GeoDataFrame(train_locations, geometry=geometry)   
    world = gpd.read_file(geodatasets.data.naturalearth.land['url'])
    gdf.plot(ax=world.plot(figsize=(10, 6)), marker='.', color='red', markersize=0.5)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Map of species sightings')
    plt.show()