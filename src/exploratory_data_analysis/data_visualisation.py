import numpy as np
from pathlib import Path
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import geodatasets
from matplotlib.colors import LogNorm

def build_mesh(grid_size:float, xmin:int=-180, ymin:int=-90, xmax:int=180, ymax:int=90):
    '''
    A function for creating a mesh to build a heat map
    '''
    cols = np.arange(xmin, xmax+grid_size, grid_size)
    rows = np.arange(ymin, ymax+grid_size, grid_size)
    polygons = [box(x, y, x + grid_size, y + grid_size) for x in cols[:-1] for y in rows[:-1]]
    mesh = gpd.GeoDataFrame({'geometry':polygons}, crs='EPSG:4326')
    return mesh

if __name__ == '__main__':
    ROOT = Path(__file__).resolve().parent.parent.parent
    train_data = np.load(ROOT / 'data/species_train.npz')

    species_data = pd.DataFrame(
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
    
    data = pd.merge(species_data, taxon_names_lookup, on='ID', how='left')

    geometry = [Point(xy) for xy in zip(data['Latitude'], data['Longitude'])]
    geo_data = gpd.GeoDataFrame(data, geometry=geometry)   
    world = gpd.read_file(geodatasets.data.naturalearth.land['url'])

    grid_size = 0.5  # degrees
    grid = build_mesh(grid_size)

    joined = gpd.sjoin(grid, geo_data, how='left')
    counts = joined.groupby(joined.index).size()
    grid['count'] = counts
    grid['count'] = grid['count'].fillna(0)

    grid_land = gpd.overlay(grid, world, how='intersection')
    grid_land['count'] = grid_land['count'].fillna(0)

    # Ensure CRS matches
    if geo_data.crs is None:
        gdf = geo_data.set_crs(epsg=4326)
    else:
        gdf = geo_data.to_crs(epsg=4326)    
    world = world.to_crs(epsg=4326)


    # ==============================================================
    # 1️⃣ Raw species sightings plot
    # ==============================================================

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    fig1.patch.set_facecolor("white")
    ax1.set_facecolor("#d7ebff")

    world.plot(ax=ax1, color='whitesmoke', edgecolor='gray', linewidth=0.5)
    gdf.plot(
        ax=ax1,
        marker='.',
        color='crimson',
        markersize=2,
        alpha=0.6,
    )

    ax1.set_title('Raw Species Sightings')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-90, 90)

    plt.tight_layout()
    plt.savefig("species_raw_plot.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # ==============================================================
    # 2️⃣ Grid-based density heatmap
    # ==============================================================

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    fig2.patch.set_facecolor("white")
    ax2.set_facecolor("#d7ebff")

    world.plot(ax=ax2, color='whitesmoke', edgecolor='none')

    grid_land.plot(
        ax=ax2,
        column='count',
        cmap='magma',
        norm=LogNorm(),
        legend=True,
        legend_kwds={'label': 'Sightings per cell (log scale)', 'shrink': 0.6},
        linewidth=0,
        alpha=0.9
    )

    world.boundary.plot(ax=ax2, color='black', linewidth=0.5)

    ax2.set_title(f'Species Sightings per {grid_size}° Grid Cell')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(-90, 90)

    plt.tight_layout()
    plt.savefig("species_grid_plot.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)