"""Download land and borough polygons (Overture Maps divisions) and build the H3 grid.

    uv run python build_grid.py

Writes data/land.parquet, data/boroughs.parquet, data/grid.parquet (land cells in the map square, clipped to land
and the square) and data/train_cells.parquet (the square plus a land buffer, full hexagons, used to fit Hex2Vec).
"""
import geopandas as gpd
import h3
import pandas as pd
from overturemaps import core
from shapely.geometry import Polygon

from config import (BBOX, BOROUGHS_PATH, DATA_DIR, GRID_PATH, H3_RES, HEX2VEC_BUFFER_M, LAND_PATH, PLOT_CRS,
                    TRAIN_CELLS_PATH, square_polygon)

# the five boroughs are coextensive with these counties
NYC_COUNTIES = {"New York County", "Kings County", "Queens County", "Bronx County", "Richmond County"}
NYC_BOROUGHS = {"Manhattan", "Brooklyn", "Queens", "The Bronx", "Bronx", "Staten Island"}


def primary_name(names):
    return names.get("primary") if isinstance(names, dict) else None


def land_only(areas):
    """Division areas clipped to the shoreline (the schema flag changed name across Overture releases)."""
    if "is_land" in areas.columns:
        return areas[areas["is_land"].fillna(False).astype(bool)]
    return areas[areas["class"] == "land"]


def select_boroughs(areas):
    areas = areas.assign(name=areas["names"].map(primary_name))
    in_ny = areas["region"] == "US-NY"
    counties = areas[in_ny & (areas["subtype"] == "county") & areas["name"].isin(NYC_COUNTIES)]
    if len(counties) == 5:
        return counties
    boroughs = areas[in_ny & (areas["subtype"] == "borough") & areas["name"].isin(NYC_BOROUGHS)]
    if len(boroughs) == 5:
        return boroughs
    candidates = areas[in_ny & areas["subtype"].isin(["county", "borough"])][["subtype", "name"]]
    raise ValueError(f"could not find the five boroughs, candidates:\n{candidates.to_string()}")


def to_wgs84(geometry):
    return gpd.GeoSeries([geometry], crs=PLOT_CRS).to_crs("EPSG:4326").iloc[0]


def cells_covering(geometry, res):
    """H3 cells whose centers fall inside geometry (EPSG:4326), as full hexagons."""
    ids = sorted(h3.geo_to_cells(geometry, res))
    polygons = [Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(cell)]) for cell in ids]
    return gpd.GeoDataFrame(geometry=polygons, index=pd.Index(ids, name="region_id"), crs="EPSG:4326")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    print("Downloading Overture division areas for", BBOX)
    areas = core.geodataframe("division_area", bbox=BBOX)
    areas = land_only(areas.set_crs("EPSG:4326", allow_override=True))

    # county areas follow legal boundaries that extend over water, so land comes from the state-level areas
    # Overture clips to the shoreline (is_territorial False; the territorial versions include water)
    states = areas[(areas["subtype"] == "region") & ~areas["is_territorial"].fillna(True).astype(bool)]
    print(f"Land from: {sorted(states['names'].map(primary_name))}")
    all_land = states.union_all()
    boroughs = select_boroughs(areas)[["name", "geometry"]].reset_index(drop=True)
    boroughs["geometry"] = boroughs.geometry.intersection(all_land)
    print(f"Boroughs: {sorted(boroughs['name'])}")

    square = square_polygon()
    square_wgs84 = to_wgs84(square.segmentize(500))
    land_geometry = all_land.intersection(square_wgs84)
    land = gpd.GeoDataFrame(geometry=[land_geometry], crs="EPSG:4326")

    grid = cells_covering(land_geometry, H3_RES)
    grid["geometry"] = grid.geometry.intersection(land_geometry)
    grid = grid[~grid.geometry.is_empty]
    lat_lon = [h3.cell_to_latlng(cell) for cell in grid.index]
    grid["lat"] = [lat for lat, _ in lat_lon]
    grid["lon"] = [lon for _, lon in lat_lon]

    train_area = all_land.intersection(to_wgs84(square.buffer(HEX2VEC_BUFFER_M, join_style="mitre")))
    train_cells = cells_covering(train_area, H3_RES)
    # make sure every map cell is in the training set
    train_cells = pd.concat([train_cells, cells_covering(land_geometry, H3_RES)])
    train_cells = train_cells[~train_cells.index.duplicated()]

    land.to_parquet(LAND_PATH)
    boroughs.to_parquet(BOROUGHS_PATH)
    grid.to_parquet(GRID_PATH)
    train_cells.to_parquet(TRAIN_CELLS_PATH)
    print(f"{len(grid)} map cells, {len(train_cells)} Hex2Vec training cells at H3 resolution {H3_RES}")


if __name__ == "__main__":
    main()
