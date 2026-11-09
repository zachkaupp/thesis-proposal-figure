"""Shared settings for the NYC embedding-similarity figure."""
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import box

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"

H3_RES = 9
# JFK Airport (lat, lon). The reference is the H3 cell containing this point.
REFERENCE_LATLON = (40.6413, -73.7781)

PLOT_CRS = "EPSG:32618"  # UTM 18N
# the map is a square in PLOT_CRS centered on the five boroughs (which span about 47 x 49 km)
SQUARE_CENTER_LATLON = (40.6975, -73.9800)
SQUARE_SIZE_M = 52000

# lon/lat box covering the square plus the Hex2Vec training buffer
BBOX = (-74.45, 40.40, -73.55, 41.02)  # (west, south, east, north)
HEX2VEC_BUFFER_M = 5000  # land within this distance of the square is also used to fit Hex2Vec

GRID_PATH = DATA_DIR / "grid.parquet"  # land cells in the square, geometry clipped to land and the square
TRAIN_CELLS_PATH = DATA_DIR / "train_cells.parquet"  # grid cells + buffer cells, full hexagons
LAND_PATH = DATA_DIR / "land.parquet"
BOROUGHS_PATH = DATA_DIR / "boroughs.parquet"


def square_bounds():
    """(left, bottom, right, top) of the map square in PLOT_CRS."""
    lat, lon = SQUARE_CENTER_LATLON
    x, y = Transformer.from_crs("EPSG:4326", PLOT_CRS, always_xy=True).transform(lon, lat)
    half = SQUARE_SIZE_M / 2
    return x - half, y - half, x + half, y + half


def square_polygon():
    return box(*square_bounds())


def embedding_path(name):
    return DATA_DIR / f"emb_{name}.parquet"
