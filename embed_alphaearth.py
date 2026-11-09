"""AlphaEarth Foundations embeddings per H3 cell, from the public Cloud Optimized GeoTIFFs (no Earth Engine account).

    uv run python embed_alphaearth.py
    uv run python embed_alphaearth.py --factor 2   # finer pixels, about 4x the download

Reads an overview of each annual embedding file (default factor 4, 40 m pixels) over the NYC box, assigns every pixel
to the H3 cell containing its center, and averages the de-quantized vectors per cell (renormalized to unit length).
A resolution-9 cell (about 0.1 km^2) holds roughly 65 pixels at 40 m.
"""
import argparse
import urllib.request

import h3
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

from config import BBOX, DATA_DIR, GRID_PATH, H3_RES, embedding_path

BUCKET = "https://storage.googleapis.com/alphaearth_foundations/"
INDEX_URL = BUCKET + "satellite_embedding/v1/annual/aef_index.parquet"
UTM_CRS = "EPSG:32618"
NODATA = -128
NUM_BANDS = 64
GDAL_OPTIONS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tiff",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_TIMEOUT": "120",
}


def load_index():
    path = DATA_DIR / "aef_index.parquet"
    if not path.exists():
        print(f"Downloading {INDEX_URL}")
        urllib.request.urlretrieve(INDEX_URL, path)
    return pd.read_parquet(path)


def dequantize(values):
    values = values.astype(np.float32)
    return (values / 127.5) ** 2 * np.sign(values)


def read_window(url, factor, bounds):
    """int8 array (bands x rows x cols) and its affine transform for the overview window covering bounds (UTM)."""
    with rasterio.Env(**GDAL_OPTIONS):
        with rasterio.open(url) as src:
            factors = src.overviews(1)
        if factor not in factors:
            raise ValueError(f"overview factor {factor} not in {factors}")
        with rasterio.open(url, overview_level=factors.index(factor)) as src:
            inverse = ~src.transform
            left, bottom, right, top = bounds
            cols, rows = zip(*(inverse * corner for corner in [(left, bottom), (left, top), (right, bottom), (right, top)]))
            c0, c1 = max(int(np.floor(min(cols))), 0), min(int(np.ceil(max(cols))), src.width)
            r0, r1 = max(int(np.floor(min(rows))), 0), min(int(np.ceil(max(rows))), src.height)
            if c0 >= c1 or r0 >= r1:
                return None, None
            window = Window(c0, r0, c1 - c0, r1 - r0)
            return src.read(window=window), src.window_transform(window)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=int, default=4, help="overview factor, pixels are 10 m x this")
    parser.add_argument("--year", type=int, default=None, help="default: latest year in the index")
    args = parser.parse_args()

    cells = pd.read_parquet(GRID_PATH, columns=["lat"]).index
    cell_row = pd.Series(np.arange(len(cells)), index=cells)

    to_utm = Transformer.from_crs("EPSG:4326", UTM_CRS, always_xy=True)
    to_lonlat = Transformer.from_crs(UTM_CRS, "EPSG:4326", always_xy=True)
    west, south, east, north = BBOX
    xs, ys = to_utm.transform([west, west, east, east], [south, north, south, north])
    bounds = (min(xs), min(ys), max(xs), max(ys))

    index = load_index()
    year = args.year or int(index["year"].max())
    files = index[(index["year"] == year) & (index["crs"] == UTM_CRS)
                  & (index["utm_west"] < bounds[2]) & (index["utm_east"] > bounds[0])
                  & (index["utm_south"] < bounds[3]) & (index["utm_north"] > bounds[1])]
    print(f"AlphaEarth year {year} (latest in index: {int(index['year'].max())}), {len(files)} files, "
          f"overview factor {args.factor} ({10 * args.factor} m pixels)")

    sums = np.zeros((len(cells), NUM_BANDS))
    counts = np.zeros(len(cells), dtype=np.int64)
    for path in files["path"]:
        url = path.replace("gs://", "https://storage.googleapis.com/")
        print(f"Reading {url}")
        data, transform = read_window(url, args.factor, bounds)
        if data is None:
            continue
        rows, cols = np.indices(data.shape[1:])
        x, y = transform * (cols.ravel() + 0.5, rows.ravel() + 0.5)
        lon, lat = to_lonlat.transform(x, y)
        pixel_cells = pd.Series([h3.latlng_to_cell(a, b, H3_RES) for a, b in zip(lat, lon)])
        target = pixel_cells.map(cell_row)
        values = data.reshape(NUM_BANDS, -1)
        keep = target.notna().to_numpy() & (values != NODATA).all(axis=0)
        target = target[keep].astype(int).to_numpy()
        np.add.at(sums, target, dequantize(values[:, keep]).T)
        np.add.at(counts, target, 1)

    covered = counts > 0
    print(f"{covered.sum()} of {len(cells)} cells have pixels (median {int(np.median(counts[covered]))} per cell)")
    means = sums[covered] / np.linalg.norm(sums[covered], axis=1, keepdims=True)
    embeddings = pd.DataFrame(means, index=cells[covered], columns=[f"aef_{i}" for i in range(NUM_BANDS)])
    embeddings.to_parquet(embedding_path("alphaearth"))
    print(f"Saved {embeddings.shape} to {embedding_path('alphaearth')}")


if __name__ == "__main__":
    main()
