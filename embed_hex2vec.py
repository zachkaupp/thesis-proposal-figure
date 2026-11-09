"""Hex2Vec embeddings (OpenStreetMap tags) per H3 cell, fit with SRAI on the map square plus a land buffer.

    uv run python embed_hex2vec.py

The OSM extract is downloaded once into data/osm/. The encoder sizes, epochs and batch size follow the SRAI Hex2Vec
example and the original paper ([150, 75, 50]).
"""
import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from srai.embedders import Hex2VecEmbedder
from srai.joiners import IntersectionJoiner
from srai.loaders.osm_loaders import OSMPbfLoader
from srai.loaders.osm_loaders.filters import HEX2VEC_FILTER
from srai.neighbourhoods import H3Neighbourhood

from config import DATA_DIR, GRID_PATH, TRAIN_CELLS_PATH, embedding_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    regions = gpd.read_parquet(TRAIN_CELLS_PATH)
    map_cells = pd.read_parquet(GRID_PATH, columns=["lat"]).index
    area = gpd.GeoDataFrame(geometry=[regions.union_all()], crs=regions.crs)

    features = OSMPbfLoader(download_directory=str(DATA_DIR / "osm")).load(area, HEX2VEC_FILTER)
    print(f"{len(features)} OSM features in {len(regions)} training cells")
    joint = IntersectionJoiner().transform(regions, features)

    embedder = Hex2VecEmbedder([150, 75, 50])
    embeddings = embedder.fit_transform(
        regions, features, joint, H3Neighbourhood(regions),
        batch_size=args.batch_size,
        trainer_kwargs={"max_epochs": args.epochs, "accelerator": "cpu"},
    )
    embeddings = embeddings.loc[map_cells]
    embeddings.columns = [f"hex2vec_{i}" for i in range(embeddings.shape[1])]
    embeddings.to_parquet(embedding_path("hex2vec"))
    print(f"Saved {embeddings.shape} to {embedding_path('hex2vec')}")


if __name__ == "__main__":
    main()
