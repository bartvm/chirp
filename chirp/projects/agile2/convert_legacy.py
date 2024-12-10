# coding=utf-8
# Copyright 2024 The Perch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Conversion for TFRecord embeddings to Hoplite DB."""

import os
from typing import Tuple
from chirp.inference import embed_lib
from chirp.inference import tf_examples
from chirp.projects.agile2 import embed
from chirp.projects.agile2 import source_info
from chirp.projects.hoplite import in_mem_impl
from chirp.projects.hoplite import interface
from chirp.projects.hoplite import sqlite_impl
from etils import epath
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm


def convert_tfrecords(
    embeddings_path: str,
    db_type: str,
    dataset_name: str,
    max_count: int = -1,
    **kwargs,
):
  """Convert a TFRecord embeddings dataset to a Hoplite DB."""
  ds = tf_examples.create_embeddings_dataset(
    embeddings_path,
    'embeddings-*',
  )

  legacy_config = embed_lib.load_embedding_config(embeddings_path)

  return convert_tfdataset(
    ds,
    db_type,
    dataset_name,
    legacy_config,
    max_count,
    **kwargs,
  )
  

def convert_tfdataset(
    ds: str,
    db_type: str,
    dataset_name: str,
    legacy_config: dict,
    max_count: int = -1,
    **kwargs,
):
  """Convert a tf dataset to a Hoplite DB."""

  # Peek at one embedding to get the embedding dimension.
  for ex in ds.as_numpy_iterator():
    emb_dim = ex['embedding'].shape[-1]
    break
  else:
    raise ValueError('No embeddings found.')
  
  if db_type == 'sqlite':
    db_path = kwargs['db_path']
    if epath.Path(db_path).exists():
      raise ValueError(f'DB path {db_path} already exists.')
    db = sqlite_impl.SQLiteGraphSearchDB.create(db_path, embedding_dim=emb_dim)
  elif db_type == 'in_mem':
    db = in_mem_impl.InMemoryGraphSearchDB.create(
        embedding_dim=emb_dim,
        max_size=kwargs['max_size'],
        degree_bound=kwargs['degree_bound'],
    )
  else:
    raise ValueError(f'Unknown db type: {db_type}')

  # Convert embedding config to new format and insert into the DB.

  model_config = embed.ModelConfig(
      model_key=legacy_config.embed_fn_config.model_key,
      embedding_dim=emb_dim,
      model_config=legacy_config.embed_fn_config.model_config,
  )
  file_id_depth = legacy_config.embed_fn_config['file_id_depth']
  audio_globs = []
  for i, glob in enumerate(legacy_config.source_file_patterns):
    base_path, file_glob = glob.split('/')[-file_id_depth - 1 :]
    if i > 0:
      partial_dataset_name = f'{dataset_name}_{i}'
    else:
      partial_dataset_name = dataset_name
    audio_globs.append(
        source_info.AudioSourceConfig(
            dataset_name=partial_dataset_name,
            base_path=base_path,
            file_glob=file_glob,
            min_audio_len_s=legacy_config.embed_fn_config.min_audio_s,
            target_sample_rate_hz=legacy_config.embed_fn_config.get(
                'target_sample_rate_hz', -2
            ),
        )
    )

  audio_sources = source_info.AudioSources(audio_globs=tuple(audio_globs))
  db.insert_metadata('legacy_config', legacy_config)
  db.insert_metadata('audio_sources', audio_sources.to_config_dict())
  db.insert_metadata('model_config', model_config.to_config_dict())
  hop_size_s = model_config.model_config.hop_size_s
  
  total_size = ds.cardinality().numpy()
  for ex in tqdm.tqdm(ds.as_numpy_iterator(), total=total_size):
    embs = ex['embedding']
    flat_embeddings = np.reshape(embs, [-1, embs.shape[-1]])
    file_id = str(ex['filename'], 'utf8')
    offset_s = ex['timestamp_s']
    if max_count > 0 and db.count_embeddings() >= max_count:
      break
    for i in range(flat_embeddings.shape[0]):
      embedding = flat_embeddings[i]
      offset = np.array(offset_s + hop_size_s * i)
      source = interface.EmbeddingSource(dataset_name, file_id, offset)
      db.insert_embedding(embedding, source)
      if max_count > 0 and db.count_embeddings() >= max_count:
        break
  db.commit()
  num_embeddings = db.count_embeddings()
  print('\n\nTotal embeddings : ', num_embeddings)
  hours_equiv = num_embeddings / 60 / 60 * hop_size_s
  print(f'\n\nHours of audio equivalent : {hours_equiv:.2f}')
  return db




def convert_parquet(
    parquet_folder: str,
    db_type: str,
    dataset_name: str,
    parquet_filepaths: list = None,
    max_count: int = -1,
    prefetch: int = 128,
    source_map_fn = lambda x: x,
    **kwargs,
):
  """
  Convert a list of parquet files into a TF dataset so it can be converted to a hoplite DB.
  Requires a config json file in the parquet folder in the same format as the TF record config.

  @param parquet_folder str; path to the folder containing the parquet files
  @param db_type str; type of DB to create, sqlite or in_mem
  @param dataset_name str; name of the dataset (the database can contain multiple datasets)
  @param max_count int; maximum number of embeddings to convert
  @param prefetch int; number of elements to prefetch
  @param source_map_fn function; optionally modify the source before inserting into the DB
  """
  
  if parquet_filepaths is None:
    print("Collecting embeddings files...")
    parquet_filepaths = [f for f in Path(parquet_folder).rglob('*.parquet')]
    print("Found ", len(parquet_filepaths), " embeddings files.")

  def generator():
    ds = pyarrow.parquet.ParquetDataset(parquet_filepaths)
    for fragment in ds.fragments:
        try:
            df = fragment.to_table().to_pandas()
            n_channels = df['channel'].nunique()
            filename = df['source'].iloc[0]
            timestamp_s = 0.0  
            df.drop(['source', 'channel', 'offset'], axis=1, inplace=True)
            embeddings = df.to_numpy().reshape(-1, n_channels, 1280)
            filename = source_map_fn(filename)
            yield {
                'filename': filename.encode(),
                'timestamp_s': timestamp_s,
                'embedding': embeddings,
                'embedding_shape': embeddings.shape
            }
        except Exception as e:
            print(f"Unexpected error on {fragment}: {e}")
            continue
        
    
  output_signature = {
      'filename': tf.TensorSpec(shape=(), dtype=tf.string),
      'timestamp_s': tf.TensorSpec(shape=(), dtype=tf.float32),
      'embedding': tf.TensorSpec(shape=(None, None, 1280), dtype=tf.float32),
      'embedding_shape': tf.TensorSpec(shape=(3,), dtype=tf.int32)
  }

  # debug only
  # for item in generator():
  #   print("Generator first item: ")
  #   for key, value in item.items():
  #     print(key, ": ", value)
  #   break
    
  ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
  ds = ds.prefetch(prefetch)

  legacy_config = embed_lib.load_embedding_config(parquet_folder)
  
  return convert_tfdataset(
      ds,
      db_type,
      dataset_name,
      legacy_config,
      max_count,
      **kwargs,
  )
    

def extract_metadata(embeddings_table: pd.DataFrame, dtype = np.float32) -> Tuple[str, float, Tuple[int, int, int]]:
    """
    Extracts embeddings and metadata from the dataframe.
    @param df pd.DataFrame; DataFrame with metadata and embeddings
    @returns Tuple; embeddings array filename, timestamp, and embedding shape

    For now, we are assuming that each DataFrame corresponds to a single audio file starting at timestamp 0.0.
    """

    filename = embeddings_table[0][0][0]
    timestamp_s = 0.0  
    embedding_shape = embeddings_table.shape
    embeddings = embeddings_table[:, :, 2:1282].astype(dtype)
    return embeddings, filename, timestamp_s, embedding_shape


def df_to_embeddings(df: pd.DataFrame) -> np.array:

    """
    Converts a dataframe (tabular format) of embeddings to a 3D array (offset, channel, feature) format.
    @param df pd.DataFrame; DataFrame with (n_segments * n_channels) rows and n_features + 2 columns (including offset and channel)
    @returns np.array; array of shape (n_segment, n_channels, n_features + 2)

    The reason for n_features + 2 is that it includes 'filename' and 'offset' columns
    """

    # Determine the number of channels and features
    n_channels = df['channel'].nunique()
    n_features = len(df.columns) - 2  # Subtracting the offset and channel columns

    # Sort the DataFrame based on 'offset' and 'channel' to ensure correct ordering
    df_sorted = df.sort_values(by=['offset', 'channel'])

    # Drop the 'channel' column and pivot the DataFrame to get the correct shape
    df_pivot = df_sorted.drop('channel', axis=1)
    reshaped_array = df_pivot.values.reshape(-1, n_channels, n_features + 1)

    return reshaped_array
