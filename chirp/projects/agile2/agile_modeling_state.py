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

"""Wrapper class for agile modeling steps."""


from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

from matplotlib import pyplot as plt
from ml_collections import config_dict

from chirp.projects.agile2 import audio_loader
from chirp.projects.agile2 import classifier
from chirp.projects.agile2 import classifier_data
from chirp.projects.agile2 import embedding_display
from chirp.projects.hoplite import brutalism
from chirp.projects.hoplite import score_functions
from chirp.projects.hoplite import sqlite_impl
from chirp.projects.zoo import models
import chirp.projects.agile2.convert_legacy as convert_legacy

import ipywidgets as widgets
from IPython.display import display

@dataclass
class agile2_config:
  db_path: str
  annotator_id: str
  baw_config: dict
  search_dataset_name: str
  embeddings_files: str = None

  def from_json(self, json_path):
    # read a json file and populate the dataclass properties from that

    if not Path(json_path).exists():
      Warning(f'Config file {json_path} does not exist.')

    def resolve_path(path):
      if not Path(path).is_absolute():
        path = Path(json_path).parent / Path(path)
      return path

    with open(json_path, 'r') as f:
      # paths in config is relative to the working directory, 
      # paths in json config is relative to the json file
      data = json.load(f)
      self.db_path = resolve_path(data['db_path'])
      self.annotator_id = data['annotator_id']
      self.baw_config = data['baw_config']
      self.search_dataset_name = data['search_dataset_name']
      if 'embeddings_files' in data:
        self.embeddings_files = resolve_path(data['embeddings_files'])
      self.check()


  def check(self):
    # check if the paths in the config exist
    if not Path(self.db_path).exists():
      Warning(f'DB path {self.db_path} does not exist.')


class agile2_state:

  config: agile2_config = None
  db: sqlite_impl.SQLiteGraphSearchDB = None
  db_model_config = None
  embed_config = None
  model_class = None
  embedding_model = None
  audio_filepath_loader = None

  # search results from the last search, whether it's a single example query
  # or a classifier-based search.
  search_results = None

  # an object representing the visual display of search_results
  # including result of interactive user-labeling of these results. 
  # this is overwritten each time a search results are displayed. 
  display_group = None

  def __init__(self, config):
    self.config = config
    self.init_db()
    self.init_loader()


  def init_db(self):
    print('Initializing DB...')
    self.db = sqlite_impl.SQLiteGraphSearchDB.create(self.config.db_path)
    self.db_model_config = self.db.get_metadata('model_config')
    self.embed_config = self.db.get_metadata('embed_config')
    self.model_class = models.model_class_map()[self.db_model_config.model_key]
    self.embedding_model = self.model_class.from_config(self.db_model_config.model_config)
    num_embeddings = self.db.count_embeddings()
    print(f'DB initialized with {num_embeddings} embeddings.')


  def init_loader(self):
    self.audio_filepath_loader = audio_loader.make_filepath_loader(
      audio_globs=self.embed_config.audio_globs,
      window_size_s=self.embedding_model.window_size_s,
      sample_rate_hz=self.embedding_model.sample_rate,
    )


  def embed_query(self, query_uri):
    """
    Embeds a single query audio clip allowing the user to select the 5s window within the specified source
    """
    query = embedding_display.QueryDisplay(
      uri=query_uri, offset_s=0.0, window_size_s=5.0, sample_rate_hz=32000)
    _ = query.display_interactive()
    self.query_embedding = self.embedding_model.embed(query.get_audio_window()).embeddings[0, 0]
  

  def search_with_query(self, query_label, num_results=50, sample_size=1_000_000, target_score=None):
    self.search(
        query=self.query_embedding,
        num_results=num_results,
        sample_size=sample_size,
        target_score=target_score)
    self.display_search_results(query_label)


  def search(self, query, bias=0.0, num_results=50, sample_size=1_000_000, target_score=None):
    """
    Searches the db using a query or model
    query: np.array of shape (embedding_dim,) (the result of self.embedding_model.embed())
           OR the result of classifier.train_linear_classifier() (kind of)
    """

    score_fn = score_functions.get_score_fn('dot', bias=bias, target_score=target_score)
    results, all_scores = brutalism.threaded_brute_search(
        self.db, query, num_results, score_fn=score_fn,
        sample_size=sample_size)
    self.search_results = results
    self.all_search_scores = all_scores

    # TODO(tomdenton): Better histogram when target sampling.
    _ = plt.hist(all_scores, bins=100)
    hit_scores = [r.sort_score for r in results.search_results]
    plt.scatter(hit_scores, np.zeros_like(hit_scores), marker='|',
                color='r', alpha=0.5)


  def display_search_results(self, query_label):
    self.display_group = embedding_display.EmbeddingDisplayGroup.from_search_results(
        self.search_results, self.db, sample_rate_hz=32000, frame_rate=100,
        audio_loader=self.audio_filepath_loader)
    
    self.display_group.baw_config = self.config.baw_config
    self.display_group.display(positive_labels=[query_label])


  def save_labels(self):
    prev_lbls, new_lbls = 0, 0
    for lbl in self.display_group.harvest_labels(self.config.annotator_id):
      row_count = self.db.insert_label(lbl, skip_duplicates=True)
      print(f'cursor.rowcount: {row_count}')
      check = True
      new_lbls += check
      prev_lbls += (1 - check)
    self.db.commit()
    print('\nnew_lbls: ', new_lbls)
    print('\nprev_lbls: ', prev_lbls)
    return prev_lbls, new_lbls
  
  
  def train_classifier(self, 
                       target_labels=None,
                       learning_rate = 1e-3,
                       weak_neg_weight = 0.05,
                       l2_mu = 0.000,
                       num_steps = 128,
                       train_ratio = 0.9,
                       batch_size = 128,
                       weak_negatives_batch_size = 128,
                       loss_fn_name = 'bce'):



    self.data_manager = classifier_data.AgileDataManager(
        target_labels=target_labels,
        db=self.db,
        train_ratio=train_ratio,
        min_eval_examples=1,
        batch_size=batch_size,
        weak_negatives_batch_size=weak_negatives_batch_size,
        rng=np.random.default_rng(seed=5))
    print('Training for target labels : ')
    print(self.data_manager.get_target_labels())
    params, eval_scores = classifier.train_linear_classifier(
        data_manager=self.data_manager,
        learning_rate=learning_rate,
        weak_neg_weight=weak_neg_weight,
        l2_mu=l2_mu,
        num_train_steps=num_steps,
        loss_name=loss_fn_name,
    )
    print('\n' + '-' * 80)
    top1 = eval_scores['top1_acc']
    print(f'top-1      {top1:.3f}')
    rocauc = eval_scores['roc_auc']
    print(f'roc_auc    {rocauc:.3f}')
    cmap = eval_scores['cmap']
    print(f'cmap       {cmap:.3f}')
    self.classifier_params = params
    self.classifier_eval_scores = eval_scores


  def search_with_classifier(self,
      target_label, num_results=50, sample_size=1_000_000, target_score=None):
    
    target_label_idx = self.data_manager.get_target_labels().index(target_label)
    class_query = self.classifier_params['beta'][:, target_label_idx]
    bias = self.classifier_params['beta_bias'][target_label_idx]

    self.search(
        query=class_query,
        bias=bias,
        num_results=num_results,
        sample_size=sample_size,
        target_score=target_score)
    
    self.display_search_results(target_label)


  def create_database(self, embeddings_files):
    """
    Checks if a database already exists at the specified path, and if not, creates one from the files at the location
    specified. If it does allows the user to delete and then creates the database. 
    """

    db_path = Path(self.config.db_path)

    def create_db():

      print(f'creating db at {db_path.resolve()}')
      db = convert_legacy.convert_parquet(embeddings_files, "sqlite", 
                                  self.config.search_dataset_name, max_count=10000, 
                                  db_path=db_path)

    if db_path.exists():

      print(f"DB path already exists at {db_path.resolve()}.")

      button = widgets.Button(description="Delete Existing Database and create it again?")
      def on_button_click(b):
          db_path.unlink()
          (db_path.parent / Path(self.config.db_path.name + "-shm")).unlink()
          (db_path.parent / Path(self.config.db_path.name + "-wal")).unlink()
          create_db()
      button.on_click(on_button_click)
      display(button)

    else:
      create_db()

