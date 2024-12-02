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
import pandas as pd
import requests
import zipfile
from tqdm import tqdm

from matplotlib import pyplot as plt
from ml_collections import config_dict

from chirp import audio_utils
from chirp.projects.agile2 import audio_loader
from chirp.projects.agile2 import classifier
from chirp.projects.agile2 import classifier_data
from chirp.projects.agile2 import embedding_display
from chirp.projects.hoplite import brutalism
from chirp.projects.hoplite import score_functions
from chirp.projects.hoplite import sqlite_impl
from chirp.projects.zoo import models
from chirp.projects.zoo import model_configs
import chirp.projects.agile2.convert_legacy as convert_legacy

import chirp.inference.baw_utils as baw_utils

import ipywidgets as widgets
from IPython.display import display

@dataclass
class agile2_config:

  # path to the sqlite db containing the embeddings
  db_path: str = None

  # annotator id to attach to labels
  annotator_id: str = None

  # config for the baw api 
  baw_config: dict = None

  # name of the dataset in the database we are working with
  search_dataset_name: str = None

  # path to the embeddings files to create the database from
  embeddings_folder: str = None

  # path to labeled examples
  labeled_examples_folder: str = None
  
  # path to the folder where we will save the classifiers
  models_folder: str = None

  # path to the folder where we will save the inference results
  predictions_folder: str = None

  # max number of embeddings to load from the embeddings_folder
  max_embeddings_count: int = -1

  def from_json(self, json_path = "./agile_config.json"):
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
      paths_to_resolve = ['db_path', 'embeddings_folder', 'labeled_examples_folder']
      for key, value in data.items():
        if key in paths_to_resolve:
          value = resolve_path(value)
        setattr(self, key, value)

    self.check()
    self.show_config()

  def show_config(self, conf_dict=None, depth=0):
    if depth == 0:
      print(f'Config:')
    if conf_dict is None:
      conf_dict = self.__dict__
    
    for key, value in conf_dict.items():
      # if value is a dict, recurse
      if isinstance(value, dict):
        print(f'{key}:')
        self.show_config(value, depth+1)
      else:
        if key in ['auth_token']:
          # replace all but the 1st and last characters with *
          value = value[0] + '*' * (len(value) - 2) + value[-1]
        print(f'{key}: {value}')


  def check(self):
    # check if the paths in the config exist
    if not Path(self.db_path).exists():
      Warning(f'DB path {self.db_path} does not exist.')


class agile2_state:

  config: agile2_config = None
  db: sqlite_impl.SQLiteGraphSearchDB = None
  db_model_config = None
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


  def initialize(self):
    self.init_db()

  def init_db(self):
    print('Initializing DB...')
    self.db = sqlite_impl.SQLiteGraphSearchDB.create(self.config.db_path)
    self.db_model_config = self.db.get_metadata('model_config')
    
    self.model_class = model_configs.MODEL_CLASS_MAP[self.db_model_config.model_key]
    self.embedding_model = self.model_class.from_config(self.db_model_config.model_config)
    num_embeddings = self.db.count_embeddings()
    print(f'DB initialized with {num_embeddings} embeddings.')

    #TODO: This is a hack to get the audio loader to be created even when it's not used.  
    # seems that embed config metadata is not inserted during legacy conversion. 
    # it's wrapped in the legacy_config object instead. 
    # In the case of ecosounds recordings, this filepath loader is not really used, but
    # it needs to exist to pass to the display_group object.

    try:
      embed_config = self.db.get_metadata('embed_config')
    except KeyError:
      embed_config = config_dict.ConfigDict()
      embed_config.audio_globs = ['**/*.wav']

    self.audio_filepath_loader = audio_loader.make_filepath_loader(
      audio_globs=embed_config.audio_globs,
      window_size_s=self.embedding_model.window_size_s,
      sample_rate_hz=self.embedding_model.sample_rate,
    )


  def display_query(self, query_uri: str | int): 
    """
    Displays the query audio so that the user can select the 5s window to embed
    """

    query_uri = str(query_uri)

    # if the query is a path relative to the self.conifg.labeled_examples_folder, resolve its full path
    if (self.config.labeled_examples_folder / Path(query_uri)).exists():
      query_uri = self.config.labeled_examples_folder / Path(query_uri)

    # if the query is an integer it's an index into the list of audio files in the labeled_examples_folder
    elif str(query_uri).isdigit():
      file_list =  Helpers.list_audio_files(self.config.labeled_examples_folder, quiet=True)
      if int(query_uri) < len(file_list):
        query_uri = file_list[int(query_uri)]
      else:
        raise ValueError(f'Index {query_uri} is out of range of the labeled examples folder')
       
    self.query_display = embedding_display.QueryDisplay(
      uri=query_uri, offset_s=0.0, window_size_s=5.0, sample_rate_hz=32000)
    _ = self.query_display.display_interactive()

  
  def embed_query(self):
    """
    Embeds a single query audio clip allowing the user to select the 5s window within the specified source
    """

    self.query_embedding = self.embedding_model.embed(self.query_display.get_audio_window()).embeddings[0, 0]


  def add_labeled_examples(self, labeled_examples_folder, label, dataset_name, sample_rate_hz = 32000):
    """
    Adds labeled examples to the database from a folder containing audio files
    """

    window_size_s = 5.0

    labeled_examples_folder = Path(labeled_examples_folder)
    labeled_examples = Helpers.list_audio_files(labeled_examples_folder)

    for example_source in labeled_examples:

      print(f'Adding {label} example to db:{dataset_name}: {example_source}')

      audio = audio_utils.load_audio(example_source, sample_rate_hz)
      # choose the middle 5s of the audio
      start = int(max(len(audio)//2 - sample_rate_hz * window_size_s // 2, 0))
      end = int(start + sample_rate_hz * window_size_s) # python handles index past the end
      audio = audio[start: end]
  
      embedding = self.embedding_model.embed(audio).embeddings[0, 0]
      self.add_query_to_db(dataset_name, embedding, example_source, label)
      print('done')
      
     


  def add_query_to_db(self, dataset_name, embedding, source, query_label):
     
     #TOOD: do we also need to add the config, so that it can construct a full path from config
     # base path and file id?
     source = str(source)
     from chirp.projects.hoplite import interface
     source = interface.EmbeddingSource(dataset_name=dataset_name, source_id=source, offsets=np.array(0.0))
     embedding_id = self.db.insert_embedding(embedding, source)
     label = interface.Label(embedding_id, query_label, type = interface.LabelType.POSITIVE, provenance=self.config.annotator_id)
     self.db.insert_label(label, skip_duplicates=True)
     self.db.commit()


  def search_with_query(self, num_results=50, sample_size=1_000_000, target_score=None):
    self.search(
        query=self.query_embedding,
        num_results=num_results,
        sample_size=sample_size,
        target_score=target_score,
        dataset=self.config.search_dataset_name)
   



  def search(self, query, bias=0.0, num_results=50, sample_size=1_000_000, target_score=None, dataset=None):
    """
    Searches the db using a query or model
    query: np.array of shape (embedding_dim,) (the result of self.embedding_model.embed())
           OR the result of classifier.train_linear_classifier() (kind of)
    """

    score_fn = score_functions.get_score_fn('dot', bias=bias, target_score=target_score)
    results, all_scores = brutalism.threaded_brute_search(
        self.db, query, num_results, score_fn=score_fn,
        sample_size=sample_size, dataset=dataset)
    self.search_results = results
    self.all_search_scores = all_scores

    # TODO(tomdenton): Better histogram when target sampling.
    _ = plt.hist(all_scores, bins=100)
    hit_scores = [r.sort_score for r in results.search_results]
    plt.scatter(hit_scores, np.zeros_like(hit_scores), marker='|',
                color='r', alpha=0.5)
    plt.show()


  def display_search_results(self, query_label):
    
    plt.figure() 
    self.display_group = embedding_display.EmbeddingDisplayGroup.from_search_results(
        self.search_results, self.db, sample_rate_hz=32000, frame_rate=100,
        audio_loader=self.audio_filepath_loader)
    
    self.display_group.baw_config = self.config.baw_config
    self.display_group.display(positive_labels=[query_label])


  def save_labels(self):
    prev_lbls, new_lbls = 0, 0
    for lbl in self.display_group.harvest_labels(self.config.annotator_id):
      row_count = self.db.insert_label(lbl, skip_duplicates=True)
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
                       num_steps = 128,
                       train_ratio = 0.9,
                       batch_size = 128,
                       weak_negatives_batch_size = 128):



    data_manager = classifier_data.AgileDataManager(
        target_labels=target_labels,
        db=self.db,
        train_ratio=train_ratio,
        min_eval_examples=1,
        batch_size=batch_size,
        weak_negatives_batch_size=weak_negatives_batch_size,
        rng=np.random.default_rng(seed=5))
    print('Training for target labels : ')
    print(data_manager.get_target_labels())
    linear_classifier, eval_scores = classifier.train_linear_classifier(
        data_manager=data_manager,
        learning_rate=learning_rate,
        weak_neg_weight=weak_neg_weight,
        num_train_steps=num_steps,
    )
    print('\n' + '-' * 80)
    top1 = eval_scores['top1_acc']
    print(f'top-1      {top1:.3f}')
    rocauc = eval_scores['roc_auc']
    print(f'roc_auc    {rocauc:.3f}')
    cmap = eval_scores['cmap']
    print(f'cmap       {cmap:.3f}')
    # self.classifier_params = params
    # self.classifier_eval_scores = eval_scores

    self.classifier = linear_classifier

    # self.wrapped_classifier = {
    #    'params': params,
    #    'eval_scores': params,
    #    'labels': data_manager.get_target_labels(),
    # }

  def save_search_results(self, path, append=False):
    rows = []
    for res in self.search_results.search_results:
      source = self.db.get_embedding_source(res.embedding_id)
      domain, arid = baw_utils.extract_arid_and_domain(source.source_id)
      offset = float(source.offsets[0])
      audio_url = baw_utils.make_baw_audio_url_from_arid(arid, offset, 5.0, domain)
      #source = self.db.get_source(embedding.source_id)
      row = {
        'embedding_id': res.embedding_id,
        'arid': arid,
        'offset': offset,
        'score': res.sort_score,
        'link': audio_url
      }
      rows.append(row)

    df = pd.DataFrame(rows)

    if append:
      df.to_csv(path, mode='a', header=False, index=False)
    else:
      df.to_csv(path, index=False)


       




  def search_with_classifier(self,
      target_label, num_results=50, sample_size=1_000_000, target_score=None, dataset=None):
    target_labels = self.classifier.classes
    print('target_labels: ', target_labels)
    target_label_idx = target_labels.index(target_label)
    class_query = self.classifier.beta[:, target_label_idx]
    bias = self.classifier.beta_bias[target_label_idx]
    

    self.search(
        query=class_query,
        bias=bias,
        num_results=num_results,
        sample_size=sample_size,
        target_score=target_score, 
        dataset=dataset)
    
 


  def create_database(self, embeddings_files):
    """
    Checks if a database already exists at the specified path, and if not, creates one from the files at the location
    specified. If it does allows the user to delete and then creates the database. 
    """

    db_path = Path(self.config.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    def create_db():

      if (Path(embeddings_files) / 'filelist.json').exists():
        parquet_filepaths = [Path(embeddings_files) / 'embeddings' / Path(str(f['site_id'])) / Path(str(f['id'])) / 'embeddings.parquet' 
                         for f in json.load(open(Path(embeddings_files) / 'filelist.json'))]
      else:
        parquet_filepaths = None

      if self.config.max_embeddings_count:
        max_count = self.config.max_embeddings_count
      else:
        max_count = -1

      print(f'creating db at {db_path.resolve()}')
      db = convert_legacy.convert_parquet(parquet_folder = Path(embeddings_files),
                                          parquet_filepaths = parquet_filepaths, 
                                          db_type = "sqlite", 
                                          dataset_name = self.config.search_dataset_name, 
                                          max_count=max_count, 
                                          db_path=db_path)
      print(f'db created at {db_path.resolve()} with {db.count_embeddings()} embeddings.')

    if db_path.exists():

      print(f"DB path already exists at {db_path.resolve()}.")

      button = widgets.Button(description="Delete Existing Database and create it again?",
                              layout=widgets.Layout(width='auto'))
      def on_button_click(b):
          db_path.unlink()
          (db_path.parent / Path(db_path.name + "-shm")).unlink()
          (db_path.parent / Path(db_path.name + "-wal")).unlink()
          create_db()
      button.on_click(on_button_click)
      display(button)

    else:
      create_db()


  def save_classifier(self, path):
     Path(path).parent.mkdir(parents=True, exist_ok=True)
     self.classifier.save(path)
     

  def run_inference(self, output_filepath, threshold=0.0, labels=None, dataset=None):
     Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
     classifier.write_inference_csv(self.classifier, self.db, output_filepath, threshold, labels, dataset)
     
    


def download_embeddings(dataset_name, embeddings_dir):
    """
    Downloads a zip file from a url based on the dataset_name and extracts it to the embeddings_dir
    Shows progress for both download and extraction
    """

    def has_embeddings(directory):
        directory = Path(directory)
        # Return True as soon as we find any .pt file
        try:
            next(directory.glob('**/*.parquet'))
            return True
        except StopIteration:
            return False
     
    def do_download():
        
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        #dowload config.json
        url = f'https://api.ecosounds.org/system/esa2024/embedding_config.json'
        config_path = Path(embeddings_dir) / "config.json"
        response = requests.get(url)
        with open(config_path, 'wb') as file:
            file.write(response.content)
        
        url = f'https://api.ecosounds.org/system/esa2024/{dataset_name}/embeddings.zip'
        
        download_file(url, zip_path, description=f'Downloading {dataset_name}')

    def do_unzip():

        # Extract with progress bar
        with zipfile.ZipFile(zip_path) as zf:
            for member in tqdm(zf.infolist(), desc=f'Extracting {dataset_name}'):
                zf.extract(member, embeddings_dir)
        
        # Optionally remove zip after extraction
        zip_path.unlink()

    def do_download_and_unzip():
        do_download()
        do_unzip()



    def download_again_button():
        button1 = widgets.Button(description="Download again and overwrite existing embeddings?",
                                 layout=widgets.Layout(width='auto'))
        button1.on_click(lambda b: do_download_and_unzip())
        display(button1)

    def unzip_again_button():
        button2 = widgets.Button(description="Unzip existing downloaded file and overwrite embeddings?",
                                 layout=widgets.Layout(width='auto'))
        button2.on_click(lambda b: do_unzip())
        display(button2)
        

    embeddings_dir = Path(embeddings_dir)
    zip_path = Path(embeddings_dir) / f"{dataset_name}.zip"

    already_unzipped = has_embeddings(embeddings_dir)
    already_downloaded = zip_path.exists()

    if already_unzipped:

      print(f"Embeddings already downloaded and extracted at {embeddings_dir.resolve()}.")
      download_again_button()

      if already_downloaded:
        # might happen if they unterrupt during unzip
        unzip_again_button()

    elif already_downloaded:
      # don't think this can happen unless we change to not remove the zip after extraction
      print(f"Embeddings already downloaded but not unzipped.")
      unzip_again_button()

    else:
      do_download()
      do_unzip()




def download_file(url: str, dest: str | Path, max_retries: int = 5, description: str = None) -> bool:
    """
    Download a file from a URL to a destination path with progress bar and resume capability.
    
    Args:
        url: URL to download from
        dest: Destination path (string or Path object)
        max_retries: Maximum number of retry attempts (default: 5)
        description: Optional description for the progress bar. If None, uses filename
    
    Returns:
        bool: True if download was successful
    
    Raises:
        Exception: If download fails after all retries
    """

    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    from tqdm import tqdm
    import time

    dest = Path(dest)
    description = description or dest.name
    
    # Set up retry strategy
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    # Configure session
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate'
    }
    
    for attempt in range(max_retries):
        try:
            # Check file size and resume support
            head = session.head(url, headers=headers)
            total_size = int(head.headers.get('content-length', 0))
            supports_resume = 'accept-ranges' in head.headers
            
            existing_size = 0
            if dest.exists():
                existing_size = dest.stat().st_size
                if existing_size == total_size:
                    print(f"File already completely downloaded ({existing_size} bytes)")
                    return True
                elif supports_resume and existing_size < total_size:
                    print(f"Resuming download from byte {existing_size}")
                    headers['Range'] = f'bytes={existing_size}-'
                else:
                    print("Cannot resume download - starting from beginning")
                    existing_size = 0
            
            # Create parent directories if they don't exist
            dest.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress bar
            response = session.get(
                url,
                stream=True,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            mode = 'ab' if existing_size else 'wb'
            chunk_size = 8 * 1024 * 1024  # 8MB chunks
            
            with open(dest, mode) as file, tqdm(
                desc=description,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                initial=existing_size
            ) as pbar:
                for data in response.iter_content(chunk_size=chunk_size):
                    size = file.write(data)
                    pbar.update(size)
            
            # Verify complete download
            if dest.stat().st_size == total_size:
                print("Download completed successfully")
                return True
            else:
                print(f"Download incomplete, retrying... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(2)
                continue
                
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Download failed, retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                print(f"Error: {str(e)}")
                time.sleep(wait_time)
                continue
            else:
                raise Exception(f"Failed to download after {max_retries} attempts: {str(e)}")
    
    raise Exception("Download failed after all retry attempts")

class Helpers:

  @staticmethod
  def list_audio_files(path, recursive=True, quiet=False, extensions=(".wav", ".flac", ".mp3")):
      path = Path(path)
      files = path.rglob("*") if recursive else path.glob("*")
      audio_files = [
          f for f in files 
          if f.is_file()
          and f.suffix.lower() in extensions
          and not f.name.startswith(".")
      ]

      if not quiet:
          print(f"Found {len(audio_files)} audio files in {str(path)}")
          print("\n".join([f"{i}: {f}" for i, f in enumerate(audio_files)]))
      return audio_files
  

