import sys
import os
bit_path = os.path.join("big_vision")
sys.path.append(scenic_path)
  
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
from datasets import load_dataset, load_from_disk

from owl_vit.scenic.scenic.projects.owl_vit import models
from owl_vit.scenic.scenic.projects.owl_vit import trainer
from owl_vit.scenic.scenic.projects.owl_vit.configs.trained_clip_for_owlvit import get_config

#FLAGS = flags.FLAGS


def main(config: ml_collections.ConfigDict, workdir: str, \
         model_storage_path : str, dataset_dir: str, dataset_format=None):
  """Main funtion for OWL-ViT training."""

  if config.checkpoint:
    # When restoring from a checkpoint, change the dataset seed to ensure that
    # the example order is new:
    train_state = checkpoints.restore_checkpoint(workdir, target=None)
    if train_state is not None:
      global_step = train_state.get('global_step', 0)
      #logging.info('Folding global_step %s into dataset seed.', global_step)

  dataset = None
  if dataset_format is None:
    dataset = load_from_dist(dataset_dir)
  else:
    dataset = load_dataset(dataset_format, data_files=dataset_dir)

  trainer.train(
      config=config,
      model_cls=models.TextZeroShotDetectionModel,
      dataset=dataset,
      workdir=workdir)

  trainer.save_model(model_storage_path)


if __name__ == '__main__':
  main(config=get_config, workdir=os.path.join(scenic_path, "projects", "owl_vit", "checkpoint"), \
       dataset_dir=os.path.join("datasets", "CN_coin_annotations", "coin_annotations.json"), \
       dataset_format='json')

