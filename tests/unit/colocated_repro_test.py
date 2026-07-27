"""Hermetic reproduction test for colocated python dataloader process count mismatch.

Demonstrates that querying uncoordinated jax.process_count() during topology transitions
causes batch shape mismatches, whereas passing explicit topology from the validated mesh
guarantees correct per-host shard slicing.
"""

import unittest
from unittest.mock import patch
import numpy as np
import jax

from maxtext.input_pipeline.multihost_dataloading import RemoteIterator


class ColocatedProcessCountReproTest(unittest.TestCase):

  def test_remote_iterator_process_count_mismatch_repro(self):
    """Proves that uncoordinated jax.process_count() == 1 produces global batch shape mismatches."""
    global_batch_size = 128
    expected_slice_batch = 32  # For 4 hosts

    def mock_get_ds_fn(dataloading_host_index, dataloading_host_count):
      # If dataloading_host_count is 1 (stale single-process fallback), batch size is 128.
      # If dataloading_host_count is 4 (correct multi-host topology), batch size is 32.
      shard_size = global_batch_size // dataloading_host_count
      return [np.zeros((shard_size, 2048), dtype=np.float32) for _ in range(5)]

    def mock_preprocessing_fn(dataset):
      return dataset

    # Simulate stale jax.process_count() == 1 during sidecar topology transition
    with patch.object(jax, "process_count", return_value=1), \
         patch.object(jax, "process_index", return_value=0):
      stale_iterator = RemoteIterator(
          get_ds_fn=mock_get_ds_fn,
          preprocessing_fn=mock_preprocessing_fn,
          global_shape=(global_batch_size, 2048),
          checkpoint_path=None,
          elastic=True,
      )
      batch = next(stale_iterator.iterator)
      # Notice the shape mismatch: got (128, 2048) instead of expected (32, 2048)!
      self.assertEqual(batch.shape, (128, 2048))
      self.assertNotEqual(batch.shape, (expected_slice_batch, 2048))

    # Simulate passing explicit topology (process_count=4) from validated mesh
    with patch.object(jax, "process_count", return_value=4), \
         patch.object(jax, "process_index", return_value=0):
      synced_iterator = RemoteIterator(
          get_ds_fn=mock_get_ds_fn,
          preprocessing_fn=mock_preprocessing_fn,
          global_shape=(global_batch_size, 2048),
          checkpoint_path=None,
          elastic=True,
      )
      batch = next(synced_iterator.iterator)
      # Notice the shape matches exactly: (32, 2048)!
      self.assertEqual(batch.shape, (expected_slice_batch, 2048))


if __name__ == "__main__":
  unittest.main()
