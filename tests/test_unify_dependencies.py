import unittest

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse


class UnifyDependenciesTest(unittest.TestCase):
    def test_scanpy_louvain_evaluation_backend(self):
        # Exercise the actual evaluation backend, including its legacy imports.
        triangle = np.ones((3, 3)) - np.eye(3)
        adjacency = sparse.block_diag([triangle, triangle], format="csr")
        data = ad.AnnData(X=np.ones((6, 2), dtype=np.float32))
        sc.tl.louvain(data, adjacency=adjacency, random_state=0)
        labels = data.obs["louvain"]
        self.assertFalse(labels.isna().any())
        self.assertEqual(labels.iloc[:3].nunique(), 1)
        self.assertEqual(labels.iloc[3:].nunique(), 1)
        self.assertNotEqual(labels.iloc[0], labels.iloc[3])
