from pathlib import Path
import sys
import tempfile
import unittest

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unify_inputs import prepare_counts, validate_embeddings


class UnifyInputsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / 'cells.h5ad'
        self.data = ad.AnnData(
            X=sparse.csr_matrix(np.array([[0.5, 1.5], [2.5, 3.5]], dtype='float32')),
            obs=pd.DataFrame({'type': ['A', 'B']}, index=['c1', 'c2']),
            var=pd.DataFrame(index=['G1', 'G2']))
        self.data.layers['counts'] = sparse.csr_matrix(np.array([[1, 2], [3, 4]], dtype='float32'))
        self.data.write_h5ad(self.path)
        self.pt = self.root / 'genes.pt'
        torch.save({'g1': torch.tensor([1., 2., 3.]), 'g2': torch.tensor([2., 3., 4.])}, self.pt)

    def test_counts_layer_and_source_preservation(self):
        before = self.path.read_bytes()
        prepared = prepare_counts(self.path, 'type', 'counts', 'gb')
        np.testing.assert_array_equal(prepared.X.toarray(), [[1, 2], [3, 4]])
        self.assertEqual(list(prepared.obs_names), ['gb:c1', 'gb:c2'])
        self.assertEqual(list(prepared.obs['unify_cell_type']), ['A', 'B'])
        self.assertEqual(before, self.path.read_bytes())

    def test_normalized_counts_rejected(self):
        with self.assertRaisesRegex(ValueError, '整数 counts'):
            prepare_counts(self.path, 'type', None, 'gb')

    def test_missing_labels_rejected(self):
        self.data.obs.loc['c1', 'type'] = None
        self.data.write_h5ad(self.path)
        with self.assertRaisesRegex(ValueError, '缺失值'):
            prepare_counts(self.path, 'type', 'counts', 'gb')

    def test_embedding_coverage_and_bad_values(self):
        report = validate_embeddings(self.pt, ['G1', 'G2'], 'gb')
        self.assertEqual(report['matched_genes'], 2)
        with self.assertRaisesRegex(ValueError, '完全不匹配'):
            validate_embeddings(self.pt, ['other'], 'gb')
        torch.save({'g1': torch.tensor([float('nan')])}, self.pt)
        with self.assertRaisesRegex(ValueError, '有限'):
            validate_embeddings(self.pt, ['G1'], 'gb')
