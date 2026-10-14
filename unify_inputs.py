"""Unify 输入计数和基因嵌入检查。"""
from pathlib import Path


def parse_pairs(items, name):
    result = {}
    for item in items:
        sid, sep, value = item.partition('=')
        if not sep or not sid.isalnum() or sid in result or not value:
            raise ValueError(f'{name}: 需要不重复的物种 ID=文件路径（ID 仅含字母/数字）')
        path = Path(value).resolve()
        if not path.is_file():
            raise ValueError(f'文件不存在：{path}')
        result[sid] = path
    return result


def prepare_counts(source, label_key, counts_layer, sid):
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse
    data = ad.read_h5ad(source)
    if not data.obs_names.is_unique or not data.var_names.is_unique:
        raise ValueError(f'{sid}: 细胞/基因 ID 必须唯一')
    if not data.n_obs or not data.n_vars:
        raise ValueError(f'{sid}: 空数据集')
    # 官方 Unify 按小写基因 ID 匹配嵌入。
    if not pd.Index(data.var_names.str.lower()).is_unique:
        raise ValueError(f'{sid}: 基因 ID 转为小写后有重复')
    if label_key not in data.obs:
        raise ValueError(f'{sid}: 缺少细胞类型列 {label_key!r}')
    labels = data.obs[label_key]
    if labels.isna().any() or labels.astype(str).str.strip().eq('').any():
        raise ValueError(f'{sid}: 细胞类型标签含缺失值')
    if counts_layer and counts_layer not in data.layers:
        raise ValueError(f'{sid}: 缺少 counts layer {counts_layer!r}')
    counts = data.layers[counts_layer] if counts_layer else data.X
    values = counts.data if sparse.issparse(counts) else np.asarray(counts)
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.allclose(values, np.rint(values), rtol=0, atol=1e-6)):
        raise ValueError(f'{sid}: 必须使用非负原始整数 counts')
    if (np.asarray(counts.sum(axis=1)).ravel() == 0).any():
        raise ValueError(f'{sid}: 请先去除零 counts 细胞')
    obs = pd.DataFrame(index=[f'{sid}:{cell}' for cell in data.obs_names])
    obs['unify_cell_type'] = pd.Categorical(labels.astype(str).to_numpy())
    obs['source_cell_id'] = data.obs_names.to_numpy(dtype=object)
    obs['source_species'] = sid
    return ad.AnnData(X=sparse.csr_matrix(counts), obs=obs,
                      var=pd.DataFrame(index=data.var_names.to_numpy(dtype=object)))


def validate_embeddings(path, genes, sid):
    import torch
    # 只加载 tensor 字典，不执行任意 pickle 对象。
    embeddings = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(embeddings, dict) or not embeddings:
        raise ValueError(f'{sid}: 嵌入必须为非空 gene_id -> Tensor 字典')
    lowered = set()
    dim = None
    for gene, vector in embeddings.items():
        if not isinstance(gene, str) or gene.lower() in lowered:
            raise ValueError(f'{sid}: 嵌入基因 ID 无效或小写后重复')
        lowered.add(gene.lower())
        if (not isinstance(vector, torch.Tensor) or vector.ndim != 1
                or not vector.is_floating_point() or not torch.isfinite(vector).all()):
            raise ValueError(f'{sid}/{gene}: 嵌入必须为有限的一维浮点 Tensor')
        if dim is None:
            dim = vector.numel()
        if not dim or vector.numel() != dim:
            raise ValueError(f'{sid}: 嵌入维度不一致')
    matched = sum(g.lower() in lowered for g in genes)
    if not matched:
        raise ValueError(f'{sid}: 表达基因与基因嵌入完全不匹配')
    print(f'{sid}: {matched}/{len(genes)} 个表达基因匹配嵌入，维度 {dim}')
    return {'matched_genes': int(matched), 'total_genes': len(genes), 'embedding_dim': dim}
