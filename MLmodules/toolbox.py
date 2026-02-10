from global_import import torch, pd, np

def series_to_tensor(series: pd.Series, dtype=torch.float32, device='cpu') -> torch.Tensor:
    """
    Converts a pandas Series to a PyTorch tensor.
    """
    values = series.values
    # Handle object/string types if necessary? Usually for numeric series.
    # If categorical, it should probably be encoded first.
    return torch.tensor(values, dtype=dtype, device=device)

def tensor_to_series(tensor: torch.Tensor, index=None, name=None) -> pd.Series:
    """
    Converts a PyTorch tensor to a pandas Series.
    """
    return pd.Series(tensor.detach().cpu().numpy(), index=index, name=name)

def get_embeddings_from_series(series: pd.Series, embedding_layer: torch.nn.Embedding, encoder=None) -> torch.Tensor:
    """
    Given a categorical pd.Series, returns the embeddings.
    If an encoder (LabelEncoder) is provided, it transforms the series first.
    Otherwise assumes series is already numerical indices.
    """
    if encoder:
        # Handle unknonwns/NaNs if necessary? Assumed handled before.
        indices = encoder.transform(series)
    else:
        indices = series.values
        
    indices_tensor = torch.tensor(indices, dtype=torch.long, device=embedding_layer.weight.device)
    return embedding_layer(indices_tensor)

def groupby_to_tensor_pad(df: pd.DataFrame, group_col: str, feature_cols, padding_value=0, padding_mode='constant', device='cpu') -> torch.Tensor:
    """
    Groups a DataFrame by group_col and converts feature_cols into a padded tensor.
    If feature_cols is a list, returns 3D tensor (Batch, Seq, Features).
    If feature_cols is a string, returns 2D tensor (Batch, Seq).
    """
    groups = df.groupby(group_col)
    max_len = groups.size().max()
    
    batch_list = []
    
    is_1d = isinstance(feature_cols, str)
    
    for _, group in groups:
        # Ensure order if needed (passed df should be sorted)
        data = group[feature_cols].values
        pad_len = max_len - len(data)
        
        if pad_len > 0:
            pad_kwargs = {'mode': padding_mode}
            if padding_mode == 'constant':
                pad_kwargs['constant_values'] = padding_value
                
            if is_1d:
                # 1D array padding
                data = np.pad(data, (0, pad_len), **pad_kwargs)
            else:
                # 2D array padding (Seq, Feat)
                # For 2D, we typically only pad the sequence dimension (axis 0)
                # The second dimension (pad_len, 0) should be (0,0) for features
                data = np.pad(data, ((0, pad_len), (0, 0)), **pad_kwargs)
            
        batch_list.append(data)
        
    dtype = torch.long if is_1d else torch.float32
    return torch.tensor(np.array(batch_list), dtype=dtype, device=device)

