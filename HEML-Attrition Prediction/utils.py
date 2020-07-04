import pandas as pd 

def OverSampleReplication(df,col=None,minority_class=None):
    ## Source: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#oversample_the_minority_class
    ## Code modified as required to avoid redundancy and to handle pandas dataframe objects
    if col is None:
        raise Warning('Column value not passed')
    if col is None:
        raise Warning('minority_class value not passed')
    bool_train_labels = df[col]==minority_class
    pos_features = df[bool_train_labels]
    neg_features = df[~bool_train_labels]

    pos_labels = df[bool_train_labels]
    neg_labels = df[~bool_train_labels]
    ids = np.arange(len(pos_features))
    choices = np.random.choice(ids, len(neg_features))

    res_pos_features = pos_features.iloc[choices]
    resampled_features = pd.concat([res_pos_features, neg_features], axis=0)
    order = np.arange(len(resampled_features))
    np.random.shuffle(order)
    resampled_features = resampled_features.iloc[order]
   
    return resampled_features.drop([col],axis=1),resampled_features[col]