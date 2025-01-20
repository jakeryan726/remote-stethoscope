from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

def kfold(model, train_ds, train_func, test_func, optimizer, device, epochs, batch_size, k=4):
    result = 0
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(train_ds):
        # Create Subsets
        train_subset = Subset(train_ds, train_idx)
        validate_subset = Subset(train_ds, val_idx)
        
        # Create DataLoader
        train_dl = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

        # Train and Test
        _ = train_func(model, train_dl, optimizer, device, epochs)
        result += test_func(model, validate_subset, device)
    return result / k