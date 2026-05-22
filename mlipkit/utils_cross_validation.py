def kfold_ind(size, k):
    '''
    Given the size of a dataset (size) and the number of folds (k), return a list containing the indices of the dataset
    of the last configuration of the corresponding fold. E.g. [11, 23] for a dataset of 34 confs means that
    there are three folds: the first is 0th-11th, the second fold is 12th-23rd and the last one is 24th-33rd. If size is not
    a mutiple of k, after distributing int(size/k) confs to each fold, an extra conf is given to the first
    size%k folds.
    Arguments:
    size(int): size of the dataset
    k(int): number of folds
    '''
    fold_sizes = []
    for i in range(k):
        fold_sizes.append(int(size/k))
        if size%k != 0 and i < size%k:
            fold_sizes[-1] += 1
    ind_list = [fold_sizes[0] - 1]
    for i in range(1, len(fold_sizes) - 1):
        ind_list.append(ind_list[-1] + fold_sizes[i])
    return fold_sizes, ind_list    