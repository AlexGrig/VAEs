import torch
import numpy as np
import sys
from urllib import request
from torch.utils.data import Dataset
#sys.path.append("../semi-supervised")
n_labels = 10 #how many different digits to consider. They are always selected as 0..(n_labels-1)
cuda = torch.cuda.is_available()


def get_mnist(location="./", batch_size=64, last_transform='bernoulli', p_random_shuffle=True, labels_per_class=100):
    """
    Inputs:
    ----------------
        last_transform: string
            last_transform, 'none', 'ceil'
    """
    
    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms
    from utils import onehot

    #import pdb; pdb.set_trace()
    if last_transform == 'bernoulli':
        flatten_trans = lambda x: transforms.ToTensor()(x).view(-1).bernoulli()
    elif last_transform == 'none':
        flatten_trans = lambda x: transforms.ToTensor()(x).view(-1)
    elif last_transform == 'ceil':
        flatten_trans = lambda x: transforms.ToTensor()(x).view(-1).ceil()
    else:
        raise ValueError("Unknow last_transfrom value")
    #t2 = lambda x: transforms.ToTensor()(x).view(-1)

    mnist_train = MNIST(location, train=True, download=True,
                        transform=flatten_trans, target_transform=onehot(n_labels))
    mnist_valid = MNIST(location, train=False, download=True,
                        transform=flatten_trans, target_transform=onehot(n_labels))

    def get_sampler(labels, number_images_take=None, previous_indices=None): # n - 
        """
        Inputs:
        ---------------
            number_images_take: int
                How many of each selected digit images to take. e.g if 10 then 
                    # 10 random images of each will be selected.
            exclude_indices: np.array
                Indices to exclude. To exclude labeled indices from unlabelled.
        """
        # Only choose digits in n_labels
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))
        
        #import pdb; pdb.set_trace()
        # Ensure uniform distribution of labels
        
        if previous_indices is not None: # exclude indices from previous iteration
            indices = np.setdiff1d(indices, previous_indices.numpy(), assume_unique=True)
        
        if p_random_shuffle:
            np.random.shuffle(indices)
            
        indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:number_images_take] for i in range(n_labels)])

        indices = torch.from_numpy(indices)
        sampler = SubsetRandomSampler(indices)
        #sampler = SubsetRandomSampler(indices)
        return sampler, indices


    # Dataloaders for MNIST
    sampler, prev_inds = get_sampler(mnist_train.train_labels.numpy(), number_images_take= labels_per_class)
    #import pdb; pdb.set_trace()
    labelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=2, pin_memory=cuda,
                                           sampler=sampler )
    sampler, _ = get_sampler(mnist_train.train_labels.numpy(), previous_indices= prev_inds)
    unlabelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=2, pin_memory=cuda,
                                             sampler=sampler)
    sampler, _ = get_sampler( mnist_valid.test_labels.numpy() )
    validation = torch.utils.data.DataLoader(mnist_valid, batch_size=batch_size, num_workers=2, pin_memory=cuda,
                                             sampler=sampler)
    return labelled, unlabelled, validation


def test_get_mnist_1():
    """
    Test sum of labeled and unlabeled
    """

    labelled, unlabelled, validation = get_mnist(location="./downloaded_datasets", batch_size=64, labels_per_class=10)

    samples_num_1 = [ ii[0].shape[0] for ii in labelled ]
    samples_num_2 = [ ii[0].shape[0] for ii in unlabelled ]
    samples_num_3 = [ ii[0].shape[0] for ii in validation ]
    
    print( "Labelled samples num:", np.sum(samples_num_1) )
    print( "Unabelled samples num:", np.sum(samples_num_2) )
    print( "Validation samples num:", np.sum(samples_num_3) )
    
    
    assert ((np.sum(samples_num_1) + np.sum(samples_num_2)) == 60000), "Labeled and unlabeled sum is not 60000"


def test_get_mnist_2():
    """
    Test that mnist is completely in unlabeled if labels_per_class=0
    """

    labelled, unlabelled, validation = get_mnist(location="./downloaded_datasets", batch_size=64, labels_per_class=0)

    samples_num_1 = [ ii[0].shape[0] for ii in labelled ]
    samples_num_2 = [ ii[0].shape[0] for ii in unlabelled ]
    samples_num_3 = [ ii[0].shape[0] for ii in validation ]
    
    print( "Labelled samples num:", np.sum(samples_num_1) )
    print( "Unabelled samples num:", np.sum(samples_num_2) )
    print( "Validation samples num:", np.sum(samples_num_3) )
    
    #assert ((samples_num_1 + samples_num_2) == 60000), "Labeled and unlabeled sum is not 60000"
    
def test_get_mnist_3():
    """
    Not working!
    Test the last transformation influence
    """

    labelled, unlabelled, validation = get_mnist(location="./downloaded_datasets", batch_size=64, last_transform='bernoulli', p_random_shuffle=False, labels_per_class=100)
    for ii in unlabelled: u0_b = ii[0][0,:].numpy(); break # take the first element
    
    labelled, unlabelled, validation = get_mnist(location="./downloaded_datasets", batch_size=64, last_transform='none', p_random_shuffle=False, labels_per_class=100)
    for ii in unlabelled: u0_n = ii[0][0,:].numpy(); break # take the first element
    
    labelled, unlabelled, validation = get_mnist(location="./downloaded_datasets", batch_size=64, last_transform='ceil', p_random_shuffle=False, labels_per_class=100)
    for ii in unlabelled: u0_c = ii[0][0,:].numpy(); break # take the first element
    
    import pdb; pdb.set_trace()
    
    print( np.all((u0_n > 0) == (u0_c >0)) )
    
if __name__ == '__main__':
    test_get_mnist_1()
    test_get_mnist_2()
    #test_get_mnist_3()