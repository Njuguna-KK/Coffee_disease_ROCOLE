import os

from PIL import Image
from collections import Counter
from torch.utils.data import Dataset, DataLoader, sampler, BatchSampler
import torchvision.transforms as transforms

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html

class LeafDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.jpg')]

        self.labels = [int(os.path.split(filename)[-1][0]) for filename in self.filenames]
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def get_classes_counts(self):
        return dict(Counter(self.labels))

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = Image.open(self.filenames[idx])  # PIL image
        image = self.transform(image)
        return image, self.labels[idx]

def get_transformer(size):
    # define a training image loader that specifies transforms on images. See documentation for more details.
    train_transformer = transforms.Compose([
        transforms.CenterCrop(size * 2),
        transforms.RandomCrop(int(size * 1.5)),
        transforms.Pad(10),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
        transforms.ToTensor(),  # transform it into a torch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # loader for evaluation, no horizontal flip
    eval_transformer = transforms.Compose([
        transforms.CenterCrop(size * 2),
        transforms.RandomCrop(int(size * 1.5)),
        transforms.Pad(10),
        transforms.Resize(size),  # resize the image to 64x64 (remove if images are already 64x64)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # transform it into a torch tensor
    return train_transformer, eval_transformer

def get_batched_weighted_sampler(params, train_dataset):
    # Calculate class weights (for each of the 6 categories)
    multiclass_dict = train_dataset.get_classes_counts()
    total_count = sum(multiclass_dict.values())
    print("Unique Labels: {} for {} total pictures".format(len(multiclass_dict), total_count))

    class_weights = dict([(k, total_count/v) for (k, v) in multiclass_dict.items()])
    img_weights = [class_weights[label] for label in train_dataset.labels]

    weighted_sampler = sampler.WeightedRandomSampler(img_weights, total_count)
    batch_weighted_sampler = BatchSampler(weighted_sampler, params.batch_size, True)
    return batch_weighted_sampler

def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    img_dimension = params.img_dimension
    dataloaders = {}

    train_transformer, eval_transformer = get_transformer(img_dimension)
    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, split)
            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                train_dataset = LeafDataset(path, train_transformer)
                weighted_sampler = get_batched_weighted_sampler(params, train_dataset)
                dl = DataLoader(train_dataset,
                                        num_workers=params.num_workers, batch_size=params.batch_size, shuffle=True,
                                        pin_memory=params.cuda)
            else:
                dl = DataLoader(LeafDataset(path, eval_transformer), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders
