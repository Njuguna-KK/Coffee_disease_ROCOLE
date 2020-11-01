import openpyxl
import collections
import os
from shutil import copyfile
import random

xlsx_path = "./Annotations/RoCoLe-classes.xlsx"
photo_path_prefix = "./Photos/"
binary_path = "./binary/"
multiclass_path = "./multiclass/"

binary_classifications = {
    "healthy": 0,
    "unhealthy": 1
}

multiclass_classifications = {
    "healthy": 0,
    "rust_level_1": 1,
    "rust_level_2": 2,
    "rust_level_3": 3,
    "rust_level_4": 4,
    "red_spider_mite": 5
}

def split_into_train_val_test(dict):
    random.seed(230)

    test = []
    val = []
    train = []

    for category in dict:
        img_names = list(dict[category])
        img_names.sort()
        random.shuffle(img_names)

        test_split = int(0.1 * len(img_names))
        val_split = int(.18 * len(img_names))

        test_img_names = img_names[:test_split]
        val_img_names = img_names[test_split: test_split + val_split]
        train_img_names = img_names[test_split + val_split:]

        test.extend(test_img_names)
        val.extend(val_img_names)
        train.extend(train_img_names)

    return {
        "test": set(test),
        "val": set(val),
        "train": set(train)
    }


def generate_binary_and_multiclass_dict():
    wb_obj = openpyxl.load_workbook(xlsx_path)
    sheet_obj = wb_obj.active

    binary_dict = collections.defaultdict(set)
    multiclass_dict = collections.defaultdict(set)

    num_row = sheet_obj.max_row

    for i in range(2, num_row + 1):
        image_name = sheet_obj.cell(row=i, column=1).value
        binary = sheet_obj.cell(row=i, column=2).value
        multiclass = sheet_obj.cell(row=i, column=3).value

        binary_dict[binary].add(image_name)
        multiclass_dict[multiclass].add(image_name)

    return binary_dict, multiclass_dict


def generate_train_val_test_split(binary_dict, multiclass_dict):
    binary_split = split_into_train_val_test(binary_dict)
    multiclass_split = split_into_train_val_test(multiclass_dict)
    return binary_split, multiclass_split


def get_split(img, split_dict):
    if img in split_dict["test"]:
        return "test"
    if img in split_dict["val"]:
        return "val"
    return "train"


def copy_photo_files_into_directories(classification_dict, new_classification_path, classification_type, split_dict):
    for (category, images) in classification_dict.items():
        for img in images:
            split = get_split(img, split_dict)
            make_dir(os.path.join(new_classification_path, split))
            img_name_prefix = classification_type[category]
            new_img_name = str(img_name_prefix) + "_" + img
            copyfile(os.path.join(photo_path_prefix, img), os.path.join(new_classification_path, split, new_img_name))


def categorize_train_val_test_split(verbose = False):
    (binary_dict, multiclass_dict) = generate_binary_and_multiclass_dict()
    if verbose:
        print("Finished categorizing pictures into their respective classes for binary and multiclass classification")
    binary_split, multiclass_split = generate_train_val_test_split(binary_dict, multiclass_dict)
    if verbose:
        print("Finished splitting dataset")
    copy_photo_files_into_directories(binary_dict, binary_path, binary_classifications, binary_split)
    if verbose:
        print("Finished copying photos into the 'binary' folder")
    copy_photo_files_into_directories(multiclass_dict, multiclass_path, multiclass_classifications, multiclass_split)
    if verbose:
        print("Finished copying photos into the 'multiclass' folder")


def make_dir(path):
    path = os.path.abspath(os.path.join(path))

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            # Raise if directory can't be made, because image cuts won't be saved.
            print('Error creating directory')
            raise e

def main():
    categorize_train_val_test_split(True)

if __name__ == "__main__":
    main()