import json
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split

def split_dataset_by_category(file_path, output_dir, test_size=0.0001*2, random_state=42):
    # Load the dataset
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Group questions by category
    # category_dict = defaultdict(list)
    # category_set = set()
    # for item in dataset:
    #     # Extract the category from the "image" key
    #     category = item['image'].split('/')[0]
    #     category_set.add(category)
    #     category_dict[category].append(item)
    #
    # Create train and test splits
    # print("All Categories:", category_set)
    train_set = []
    test_set = []

    # for category, items in category_dict.items():
    #     # Split the items in the current category
    train_items, test_items = train_test_split(
        dataset, test_size=test_size, random_state=random_state
    )
    train_set.extend(train_items)
    test_set.extend(test_items)


    # Save the splits to output directory
    print("Size of test:", len(test_set))
    os.makedirs(output_dir, exist_ok=True)
    # train_path = os.path.join(output_dir, 'tr-llava-train.json')
    test_path = os.path.join(output_dir, 'pretrain-tr-llava-train-overfit.json')

    # with open(train_path, 'w', encoding='utf-8') as f:
    #     json.dump(train_set, f, ensure_ascii=False, indent=2)

    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)

    print(f"Train and test sets saved to {output_dir}")

# Example usage
split_dataset_by_category('pretrain-tr-llava-train.json', './')


