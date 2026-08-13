import os
import shutil

def organize_splits(split_file, dataset_dir, output_dir, split_name):
    dir = os.path.join(output_dir, split_name)
    os.makedirs(dir, exist_ok=True)

    matched_count = 0
    missing_count = 0

    with open(split_file, 'r') as f:
        for line in f:
            vid_part = line.strip().split()

            path = vid_part[0]

            category, filename = os.path.split(path)
            vid_name = os.path.splitext(filename)[0]

            src_file = os.path.join(dataset_dir, category, f"{vid_name}.npy")

            dest_dir = os.path.join(dir, category)
            os.makedirs(dest_dir, exist_ok=True)
            dest_file = os.path.join(dest_dir, f"{vid_name}.npy")
            print(src_file)

            if os.path.exists(src_file):
                shutil.copy(src_file, dest_file)
                matched_count += 1
            else:
                missing_count += 1

            print(f"[{split_name.upper()}] Successfully copied: {matched_count} files. Missing/Unprocessed: {missing_count} files.")





if __name__ == "__main__":
    testlist = "testlist01.txt"
    trainlist = "trainlist01.txt"
   
    dataset_dir = "completed_flow_poses"

    output_dir = "sorted_flow_poses"

    testList = f"ucfTrainTestlist/{testlist}"
    trainlist = f"ucfTrainTestlist/{trainlist}"

    organize_splits(testList, dataset_dir, output_dir, "test")
    organize_splits(trainlist, dataset_dir, output_dir, "train")
