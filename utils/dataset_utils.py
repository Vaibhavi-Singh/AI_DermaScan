from pathlib import Path
from sklearn.model_selection import train_test_split

def make_train_val_test_split(labeled_dir, out_dir, test_size=0.1, val_size=0.1, random_state=42):
    labeled_dir = Path(labeled_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = [p.name for p in labeled_dir.iterdir() if p.is_dir()]

    for cls in classes:
        src = labeled_dir / cls
        images = list(src.glob("*"))

        train_and_val, test = train_test_split(images, test_size=test_size, random_state=random_state)
        train, val = train_test_split(train_and_val, test_size=val_size/(1-test_size), random_state=random_state)

        for split_name, split_list in [
            ("train", train), ("val", val), ("test", test)
        ]:
            dest_dir = out_dir / split_name / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for p in split_list:
                dest = dest_dir / p.name
                dest.write_bytes(p.read_bytes())  # copy file

    print("Dataset split completed!")
