import os
from pathlib import Path
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from tqdm import tqdm

IMAGE_SIZE = (224, 224)

def resize_and_save(src_path, dest_path):
    img = Image.open(src_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest_path)

def build_augmented_dataset(input_dir, output_dir, augment_count_per_image=5):
    datagen = ImageDataGenerator(
        # rotation_range=20,
        # width_shift_range=0.08,
        # height_shift_range=0.08,
        # zoom_range=0.12,
        # horizontal_flip=True,
        # fill_mode='nearest'
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.20,
        brightness_range=[0.7, 1.3],
        horizontal_flip=True,
        shear_range=0.2,
        fill_mode='nearest'

    )

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for split in ["train", "val", "test"]:
        split_path = input_dir / split
        for cls_dir in split_path.iterdir():
            if not cls_dir.is_dir():
                continue

            out_cls_dir = output_dir / split / cls_dir.name
            out_cls_dir.mkdir(parents=True, exist_ok=True)

            for img_path in tqdm(list(cls_dir.glob("*")), desc=f"{split}/{cls_dir.name}"):
                img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
                img_arr = img_to_array(img)
                fname = img_path.stem

                # save original resized image
                img.save(out_cls_dir / f"{fname}.jpg")

                # augment
                x = np.expand_dims(img_arr, axis=0)
                for i, batch in zip(range(augment_count_per_image), datagen.flow(x, batch_size=1)):
                    aug_img = array_to_img(batch[0])
                    aug_img.save(out_cls_dir / f"{fname}_aug{i}.jpg")

if __name__ == "__main__":
    # build_augmented_dataset("../data/processed", "../data/processed_aug", augment_count_per_image=2)
    build_augmented_dataset("data/processed", "data/processed_aug", augment_count_per_image=5)

    print("Preprocessing + augmentation completed successfully!")
