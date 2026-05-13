import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import random
import shutil
from pathlib import Path

import ultralytics
from ultralytics import YOLO

lab_dir = Path(r"C:\Users\slywater\Projects\neuro\lab6")

ultralytics.checks()

#%%
# Пути к исходному датасету, который размечался через yolo_labeler_app.
source_images_dir = Path(r"C:\Users\slywater\Desktop\images\train")
source_labels_dir = source_images_dir / "labels"
test_images_dir = Path(r"C:\Users\slywater\Projects\neuro\lab6\data_helmet_full\images\test")

# Папка, куда соберем датасет в стандартной структуре YOLO.
dataset_dir = lab_dir / "data_helmet_full"
data_yaml = lab_dir / "helmet_full.yaml"
masks_dir = lab_dir / "masks"

image_extensions = [".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".webp"]

#%%
labels = sorted(
    source_labels_dir.glob("*.txt"),
    key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
)

image_label_pairs = []
for label_path in labels:
    image_path = None
    for extension in image_extensions:
        candidate = source_images_dir / f"{label_path.stem}{extension}"
        if candidate.exists():
            image_path = candidate
            break

    if image_path is None:
        raise FileNotFoundError(f"Не найдена картинка для файла разметки: {label_path.name}")

    image_label_pairs.append((image_path, label_path))

print("Всего размеченных картинок:", len(image_label_pairs))

#%%
# Делим датасет на train и val.
# 80% пойдет на обучение, 20% на проверку во время обучения.
random.seed(42)
random.shuffle(image_label_pairs)

train_count = int(len(image_label_pairs) * 0.8)
train_pairs = image_label_pairs[:train_count]
val_pairs = image_label_pairs[train_count:]

print("Train:", len(train_pairs))
print("Val:", len(val_pairs))

#%%
# Очищаем старую собранную копию и создаем папки для YOLO.
for split in ["train", "val", "test"]:
    for kind in ["images", "labels"]:
        target_dir = dataset_dir / kind / split
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

#%%
# Копируем картинки и txt-файлы в стандартную YOLO-структуру.
for split, pairs in [("train", train_pairs), ("val", val_pairs)]:
    image_dir = dataset_dir / "images" / split
    label_dir = dataset_dir / "labels" / split

    for image_path, label_path in pairs:
        shutil.copy2(image_path, image_dir / image_path.name)
        shutil.copy2(label_path, label_dir / label_path.name)

print("Датасет подготовлен:", dataset_dir)

#%%
# Копируем отдельную папку test внутрь датасета, чтобы потом проверить модель.
source_test_dir = Path(r"C:\Users\slywater\Desktop\images\test")
test_image_dir = dataset_dir / "images" / "test"

for image_path in source_test_dir.iterdir():
    if image_path.suffix.lower() in image_extensions:
        shutil.copy2(image_path, test_image_dir / image_path.name)

print("Test-картинки скопированы:", len(list(test_image_dir.iterdir())))

#%%
model = YOLO(str(lab_dir / "yolov8s.pt"))
results = model.train(data=str(data_yaml), model="yolov8s.pt", epochs=10, batch=8,
                      project=str(masks_dir), name='helmet_full', exist_ok=True,
                      val=True, verbose=True)

#%%
# Проверим обученную модель на папке test.
best_weights = masks_dir / "helmet_full" / "weights" / "best.pt"
if not best_weights.exists():
    best_weights = Path(r"C:\Users\slywater\Projects\neuro\masks\helmet_full\weights\best.pt")
model = YOLO(str(best_weights))

results = model.predict(source=str(test_images_dir), save=True, conf=0.25,
                        project=str(masks_dir), name='helmet_test_predictions',
                        exist_ok=True)

print("Проверка test сохранена в папку masks/helmet_test_predictions")
