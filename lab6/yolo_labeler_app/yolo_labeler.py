import math
import tkinter as tk
import json
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog

try:
    from PIL import Image, ImageTk
except ImportError as exc:
    raise SystemExit(
        "Не найден Pillow. Установите его командой: pip install pillow"
    ) from exc


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".webp"}
CONFIG_FILE = Path(__file__).with_name("labeler_config.json")


class YoloLabelerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("YOLO Labeler")
        self.root.geometry("1280x820")
        self.root.minsize(980, 620)

        self.image_dir: Path | None = None
        self.label_dir: Path | None = None
        self.image_paths: list[Path] = []
        self.current_index = -1
        self.class_names: list[str] = []
        self.boxes_by_image: dict[Path, list[dict]] = {}
        self.undo_stack: list[tuple[Path, dict]] = []

        self.original_image: Image.Image | None = None
        self.tk_image = None
        self.display_scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.current_image_size = (1, 1)

        self.drag_start: tuple[float, float] | None = None
        self.preview_rect_id: int | None = None
        self.box_canvas_ids: list[int] = []
        self.selected_box_index: int | None = None
        self.guide_line_x_id: int | None = None
        self.guide_line_y_id: int | None = None

        self._build_ui()
        self._load_config()
        self._bind_keys()
        self.root.after(200, self._startup_open)

    def _build_ui(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        sidebar = tk.Frame(self.root, padx=10, pady=10)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.rowconfigure(8, weight=1)

        tk.Button(
            sidebar, text="Открыть папку", command=self.open_image_folder, width=22
        ).grid(row=0, column=0, sticky="ew", pady=(0, 6))
        tk.Button(
            sidebar, text="Обновить картинки", command=self.refresh_image_list, width=22
        ).grid(row=1, column=0, sticky="ew", pady=(0, 12))

        self.folder_label = tk.Label(
            sidebar, text="Папка не выбрана", justify="left", wraplength=220, anchor="w"
        )
        self.folder_label.grid(row=2, column=0, sticky="ew", pady=(0, 12))

        tk.Label(sidebar, text="Классы").grid(row=3, column=0, sticky="w")
        self.class_listbox = tk.Listbox(sidebar, height=8, exportselection=False)
        self.class_listbox.grid(row=4, column=0, sticky="nsew")
        self.class_listbox.bind("<<ListboxSelect>>", lambda _event: self._refresh_boxes())

        class_actions = tk.Frame(sidebar)
        class_actions.grid(row=5, column=0, sticky="ew", pady=(6, 12))
        class_actions.columnconfigure((0, 1), weight=1)
        tk.Button(class_actions, text="+", command=self.add_class).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        tk.Button(class_actions, text="-", command=self.remove_selected_class).grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )

        tk.Label(sidebar, text="Аннотации").grid(row=6, column=0, sticky="w", pady=(0, 0))
        self.annotation_listbox = tk.Listbox(sidebar, height=12, exportselection=False)
        self.annotation_listbox.grid(row=7, column=0, sticky="nsew")
        self.annotation_listbox.bind(
            "<<ListboxSelect>>", lambda _event: self._select_box_from_list()
        )

        actions = tk.Frame(sidebar)
        actions.grid(row=8, column=0, sticky="ew", pady=(10, 10))
        actions.columnconfigure((0, 1), weight=1)
        tk.Button(actions, text="Удалить", command=self.delete_selected_box).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        tk.Button(actions, text="Сохранить", command=self.save_current_labels).grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )

        info_text = (
            "Управление:\n"
            "ЛКМ: нарисовать рамку\n"
            "Ctrl+Z: отменить последнюю рамку\n"
            "Del: удалить выбранную\n"
            "A / D: предыдущий / следующий класс\n"
            "← / →: предыдущее / следующее изображение\n"
            "+: добавить класс\n"
            "Ctrl+S: сохранить"
        )
        tk.Label(sidebar, text=info_text, justify="left", anchor="w").grid(
            row=9, column=0, sticky="sw"
        )

        viewer = tk.Frame(self.root, padx=10, pady=10)
        viewer.grid(row=0, column=1, sticky="nsew")
        viewer.columnconfigure(0, weight=1)
        viewer.rowconfigure(1, weight=1)

        topbar = tk.Frame(viewer)
        topbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        topbar.columnconfigure(1, weight=1)
        tk.Button(topbar, text="◀", width=4, command=self.prev_image).grid(
            row=0, column=0, padx=(0, 6)
        )
        self.image_status = tk.Label(topbar, text="Изображение не выбрано", anchor="w")
        self.image_status.grid(row=0, column=1, sticky="ew")
        tk.Button(topbar, text="▶", width=4, command=self.next_image).grid(
            row=0, column=2, padx=(6, 0)
        )

        self.canvas = tk.Canvas(viewer, bg="#202124", highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._render_current_image())
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Leave>", lambda _event: self._hide_guides())

    def _bind_keys(self) -> None:
        self.root.bind_all("<Delete>", lambda _event: self.delete_selected_box())
        self.root.bind_all("<Control-s>", lambda _event: self.save_current_labels())
        self.root.bind_all("<Control-S>", lambda _event: self.save_current_labels())
        self.root.bind_all("<KeyPress-plus>", lambda _event: self.add_class())
        self.root.bind_all("<KeyPress-KP_Add>", lambda _event: self.add_class())
        self.root.bind_all("<Left>", lambda _event: self.prev_image())
        self.root.bind_all("<Right>", lambda _event: self.next_image())
        self.root.bind_all("<KeyPress>", self._handle_keypress)

    def _handle_keypress(self, event: tk.Event) -> None:
        char = (event.char or "").lower()
        keysym = (event.keysym or "").lower()
        control_pressed = bool(event.state & 0x4)

        if control_pressed and (
            event.keycode == 90 or char in {"z", "я"} or keysym in {"z", "cyrillic_ya"}
        ):
            self.undo_last_box()
            return

        if char in {"a", "ф"}:
            self.select_prev_class()
            return

        if char in {"d", "в"}:
            self.select_next_class()
            return

        if char.isdigit():
            self.select_class_by_index(int(char))
            return

        if keysym == "plus":
            self.add_class()

    def _startup_open(self) -> None:
        if self.image_dir and self.image_dir.exists():
            self.refresh_image_list()
            if self.image_paths:
                self.load_image(0)
                return
        self.open_image_folder()

    def _load_config(self) -> None:
        if CONFIG_FILE.exists():
            try:
                config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                config = {}
        else:
            config = {}

        saved_dir = config.get("last_image_dir")
        if saved_dir:
            saved_path = Path(saved_dir)
            if saved_path.exists():
                self.image_dir = saved_path
                self.label_dir = self._resolve_label_dir(saved_path)

        self.class_names = [item for item in config.get("classes", []) if isinstance(item, str) and item.strip()]
        self._rebuild_class_listbox()

    def _save_config(self) -> None:
        config = {
            "last_image_dir": str(self.image_dir) if self.image_dir else "",
            "classes": self.class_names,
        }
        CONFIG_FILE.write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def open_image_folder(self) -> None:
        chosen = filedialog.askdirectory(
            title="Выберите папку с изображениями",
            initialdir=str(self.image_dir or Path.cwd()),
        )
        if not chosen:
            return

        self.image_dir = Path(chosen)
        self.boxes_by_image.clear()
        self.undo_stack.clear()
        self.current_index = 0
        self.label_dir = self._resolve_label_dir(self.image_dir)
        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.folder_label.config(
            text=f"Изображения:\n{self.image_dir}\n\nРазметка:\n{self.label_dir}"
        )

        self._save_config()
        self.refresh_image_list()

    def refresh_image_list(self) -> None:
        if not self.image_dir or not self.image_dir.exists():
            return

        current_path = self._current_image_path()
        image_paths = sorted(
            path for path in self.image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        self.image_paths = image_paths
        self.label_dir = self._resolve_label_dir(self.image_dir)
        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.folder_label.config(
            text=f"Изображения:\n{self.image_dir}\n\nРазметка:\n{self.label_dir}"
        )

        if not image_paths:
            self.current_index = -1
            self.original_image = None
            self.canvas.delete("all")
            self.image_status.config(text="В папке нет изображений")
            self.annotation_listbox.delete(0, tk.END)
            return

        if current_path in image_paths:
            new_index = image_paths.index(current_path)
        else:
            new_index = min(self.current_index if self.current_index >= 0 else 0, len(image_paths) - 1)

        self._save_config()
        self.load_image(new_index)

    def _resolve_label_dir(self, image_dir: Path) -> Path:
        # Keep labels inside the exact source folder selected by the user.
        return image_dir / "labels"

    def _legacy_label_dirs(self, image_dir: Path) -> list[Path]:
        legacy_dirs: list[Path] = []
        if image_dir.parent.name.lower() == "images":
            legacy_dirs.append(image_dir.parent.parent / "labels" / image_dir.name)
        if image_dir.name.lower() == "images":
            legacy_dirs.append(image_dir.parent / "labels")
        return [path for path in legacy_dirs if path != self.label_dir]

    def _rebuild_class_listbox(self) -> None:
        self.class_listbox.delete(0, tk.END)
        for index, class_name in enumerate(self.class_names):
            self.class_listbox.insert(tk.END, f"{index}: {class_name}")
        if self.class_names:
            self.class_listbox.selection_set(0)
            self.class_listbox.activate(0)

    def add_class(self) -> None:
        answer = simpledialog.askstring(
            "Новый класс",
            "Введите название нового класса:",
            parent=self.root,
        )
        if answer is None:
            return

        class_name = answer.strip()
        if not class_name:
            return

        self.class_names.append(class_name)
        self._rebuild_class_listbox()
        self.select_class_by_index(len(self.class_names) - 1)
        self._write_classes_file()
        self._save_config()

    def remove_selected_class(self) -> None:
        selection = self.class_listbox.curselection()
        if not selection:
            return

        class_index = selection[0]
        if any(
            box["class_id"] == class_index
            for boxes in self.boxes_by_image.values()
            for box in boxes
        ):
            messagebox.showerror(
                "Ошибка",
                "Нельзя удалить класс, который уже используется в разметке.",
            )
            return

        self.class_names.pop(class_index)
        for boxes in self.boxes_by_image.values():
            for box in boxes:
                if box["class_id"] > class_index:
                    box["class_id"] -= 1

        self._rebuild_class_listbox()
        if self.class_names:
            self.select_class_by_index(min(class_index, len(self.class_names) - 1))
        self._write_classes_file()
        self._save_config()
        self._refresh_boxes()

    def _write_classes_file(self) -> None:
        if not self.image_dir or self.label_dir is None:
            return
        classes_file = self.label_dir.parent / "classes.txt" if self.label_dir else self.image_dir / "classes.txt"
        if self.class_names:
            classes_file.write_text("\n".join(self.class_names) + "\n", encoding="utf-8")
        elif classes_file.exists():
            classes_file.unlink()

    def load_image(self, index: int) -> None:
        if not self.image_paths:
            return

        self.save_current_labels()
        self.current_index = max(0, min(index, len(self.image_paths) - 1))
        image_path = self.image_paths[self.current_index]
        self.original_image = Image.open(image_path).convert("RGB")
        self.current_image_size = self.original_image.size

        if image_path not in self.boxes_by_image:
            self.boxes_by_image[image_path] = self._load_labels_for_image(image_path)

        self.selected_box_index = None
        self.image_status.config(
            text=f"{self.current_index + 1}/{len(self.image_paths)}  {image_path.name}"
        )
        self._render_current_image()
        self._refresh_annotation_list()

    def _load_labels_for_image(self, image_path: Path) -> list[dict]:
        label_path = self._label_path_for_image(image_path)
        if not label_path.exists():
            label_path = next(
                (
                    legacy_dir / f"{image_path.stem}.txt"
                    for legacy_dir in self._legacy_label_dirs(image_path.parent)
                    if (legacy_dir / f"{image_path.stem}.txt").exists()
                ),
                label_path,
            )

        if not label_path.exists():
            return []

        width, height = Image.open(image_path).size
        boxes = []
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(parts[0])
                x_center, y_center, box_width, box_height = map(float, parts[1:])
            except ValueError:
                continue

            x1 = (x_center - box_width / 2) * width
            y1 = (y_center - box_height / 2) * height
            x2 = (x_center + box_width / 2) * width
            y2 = (y_center + box_height / 2) * height
            boxes.append(
                {
                    "class_id": class_id,
                    "x1": max(0.0, x1),
                    "y1": max(0.0, y1),
                    "x2": min(width, x2),
                    "y2": min(height, y2),
                }
            )
        return boxes

    def _label_path_for_image(self, image_path: Path) -> Path:
        assert self.label_dir is not None
        return self.label_dir / f"{image_path.stem}.txt"

    def _render_current_image(self) -> None:
        if not self.original_image:
            return

        canvas_width = max(self.canvas.winfo_width(), 100)
        canvas_height = max(self.canvas.winfo_height(), 100)
        img_width, img_height = self.original_image.size

        self.display_scale = min(canvas_width / img_width, canvas_height / img_height)
        display_width = max(1, int(img_width * self.display_scale))
        display_height = max(1, int(img_height * self.display_scale))
        self.offset_x = (canvas_width - display_width) // 2
        self.offset_y = (canvas_height - display_height) // 2

        resized = self.original_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        self.guide_line_x_id = None
        self.guide_line_y_id = None
        self.preview_rect_id = None
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.tk_image)
        self._draw_boxes()

    def _draw_boxes(self) -> None:
        self.box_canvas_ids.clear()
        image_path = self._current_image_path()
        if not image_path:
            return

        selected_class = self._selected_class_id()
        boxes = self.boxes_by_image.get(image_path, [])
        for index, box in enumerate(boxes):
            x1, y1 = self._image_to_canvas(box["x1"], box["y1"])
            x2, y2 = self._image_to_canvas(box["x2"], box["y2"])

            color = "#00c853" if box["class_id"] == selected_class else "#d84a4a"
            label_color = "#6fbf8f" if box["class_id"] == selected_class else "#c27b7b"
            width = 2 if index == self.selected_box_index else 1
            rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width)
            label = self.class_names[box["class_id"]] if box["class_id"] < len(self.class_names) else str(box["class_id"])
            self.canvas.create_text(
                x1 + 4,
                max(9, y1 + 9),
                text=label,
                fill=label_color,
                anchor="w",
                font=("Arial", 8),
            )
            self.box_canvas_ids.append(rect_id)

    def _refresh_boxes(self) -> None:
        self._render_current_image()
        self._refresh_annotation_list()

    def _refresh_annotation_list(self) -> None:
        self.annotation_listbox.delete(0, tk.END)
        image_path = self._current_image_path()
        if not image_path:
            return

        for index, box in enumerate(self.boxes_by_image.get(image_path, [])):
            label = self.class_names[box["class_id"]] if box["class_id"] < len(self.class_names) else str(box["class_id"])
            width = int(box["x2"] - box["x1"])
            height = int(box["y2"] - box["y1"])
            self.annotation_listbox.insert(tk.END, f"{index + 1}. {label} [{width}x{height}]")

        if self.selected_box_index is not None and self.selected_box_index < self.annotation_listbox.size():
            self.annotation_listbox.selection_set(self.selected_box_index)

    def _select_box_from_list(self) -> None:
        selection = self.annotation_listbox.curselection()
        self.selected_box_index = selection[0] if selection else None
        self._render_current_image()

    def on_mouse_down(self, event: tk.Event) -> None:
        if not self._current_image_path() or not self.class_names:
            return
        point = self._canvas_to_image(event.x, event.y)
        if point is None:
            return
        self.drag_start = point
        self.selected_box_index = None
        self._refresh_annotation_list()

    def on_mouse_drag(self, event: tk.Event) -> None:
        self._update_guides(event.x, event.y)
        if self.drag_start is None:
            return

        current = self._canvas_to_image(event.x, event.y, clamp=True)
        if current is None:
            return
        x1, y1 = self._image_to_canvas(*self.drag_start)
        x2, y2 = self._image_to_canvas(*current)

        if self.preview_rect_id is not None:
            self.canvas.delete(self.preview_rect_id)
        self.preview_rect_id = self.canvas.create_rectangle(
            x1, y1, x2, y2, outline="#00b0ff", width=1, dash=(5, 4)
        )

    def on_mouse_up(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return

        end_point = self._canvas_to_image(event.x, event.y, clamp=True)
        start_point = self.drag_start
        self.drag_start = None

        if self.preview_rect_id is not None:
            self.canvas.delete(self.preview_rect_id)
            self.preview_rect_id = None

        if end_point is None:
            return

        x1 = min(start_point[0], end_point[0])
        y1 = min(start_point[1], end_point[1])
        x2 = max(start_point[0], end_point[0])
        y2 = max(start_point[1], end_point[1])

        if math.hypot(x2 - x1, y2 - y1) < 8:
            return

        image_path = self._current_image_path()
        if not image_path:
            return

        box = {
            "class_id": self._selected_class_id(),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }
        self.boxes_by_image.setdefault(image_path, []).append(box)
        self.selected_box_index = len(self.boxes_by_image[image_path]) - 1
        self.undo_stack.append((image_path, box))
        self._refresh_boxes()
        self._update_guides(event.x, event.y)

    def on_mouse_move(self, event: tk.Event) -> None:
        self._update_guides(event.x, event.y)

    def delete_selected_box(self) -> None:
        image_path = self._current_image_path()
        if not image_path or self.selected_box_index is None:
            return

        boxes = self.boxes_by_image.get(image_path, [])
        if 0 <= self.selected_box_index < len(boxes):
            boxes.pop(self.selected_box_index)
        self.selected_box_index = None
        self._refresh_boxes()

    def undo_last_box(self) -> None:
        while self.undo_stack:
            image_path, box = self.undo_stack.pop()
            boxes = self.boxes_by_image.get(image_path)
            if boxes is None:
                continue

            box_index = next((index for index, item in enumerate(boxes) if item is box), None)
            if box_index is None:
                continue

            boxes.pop(box_index)
            with Image.open(image_path) as image:
                image_size = image.size
            self._save_labels_for_image(image_path, image_size)
            if image_path == self._current_image_path():
                self.selected_box_index = None
                self._refresh_boxes()
            return

    def save_current_labels(self) -> None:
        image_path = self._current_image_path()
        if not image_path or self.label_dir is None:
            return

        self._save_labels_for_image(image_path, self.current_image_size)

    def _save_labels_for_image(self, image_path: Path, image_size: tuple[int, int]) -> None:
        if self.label_dir is None:
            return

        label_path = self._label_path_for_image(image_path)
        boxes = self.boxes_by_image.get(image_path, [])
        width, height = image_size
        lines = []
        for box in boxes:
            x_center = ((box["x1"] + box["x2"]) / 2) / width
            y_center = ((box["y1"] + box["y2"]) / 2) / height
            box_width = (box["x2"] - box["x1"]) / width
            box_height = (box["y2"] - box["y1"]) / height
            lines.append(
                f'{box["class_id"]} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}'
            )

        if lines:
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif label_path.exists():
            label_path.unlink()

    def next_image(self) -> None:
        if self.current_index < len(self.image_paths) - 1:
            self.load_image(self.current_index + 1)

    def prev_image(self) -> None:
        if self.current_index > 0:
            self.load_image(self.current_index - 1)

    def _current_image_path(self) -> Path | None:
        if 0 <= self.current_index < len(self.image_paths):
            return self.image_paths[self.current_index]
        return None

    def _selected_class_id(self) -> int:
        selection = self.class_listbox.curselection()
        return selection[0] if selection else 0

    def select_class_by_index(self, class_index: int) -> None:
        if not self.class_names or not (0 <= class_index < len(self.class_names)):
            return

        self.class_listbox.selection_clear(0, tk.END)
        self.class_listbox.selection_set(class_index)
        self.class_listbox.activate(class_index)
        self.class_listbox.see(class_index)
        self._refresh_boxes()

    def select_prev_class(self) -> None:
        if not self.class_names:
            return
        current = self._selected_class_id()
        self.select_class_by_index((current - 1) % len(self.class_names))

    def select_next_class(self) -> None:
        if not self.class_names:
            return
        current = self._selected_class_id()
        self.select_class_by_index((current + 1) % len(self.class_names))

    def _canvas_to_image(self, x: float, y: float, clamp: bool = False) -> tuple[float, float] | None:
        img_x = (x - self.offset_x) / self.display_scale
        img_y = (y - self.offset_y) / self.display_scale
        width, height = self.current_image_size

        if clamp:
            img_x = min(max(img_x, 0), width)
            img_y = min(max(img_y, 0), height)
            return img_x, img_y

        if 0 <= img_x <= width and 0 <= img_y <= height:
            return img_x, img_y
        return None

    def _image_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return x * self.display_scale + self.offset_x, y * self.display_scale + self.offset_y

    def _update_guides(self, canvas_x: float, canvas_y: float) -> None:
        if self._canvas_to_image(canvas_x, canvas_y) is None:
            self._hide_guides()
            return

        x1 = self.offset_x
        y1 = self.offset_y
        x2 = self.offset_x + int(self.current_image_size[0] * self.display_scale)
        y2 = self.offset_y + int(self.current_image_size[1] * self.display_scale)

        if self.guide_line_x_id is None:
            self.guide_line_x_id = self.canvas.create_line(
                x1,
                canvas_y,
                x2,
                canvas_y,
                fill="#ffd54f",
                dash=(4, 4),
                width=1,
            )
        else:
            self.canvas.coords(self.guide_line_x_id, x1, canvas_y, x2, canvas_y)

        if self.guide_line_y_id is None:
            self.guide_line_y_id = self.canvas.create_line(
                canvas_x,
                y1,
                canvas_x,
                y2,
                fill="#ffd54f",
                dash=(4, 4),
                width=1,
            )
        else:
            self.canvas.coords(self.guide_line_y_id, canvas_x, y1, canvas_x, y2)

        self.canvas.tag_raise(self.guide_line_x_id)
        self.canvas.tag_raise(self.guide_line_y_id)
        if self.preview_rect_id is not None:
            self.canvas.tag_raise(self.preview_rect_id)

    def _hide_guides(self) -> None:
        if self.guide_line_x_id is not None:
            self.canvas.delete(self.guide_line_x_id)
            self.guide_line_x_id = None
        if self.guide_line_y_id is not None:
            self.canvas.delete(self.guide_line_y_id)
            self.guide_line_y_id = None


def main() -> None:
    root = tk.Tk()
    app = YoloLabelerApp(root)
    root.protocol(
        "WM_DELETE_WINDOW",
        lambda: (app.save_current_labels(), app._save_config(), root.destroy()),
    )
    root.mainloop()


if __name__ == "__main__":
    main()
