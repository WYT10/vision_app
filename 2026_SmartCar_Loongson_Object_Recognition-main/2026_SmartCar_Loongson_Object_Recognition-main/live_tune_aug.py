import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
import random
import json
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

class AugmentationTuner:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Augmentation Tuner")
        
        self.sequence = [] # List of dicts representing operations
        self.current_step = 0
        self.refresh_counter = 0
        
        self.original_image_bgr = None
        self.rng = random.Random(42)

        self._build_ui()
        self._load_default_image()

    def _build_ui(self):
        # Top toolbar
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Button(toolbar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Refresh Random", command=self.refresh_randomized).pack(side=tk.RIGHT, padx=5)
        ttk.Button(toolbar, text="Export to Dataset Config", command=self.export_config).pack(side=tk.RIGHT, padx=5)
        
        # Layout: Left for image, Right for sequence
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)
        
        # Left side: Image and step slider
        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=1)
        
        # Canvas for image
        self.canvas = tk.Canvas(left_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas.bind("<Configure>", lambda e: self.update_image_view())
        
        # Step Slider
        self.step_var = tk.IntVar(value=0)
        self.step_slider = ttk.Scale(left_frame, from_=0, to=0, variable=self.step_var, command=self.on_step_change, orient=tk.HORIZONTAL)
        self.step_slider.pack(fill=tk.X, padx=5, pady=5)
        
        self.step_label = ttk.Label(left_frame, text="Current Step: 0 (Original)")
        self.step_label.pack(side=tk.BOTTOM, pady=2)
        
        # Right side: Sequence controls
        right_frame = ttk.Frame(main_pane, width=300)
        main_pane.add(right_frame, weight=0)
        
        # Buttons to add operations
        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Add Gaussian Noise", command=lambda: self.add_op("gn")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Add Median Blur", command=lambda: self.add_op("mb")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Add Resize", command=lambda: self.add_op("rs")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Add Affine", command=lambda: self.add_op("af")).pack(side=tk.LEFT, padx=2)
        
        # Scrollable area for operations
        self.ops_canvas = tk.Canvas(right_frame)
        self.scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=self.ops_canvas.yview)
        self.ops_frame = ttk.Frame(self.ops_canvas)
        
        self.ops_frame.bind(
            "<Configure>",
            lambda e: self.ops_canvas.configure(
                scrollregion=self.ops_canvas.bbox("all")
            )
        )
        
        self.ops_canvas.create_window((0, 0), window=self.ops_frame, anchor="nw")
        self.ops_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.ops_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        self.op_widgets = [] # Keep track of UI frames

    def _imread_unicode(self, path):
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img

    def _load_default_image(self):
        # Try to find a random image from the img_dataset
        images = self._collect_dataset_images()
        if images:
            selected = random.choice(images)
            self.original_image_bgr = self._imread_unicode(selected)
            if self.original_image_bgr is not None:
                self.update_pipeline()

    def _collect_dataset_images(self):
        ds_path = Path(__file__).resolve().parent / "img_dataset"
        if not ds_path.exists():
            return []
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        images = []
        for ext in exts:
            images.extend(ds_path.rglob(ext))
        return images

    def load_image(self):
        images = self._collect_dataset_images()
        if images:
            selected = random.choice(images)
            self.original_image_bgr = self._imread_unicode(selected)
            if self.original_image_bgr is not None:
                self.update_pipeline()
                return

        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if path:
            self.original_image_bgr = self._imread_unicode(path)
            if self.original_image_bgr is not None:
                self.update_pipeline()

    def export_config(self):
        default_path = Path(__file__).resolve().parent / "aug_config.json"
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile="aug_config.json",
            initialdir=str(default_path.parent),
            title="Save Augmentation Config"
        )
        if not path: return
        
        # Save sequence out 
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "note": "Auto-exported from Live Tuner",
                "custom_sequence": True,
                "sequence": self.sequence
            }, f, indent=4)
        messagebox.showinfo("Saved", f"Configuration exported to:\n{path}\n\nThe next run of prepare_cls_dataset.py will automatically pick this up!")

    def refresh_randomized(self):
        self.refresh_counter += 1
        self.update_pipeline()

    def add_op(self, op_type):
        if op_type == "gn":
            op = {
                "type": "gn",
                "mean_limit": 0.05,
                "var_min": 0.0,
                "var_max": 0.02,
            }
        elif op_type == "mb":
            op = {"type": "mb", "k": 3}
        elif op_type == "rs":
            op = {"type": "rs", "size": 40}
        elif op_type == "af":
            op = {"type": "af", "angle_limit": 20.0}
            
        self.sequence.append(op)
        self.rebuild_op_ui()
        self.update_pipeline()

    def remove_op(self, idx):
        self.sequence.pop(idx)
        self.rebuild_op_ui()
        self.update_pipeline()

    def move_op(self, idx, direction):
        if 0 <= idx + direction < len(self.sequence):
            # Swap
            self.sequence[idx], self.sequence[idx+direction] = self.sequence[idx+direction], self.sequence[idx]
            self.rebuild_op_ui()
            self.update_pipeline()

    def _op_label(self, op_type):
        mapping = {
            "gn": "Gaussian Noise",
            "mb": "Median Blur",
            "rs": "Resize",
            "af": "Affine Transform",
        }
        return mapping.get(op_type, op_type)

    def _add_slider_with_entry(self, parent, idx, param, label_text, from_, to_, value, is_int=False):
        ttk.Label(parent, text=label_text).pack(side=tk.TOP, anchor=tk.W)
        row = ttk.Frame(parent)
        row.pack(fill=tk.X)

        var = tk.IntVar(value=int(value)) if is_int else tk.DoubleVar(value=float(value))
        scale = ttk.Scale(
            row,
            from_=from_,
            to=to_,
            variable=var,
            command=lambda v, i=idx, p=param, vref=var: self.on_param_change(i, p, v, vref),
        )
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        entry = ttk.Entry(row, width=10, textvariable=var)
        entry.pack(side=tk.LEFT, padx=(6, 0))
        entry.bind("<Return>", lambda e, i=idx, p=param, vref=var: self.on_param_change(i, p, vref.get(), vref))
        entry.bind("<FocusOut>", lambda e, i=idx, p=param, vref=var: self.on_param_change(i, p, vref.get(), vref))

    def rebuild_op_ui(self):
        for widget in self.ops_frame.winfo_children():
            widget.destroy()
            
        self.op_widgets = []
        
        for i, op in enumerate(self.sequence):
            frame = ttk.LabelFrame(self.ops_frame, text=f"Step {i+1}: {self._op_label(op['type'])}")
            frame.pack(fill=tk.X, padx=5, pady=5)
            
            ctrl_frame = ttk.Frame(frame)
            ctrl_frame.pack(fill=tk.X, padx=2, pady=2)
            
            if op['type'] == 'gn':
                self._add_slider_with_entry(
                    ctrl_frame,
                    i,
                    'mean_limit',
                    "Mean Limit (0.0..0.5), sampled in [-limit,+limit]:",
                    0.0,
                    0.5,
                    op['mean_limit'],
                    is_int=False,
                )
                self._add_slider_with_entry(
                    ctrl_frame,
                    i,
                    'var_min',
                    "Variance Min (0.0..0.5):",
                    0.0,
                    0.5,
                    op['var_min'],
                    is_int=False,
                )
                self._add_slider_with_entry(
                    ctrl_frame,
                    i,
                    'var_max',
                    "Variance Max (0.0..0.05), sampled in [0,var_max]:",
                    0.0,
                    0.05,
                    op['var_max'],
                    is_int=False,
                )
                
            elif op['type'] == 'mb':
                self._add_slider_with_entry(
                    ctrl_frame,
                    i,
                    'k',
                    "Kernel Size (1..41):",
                    1,
                    41,
                    op['k'],
                    is_int=True,
                )
                
            elif op['type'] == 'rs':
                self._add_slider_with_entry(
                    ctrl_frame,
                    i,
                    'size',
                    "Size (10..200):",
                    10,
                    200,
                    op['size'],
                    is_int=True,
                )

            elif op['type'] == 'af':
                self._add_slider_with_entry(
                    ctrl_frame,
                    i,
                    'angle_limit',
                    "Angle Limit (0..180), sampled in [-limit,+limit]:",
                    0.0,
                    180.0,
                    op['angle_limit'],
                    is_int=False,
                )

            # Move/Delete buttons
            action_frame = ttk.Frame(frame)
            action_frame.pack(fill=tk.X)
            ttk.Button(action_frame, text="^", width=2, command=lambda idx=i: self.move_op(idx, -1)).pack(side=tk.LEFT)
            ttk.Button(action_frame, text="v", width=2, command=lambda idx=i: self.move_op(idx, 1)).pack(side=tk.LEFT)
            ttk.Button(action_frame, text="X", width=2, command=lambda idx=i: self.remove_op(idx)).pack(side=tk.RIGHT)

        # Update step slider
        max_step = len(self.sequence)
        self.step_slider.configure(to=max_step)
        if self.step_var.get() > max_step:
            self.step_var.set(max_step)
            
        # Ensure we always view max step when modifying sequence length to avoid confusion
        self.step_var.set(max_step)
        self.update_step_label()

    def on_param_change(self, idx, param, val, tk_var=None):
        val = float(val)
        if param == 'k':
            # ensure odd integer
            val = int(val)
            if val % 2 == 0:
                val += 1
            if val < 1: val = 1
            if val > 41: val = 41
        if param in ['size']:
            val = int(val)
        if param == 'size':
            if val < 10: val = 10
            if val > 200: val = 200
        if param == 'mean_limit':
            if val < 0.0: val = 0.0
            if val > 0.5: val = 0.5
        if param == 'var_min':
            if val < 0.0: val = 0.0
            if val > 0.5: val = 0.5
        if param == 'var_max':
            if val < 0.0: val = 0.0
            if val > 0.05: val = 0.05
        if param == 'angle_limit':
            if val < 0.0: val = 0.0
            if val > 180.0: val = 180.0
            
        self.sequence[idx][param] = val

        if param in ['var_min', 'var_max']:
            # Keep interval valid while honoring var_max upper bound.
            self.sequence[idx]['var_max'] = min(self.sequence[idx]['var_max'], 0.05)
            self.sequence[idx]['var_min'] = min(self.sequence[idx]['var_min'], self.sequence[idx]['var_max'])

        if tk_var is not None:
            tk_var.set(int(val) if isinstance(tk_var, tk.IntVar) else float(val))
        self.update_pipeline()

    def on_step_change(self, *args):
        self.update_step_label()
        self.update_image_view()

    def update_step_label(self):
        step = self.step_var.get()
        if step == 0:
            self.step_label.config(text="Current Step: 0 (Original)")
        else:
            op = self.sequence[step-1]
            op_name = self._op_label(op['type'])
            self.step_label.config(text=f"Current Step: {step} ({op_name})")

    def _apply_gn(self, img_pil, mean, variance, seed):
        img = np.array(img_pil).astype(np.float32) / 255.0
        std = np.sqrt(max(variance, 0.0))
        noise = np.random.default_rng(seed).normal(loc=mean, scale=std, size=img.shape)
        noisy = np.clip(img + noise, 0.0, 1.0)
        return Image.fromarray((noisy * 255.0).astype(np.uint8))

    def _apply_mb(self, img_pil, k):
        img = np.array(img_pil)
        img = cv2.medianBlur(img, k)
        return Image.fromarray(img)

    def _apply_rs(self, img_pil, size):
        return TF.resize(
            img_pil,
            [size, size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

    def _apply_af(self, img_pil, angle):
        return TF.affine(
            img_pil,
            angle=angle,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )

    def update_pipeline(self):
        if self.original_image_bgr is None: return
        
        # Keep deterministic randomness for the same refresh cycle.
        self.rng.seed(42 + self.refresh_counter)
        
        self.pipeline_images = []
        
        img_rgb = cv2.cvtColor(self.original_image_bgr, cv2.COLOR_BGR2RGB)
        current_pil = Image.fromarray(img_rgb)
        
        self.pipeline_images.append(current_pil)
        
        for op in self.sequence:
            if op['type'] == 'gn':
                mean = self.rng.uniform(-op['mean_limit'], op['mean_limit'])
                variance = self.rng.uniform(op.get('var_min', 0.0), op['var_max'])
                seed = self.rng.randint(0, 10**9)
                current_pil = self._apply_gn(current_pil, mean, variance, seed)
            elif op['type'] == 'mb':
                current_pil = self._apply_mb(current_pil, op['k'])
            elif op['type'] == 'rs':
                current_pil = self._apply_rs(current_pil, op['size'])
            elif op['type'] == 'af':
                angle = self.rng.uniform(-op['angle_limit'], op['angle_limit'])
                current_pil = self._apply_af(current_pil, angle)
            self.pipeline_images.append(current_pil)
            
        self.update_image_view()

    def update_image_view(self):
        if not hasattr(self, 'pipeline_images') or not self.pipeline_images: return
        
        step = self.step_var.get()
        if step >= len(self.pipeline_images):
            step = len(self.pipeline_images) - 1
            
        pil_img = self.pipeline_images[step]
        
        # Resize to fit canvas
        self.root.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        
        if cw > 10 and ch > 10:
            img_ratio = pil_img.width / pil_img.height
            c_ratio = cw / ch
            if img_ratio > c_ratio:
                new_w = cw
                new_h = int(cw / img_ratio)
            else:
                new_h = ch
                new_w = int(ch * img_ratio)
                
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.NEAREST)
            
        self.tk_image = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, anchor=tk.CENTER, image=self.tk_image)

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    app = AugmentationTuner(root)
    root.mainloop()
