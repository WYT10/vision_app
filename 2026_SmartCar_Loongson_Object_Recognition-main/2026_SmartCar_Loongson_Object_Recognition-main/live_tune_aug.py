import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
import random

class AugmentationTuner:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Augmentation Tuner")
        
        self.sequence = [] # List of dicts representing operations
        self.current_step = 0
        
        self.original_image_bgr = None
        self.rng = random.Random(42)

        self._build_ui()
        self._load_default_image()

    def _build_ui(self):
        # Top toolbar
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Button(toolbar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        
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
        ttk.Button(btn_frame, text="Add Salt-Pepper (a)", command=lambda: self.add_op("sp")).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Add Median Blur (b)", command=lambda: self.add_op("mb")).pack(side=tk.LEFT, padx=2)
        
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
        ds_path = Path(__file__).resolve().parent / "img_dataset"
        if ds_path.exists():
            images = list(ds_path.rglob("*.jpg"))
            if images:
                self.original_image_bgr = self._imread_unicode(images[0])
                if self.original_image_bgr is not None:
                    self.update_pipeline()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.original_image_bgr = self._imread_unicode(path)
            if self.original_image_bgr is not None:
                self.update_pipeline()

    def add_op(self, op_type):
        if op_type == "sp":
            op = {"type": "sp", "amount": 0.01, "ratio": 0.5, "seed": 42}
        else:
            op = {"type": "mb", "k": 3}
            
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

    def rebuild_op_ui(self):
        for widget in self.ops_frame.winfo_children():
            widget.destroy()
            
        self.op_widgets = []
        
        for i, op in enumerate(self.sequence):
            frame = ttk.LabelFrame(self.ops_frame, text=f"Step {i+1}: {'Salt-Pepper' if op['type']=='sp' else 'Median Blur'}")
            frame.pack(fill=tk.X, padx=5, pady=5)
            
            ctrl_frame = ttk.Frame(frame)
            ctrl_frame.pack(fill=tk.X, padx=2, pady=2)
            
            if op['type'] == 'sp':
                ttk.Label(ctrl_frame, text="Amount (0.0..0.20):").pack(side=tk.TOP, anchor=tk.W)
                amount_var = tk.DoubleVar(value=op['amount'])
                s1 = ttk.Scale(ctrl_frame, from_=0.0, to=0.20, variable=amount_var, command=lambda v, idx=i: self.on_param_change(idx, 'amount', v))
                s1.pack(fill=tk.X)
                
                ttk.Label(ctrl_frame, text="Ratio S/P (0.0..1.0):").pack(side=tk.TOP, anchor=tk.W)
                ratio_var = tk.DoubleVar(value=op['ratio'])
                s2 = ttk.Scale(ctrl_frame, from_=0.0, to=1.0, variable=ratio_var, command=lambda v, idx=i: self.on_param_change(idx, 'ratio', v))
                s2.pack(fill=tk.X)
                
                ttk.Label(ctrl_frame, text="Seed (0..100):").pack(side=tk.TOP, anchor=tk.W)
                seed_var = tk.IntVar(value=op.get('seed', 42))
                s3 = ttk.Scale(ctrl_frame, from_=0, to=100, variable=seed_var, command=lambda v, idx=i: self.on_param_change(idx, 'seed', v))
                s3.pack(fill=tk.X)
                
            elif op['type'] == 'mb':
                ttk.Label(ctrl_frame, text="Kernel Size (1..31):").pack(side=tk.TOP, anchor=tk.W)
                k_var = tk.IntVar(value=op['k'])
                s1 = ttk.Scale(ctrl_frame, from_=1, to=31, variable=k_var, command=lambda v, idx=i: self.on_param_change(idx, 'k', v))
                s1.pack(fill=tk.X)

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

    def on_param_change(self, idx, param, val):
        val = float(val)
        if param == 'k':
            # ensure odd integer
            val = int(val)
            if val % 2 == 0:
                val += 1
            if val < 1: val = 1
        if param == 'seed':
            val = int(val)
            
        self.sequence[idx][param] = val
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
            op_name = 'Salt-Pepper' if op['type']=='sp' else 'Median Blur'
            self.step_label.config(text=f"Current Step: {step} ({op_name})")

    def _apply_sp(self, img_pil, amount, ratio, seed):
        rng = random.Random(seed)
        img = np.array(img_pil).copy()
        h, w, _ = img.shape
        total_pixels = h * w
        noisy_pixels = max(1, int(total_pixels * amount))

        num_salt = int(noisy_pixels * ratio)
        num_pepper = max(0, noisy_pixels - num_salt)

        if num_salt > 0:
            ys = np.array([rng.randrange(h) for _ in range(num_salt)], dtype=np.int32)
            xs = np.array([rng.randrange(w) for _ in range(num_salt)], dtype=np.int32)
            img[ys, xs] = 255

        if num_pepper > 0:
            ys = np.array([rng.randrange(h) for _ in range(num_pepper)], dtype=np.int32)
            xs = np.array([rng.randrange(w) for _ in range(num_pepper)], dtype=np.int32)
            img[ys, xs] = 0

        return Image.fromarray(img)

    def _apply_mb(self, img_pil, k):
        img = np.array(img_pil)
        img = cv2.medianBlur(img, k)
        return Image.fromarray(img)

    def update_pipeline(self):
        if self.original_image_bgr is None: return
        
        # Reset RNG seed so noise is deterministic per tune
        self.rng.seed(42)
        
        self.pipeline_images = []
        
        img_rgb = cv2.cvtColor(self.original_image_bgr, cv2.COLOR_BGR2RGB)
        current_pil = Image.fromarray(img_rgb)
        
        self.pipeline_images.append(current_pil)
        
        for op in self.sequence:
            if op['type'] == 'sp':
                current_pil = self._apply_sp(current_pil, op['amount'], op['ratio'], op.get('seed', 42))
            elif op['type'] == 'mb':
                current_pil = self._apply_mb(current_pil, op['k'])
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
