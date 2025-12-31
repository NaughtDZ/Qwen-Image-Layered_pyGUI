
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import torch
from diffusers import QwenImageLayeredPipeline

class ImageLayeredGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Qwen Image Layered GUI")

        # Variables
        self.image_path = tk.StringVar()
        self.scale_factor = tk.DoubleVar(value=1.0)
        self.layers = tk.IntVar(value=4)
        self.resolution = tk.IntVar(value=640)
        self.num_inference_steps = tk.IntVar(value=50)
        self.true_cfg_scale = tk.DoubleVar(value=4.0)
        self.negative_prompt = tk.StringVar(value=" ")
        self.num_images_per_prompt = tk.IntVar(value=1)
        self.cfg_normalize = tk.BooleanVar(value=True)
        self.use_en_prompt = tk.BooleanVar(value=True)
        self.seed = tk.IntVar(value=777)

        # GUI Elements
        tk.Label(root, text="Image Path:").grid(row=0, column=0, padx=10, pady=5)
        self.path_entry = tk.Entry(root, textvariable=self.image_path, width=50)
        self.path_entry.grid(row=0, column=1, padx=10, pady=5)
        tk.Button(root, text="Browse", command=self.browse_image).grid(row=0, column=2, padx=10, pady=5)

        tk.Label(root, text="Scale Factor (e.g., 0.5 for half size):").grid(row=1, column=0, padx=10, pady=5)
        tk.Entry(root, textvariable=self.scale_factor, width=10).grid(row=1, column=1, padx=10, pady=5, sticky="w")

        tk.Label(root, text="Number of Layers:").grid(row=2, column=0, padx=10, pady=5)
        tk.Spinbox(root, from_=1, to=10, textvariable=self.layers, width=5).grid(row=2, column=1, padx=10, pady=5, sticky="w")

        tk.Label(root, text="Resolution:").grid(row=3, column=0, padx=10, pady=5)
        res_options = [640, 1024]
        tk.OptionMenu(root, self.resolution, *res_options).grid(row=3, column=1, padx=10, pady=5, sticky="w")

        tk.Label(root, text="Num Inference Steps:").grid(row=4, column=0, padx=10, pady=5)
        tk.Spinbox(root, from_=1, to=100, textvariable=self.num_inference_steps, width=5).grid(row=4, column=1, padx=10, pady=5, sticky="w")

        tk.Label(root, text="True CFG Scale:").grid(row=5, column=0, padx=10, pady=5)
        tk.Entry(root, textvariable=self.true_cfg_scale, width=10).grid(row=5, column=1, padx=10, pady=5, sticky="w")

        tk.Label(root, text="Negative Prompt:").grid(row=6, column=0, padx=10, pady=5)
        tk.Entry(root, textvariable=self.negative_prompt, width=50).grid(row=6, column=1, padx=10, pady=5)

        tk.Label(root, text="Num Images per Prompt:").grid(row=7, column=0, padx=10, pady=5)
        tk.Spinbox(root, from_=1, to=5, textvariable=self.num_images_per_prompt, width=5).grid(row=7, column=1, padx=10, pady=5, sticky="w")

        tk.Checkbutton(root, text="CFG Normalize", variable=self.cfg_normalize).grid(row=8, column=0, padx=10, pady=5, sticky="w")

        tk.Checkbutton(root, text="Use EN Prompt", variable=self.use_en_prompt).grid(row=8, column=1, padx=10, pady=5, sticky="w")

        tk.Label(root, text="Seed:").grid(row=9, column=0, padx=10, pady=5)
        tk.Entry(root, textvariable=self.seed, width=10).grid(row=9, column=1, padx=10, pady=5, sticky="w")

        tk.Button(root, text="Process Image", command=self.process_image).grid(row=10, column=0, columnspan=3, pady=20)

        # Load pipeline
        self.pipeline = None
        self.load_pipeline()

    def load_pipeline(self):
        try:
            self.pipeline = QwenImageLayeredPipeline.from_pretrained("Qwen/Qwen-Image-Layered")
            self.pipeline = self.pipeline.to("cuda", torch.bfloat16)
            self.pipeline.set_progress_bar_config(disable=None)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load pipeline: {str(e)}")
            self.root.quit()

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            self.image_path.set(file_path)

    def process_image(self):
        path = self.image_path.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Invalid image path.")
            return

        try:
            # Load and scale image
            image = Image.open(path).convert("RGBA")
            if self.scale_factor.get() != 1.0:
                new_size = (int(image.width * self.scale_factor.get()), int(image.height * self.scale_factor.get()))
                image = image.resize(new_size, Image.LANCZOS)

            # Prepare inputs
            inputs = {
                "image": image,
                "generator": torch.Generator(device='cuda').manual_seed(self.seed.get()),
                "true_cfg_scale": self.true_cfg_scale.get(),
                "negative_prompt": self.negative_prompt.get(),
                "num_inference_steps": self.num_inference_steps.get(),
                "num_images_per_prompt": self.num_images_per_prompt.get(),
                "layers": self.layers.get(),
                "resolution": self.resolution.get(),
                "cfg_normalize": self.cfg_normalize.get(),
                "use_en_prompt": self.use_en_prompt.get(),
            }

            # Run pipeline
            with torch.inference_mode():
                output = self.pipeline(**inputs)
                output_images = output.images[0]

            # Create output directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            image_name = os.path.splitext(os.path.basename(path))[0]
            output_dir = os.path.join(script_dir, image_name)
            os.makedirs(output_dir, exist_ok=True)

            # Save layers
            for i, img in enumerate(output_images):
                save_path = os.path.join(output_dir, f"{i}.png")
                img.save(save_path)

            messagebox.showinfo("Success", f"Layers saved to: {output_dir}")

        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLayeredGUI(root)
    root.mainloop()
