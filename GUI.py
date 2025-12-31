import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image
import torch
from diffusers import QwenImageLayeredPipeline

class ImageLayeredGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Qwen Image Layered GUI (FP8 低显存版)")
        self.root.geometry("700x750")

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
        self.enable_offload = tk.BooleanVar(value=True)  # 新增：默认开启 offload 省显存

        self.original_size = (0, 0)
        self.scaled_size_label = tk.StringVar(value="未加载图片")

        # GUI 布局
        frame = ttk.Frame(root, padding="15")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        ttk.Label(frame, text="图片路径 (支持直接粘贴):").grid(row=0, column=0, sticky=tk.W, pady=8)
        ttk.Entry(frame, textvariable=self.image_path, width=60).grid(row=0, column=1, pady=8, padx=5)
        ttk.Button(frame, text="浏览", command=self.browse_image).grid(row=0, column=2, pady=8)

        # 缩放滑块
        ttk.Label(frame, text="等比例缩放:").grid(row=1, column=0, sticky=tk.W, pady=8)
        scale_frame = ttk.Frame(frame)
        scale_frame.grid(row=1, column=1, sticky=tk.W, pady=8)
        self.slider = ttk.Scale(scale_frame, from_=0.1, to=3.0, orient=tk.HORIZONTAL,
                                variable=self.scale_factor, length=400, command=self.update_scaled_size)
        self.slider.grid(row=0, column=0)
        ttk.Label(scale_frame, textvariable=self.scale_factor, width=5).grid(row=0, column=1, padx=(10,0))
        ttk.Label(scale_frame, text="×").grid(row=0, column=2)
        ttk.Label(scale_frame, textvariable=self.scaled_size_label, foreground="blue").grid(row=1, column=0, columnspan=3, pady=5)

        # 其他参数（保持不变）
        ttk.Label(frame, text="图层数量:").grid(row=2, column=0, sticky=tk.W, pady=8)
        ttk.Spinbox(frame, from_=1, to=12, textvariable=self.layers, width=8).grid(row=2, column=1, sticky=tk.W, pady=8)

        ttk.Label(frame, text="分辨率桶 (推荐640):").grid(row=3, column=0, sticky=tk.W, pady=8)
        ttk.Combobox(frame, textvariable=self.resolution, values=[640, 1024], state="readonly", width=8).grid(row=3, column=1, sticky=tk.W, pady=8)

        ttk.Label(frame, text="推理步数:").grid(row=4, column=0, sticky=tk.W, pady=8)
        ttk.Spinbox(frame, from_=10, to=100, textvariable=self.num_inference_steps, width=8).grid(row=4, column=1, sticky=tk.W, pady=8)

        ttk.Label(frame, text="True CFG Scale:").grid(row=5, column=0, sticky=tk.W, pady=8)
        ttk.Entry(frame, textvariable=self.true_cfg_scale, width=10).grid(row=5, column=1, sticky=tk.W, pady=8)

        ttk.Label(frame, text="Negative Prompt:").grid(row=6, column=0, sticky=tk.W, pady=8)
        ttk.Entry(frame, textvariable=self.negative_prompt, width=70).grid(row=6, column=1, columnspan=2, sticky=tk.W+tk.E, pady=8)

        ttk.Label(frame, text="每提示生成数量:").grid(row=7, column=0, sticky=tk.W, pady=8)
        ttk.Spinbox(frame, from_=1, to=5, textvariable=self.num_images_per_prompt, width=8).grid(row=7, column=1, sticky=tk.W, pady=8)

        ttk.Checkbutton(frame, text="CFG Normalize", variable=self.cfg_normalize).grid(row=8, column=0, sticky=tk.W, pady=5)
        ttk.Checkbutton(frame, text="Use English Prompt", variable=self.use_en_prompt).grid(row=8, column=1, sticky=tk.W, pady=5)

        ttk.Checkbutton(frame, text="启用 CPU Offload (强烈推荐，显著省显存)", variable=self.enable_offload).grid(row=9, column=0, columnspan=2, sticky=tk.W, pady=10)

        ttk.Label(frame, text="随机种子 (-1 为随机):").grid(row=10, column=0, sticky=tk.W, pady=8)
        ttk.Entry(frame, textvariable=self.seed, width=10).grid(row=10, column=1, sticky=tk.W, pady=8)

        ttk.Button(frame, text="开始拆分图层", command=self.process_image, style="Accent.TButton").grid(row=11, column=0, columnspan=3, pady=25)

        self.progress = ttk.Progressbar(frame, orient="horizontal", mode="indeterminate")
        self.progress.grid(row=12, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        self.pipeline = None
        self.load_pipeline()

    def update_scaled_size(self, val=None):
        if self.original_size == (0, 0):
            self.scaled_size_label.set("未加载图片")
            return
        scale = self.scale_factor.get()
        new_w = int(self.original_size[0] * scale)
        new_h = int(self.original_size[1] * scale)
        self.scaled_size_label.set(f"{new_w} × {new_h}")

    def load_image_info(self, path):
        try:
            with Image.open(path) as img:
                self.original_size = img.size
                self.update_scaled_size()
        except Exception:
            self.original_size = (0, 0)
            self.scaled_size_label.set("图片加载失败")
			
    def load_pipeline(self):
        try:
            messagebox.showinfo("加载模型", "正在加载官方 Qwen-Image-Layered（bf16版，首次~57GB）...\n"
                                            "已自动启用多项显存优化，适合32GB显存")
            self.pipeline = QwenImageLayeredPipeline.from_pretrained(
                "Qwen/Qwen-Image-Layered",  # 官方完整仓库
                torch_dtype=torch.bfloat16
            )
            if torch.cuda.is_available():
                self.pipeline.to("cuda")

            # === 关键显存优化（强烈推荐全部开启）===
            self.pipeline.enable_model_cpu_offload()       # 最有效：模型层动态在CPU/GPU间切换，显存大幅降低（推荐首选）
            # 如果还爆显存，可取消上面一行，改用下面这行（更省但更慢）：
            # self.pipeline.enable_sequential_cpu_offload()

            self.pipeline.enable_vae_slicing()             # VAE 分片，省一点
            # self.pipeline.enable_attention_slicing()     # Attention 分片，进一步省（可略微影响速度）

            messagebox.showinfo("成功", "模型加载完成！\n"
                                        "已启用 CPU offload 等优化，你的32GB显存应该够用了～\n"
                                        "建议：resolution=640 + 缩放0.6~0.8 更稳")
        except Exception as e:
            messagebox.showerror("加载失败", f"出错：{str(e)}\n\n"
                                            "请确认已安装最新版：\n"
                                            "pip install git+https://github.com/huggingface/diffusers\n"
                                            "pip install -U transformers accelerate")
            self.root.quit()

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.webp *.tiff")])
        if file_path:
            self.image_path.set(file_path)
            self.load_image_info(file_path)

    def process_image(self):
        # （处理逻辑完全不变，只加了 offload 动态控制）
        path = self.image_path.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("错误", "图片路径无效")
            return

        self.load_image_info(path)
        if self.original_size == (0, 0):
            messagebox.showerror("错误", "无法读取图片")
            return

        # 动态控制 offload
        if self.pipeline and self.enable_offload.get():
            self.pipeline.enable_model_cpu_offload()

        self.progress.start()
        self.root.update_idletasks()

        try:
            image = Image.open(path).convert("RGBA")
            scale = self.scale_factor.get()
            if scale != 1.0:
                new_size = (int(image.width * scale), int(image.height * scale))
                image = image.resize(new_size, Image.LANCZOS)

            seed_val = self.seed.get()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if seed_val == -1:
                generator = torch.Generator(device=device).manual_seed(torch.randint(0, 2**32, (1,)).item())
            else:
                generator = torch.Generator(device=device).manual_seed(seed_val)

            inputs = {
                "image": image,
                "generator": generator,
                "true_cfg_scale": self.true_cfg_scale.get(),
                "negative_prompt": self.negative_prompt.get(),
                "num_inference_steps": self.num_inference_steps.get(),
                "num_images_per_prompt": self.num_images_per_prompt.get(),
                "layers": self.layers.get(),
                "resolution": self.resolution.get(),
                "cfg_normalize": self.cfg_normalize.get(),
                "use_en_prompt": self.use_en_prompt.get(),
            }

            with torch.inference_mode():
                output = self.pipeline(**inputs)

            output_images = output.images[0]

            script_dir = os.path.dirname(os.path.abspath(__file__))
            image_name = os.path.splitext(os.path.basename(path))[0]
            output_dir = os.path.join(script_dir, image_name)
            os.makedirs(output_dir, exist_ok=True)

            for i, layer_img in enumerate(output_images):
                layer_img.save(os.path.join(output_dir, f"layer_{i:02d}.png"))

            preview = Image.new("RGBA", image.size)
            for layer in output_images:
                preview = Image.alpha_composite(preview, layer)
            preview.save(os.path.join(output_dir, "preview_combined.png"))

            messagebox.showinfo("完成！", f"成功拆分 {len(output_images)} 个图层！\n保存路径：{output_dir}")

        except Exception as e:
            messagebox.showerror("处理失败", f"出错：{str(e)}")
        finally:
            self.progress.stop()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLayeredGUI(root)
    root.mainloop()
