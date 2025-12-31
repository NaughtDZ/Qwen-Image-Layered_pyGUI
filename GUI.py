import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image
import torch
from diffusers import QwenImageLayeredPipeline

class ImageLayeredGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Qwen Image Layered GUI")
        self.root.geometry("600x650")

        # Variables 同上
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

        # GUI 布局（保持不变，略）

        frame = ttk.Frame(root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(frame, text="图片路径 (支持直接粘贴):").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=self.image_path, width=60).grid(row=0, column=1, pady=5)
        ttk.Button(frame, text="浏览", command=self.browse_image).grid(row=0, column=2, pady=5)

        ttk.Label(frame, text="等比例缩放因子 (1.0=原尺寸, 0.5=一半):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=self.scale_factor, width=10).grid(row=1, column=1, sticky=tk.W, pady=5)

        ttk.Label(frame, text="图层数量:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(frame, from_=1, to=12, textvariable=self.layers, width=8).grid(row=2, column=1, sticky=tk.W, pady=5)

        ttk.Label(frame, text="分辨率桶 (推荐640):").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(frame, textvariable=self.resolution, values=[640, 1024], state="readonly", width=8).grid(row=3, column=1, sticky=tk.W, pady=5)

        ttk.Label(frame, text="推理步数:").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(frame, from_=10, to=100, textvariable=self.num_inference_steps, width=8).grid(row=4, column=1, sticky=tk.W, pady=5)

        ttk.Label(frame, text="True CFG Scale:").grid(row=5, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=self.true_cfg_scale, width=10).grid(row=5, column=1, sticky=tk.W, pady=5)

        ttk.Label(frame, text="Negative Prompt:").grid(row=6, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=self.negative_prompt, width=60).grid(row=6, column=1, columnspan=2, pady=5)

        ttk.Label(frame, text="每提示生成数量:").grid(row=7, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(frame, from_=1, to=5, textvariable=self.num_images_per_prompt, width=8).grid(row=7, column=1, sticky=tk.W, pady=5)

        ttk.Checkbutton(frame, text="CFG Normalize", variable=self.cfg_normalize).grid(row=8, column=0, sticky=tk.W, pady=5)
        ttk.Checkbutton(frame, text="Use English Prompt (自动描述语言)", variable=self.use_en_prompt).grid(row=8, column=1, sticky=tk.W, pady=5)

        ttk.Label(frame, text="随机种子 (-1 为随机):").grid(row=9, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=self.seed, width=10).grid(row=9, column=1, sticky=tk.W, pady=5)

        ttk.Button(frame, text="开始拆分图层", command=self.process_image).grid(row=10, column=0, columnspan=3, pady=20)

        self.progress = ttk.Progressbar(frame, orient="horizontal", mode="indeterminate")
        self.progress.grid(row=11, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        self.pipeline = None
        self.load_pipeline()

    def load_pipeline(self):
        try:
            messagebox.showinfo("加载模型", "正在加载 Qwen-Image-Layered 模型（首次会下载约57GB，请耐心等待）...")
            # 关键修改：移除 variant，只用 torch_dtype
            self.pipeline = QwenImageLayeredPipeline.from_pretrained(
                "Qwen/Qwen-Image-Layered",
                torch_dtype=torch.bfloat16
            )
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
            messagebox.showinfo("成功", "模型加载完成！现在可以处理图片了。")
        except Exception as e:
            messagebox.showerror("加载失败", f"无法加载模型：{str(e)}\n\n"
                                            "常见原因：\n"
                                            "1. diffusers 不是从 git 安装（必须用下面命令）\n"
                                            "   pip install git+https://github.com/huggingface/diffusers\n"
                                            "2. 显存不足（需要 >=24GB）\n"
                                            "3. 网络问题导致下载中断")
            self.root.quit()

    # browse_image 和 process_image 函数保持完全不变（包括缩放、保存子目录、预览图等）

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        if file_path:
            self.image_path.set(file_path)

    def process_image(self):
        path = self.image_path.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("错误", "图片路径无效，请检查是否正确复制粘贴或选择文件。")
            return

        self.progress.start()
        self.root.update()

        try:
            image = Image.open(path).convert("RGBA")
            original_size = image.size
            if self.scale_factor.get() != 1.0:
                new_size = (int(image.width * self.scale_factor.get()), int(image.height * self.scale_factor.get()))
                image = image.resize(new_size, Image.LANCZOS)

            seed_val = self.seed.get()
            if seed_val == -1:
                generator = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu')
                generator = generator.manual_seed(torch.randint(0, 2**32, (1,)).item())
            else:
                generator = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu').manual_seed(seed_val)

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
                save_path = os.path.join(output_dir, f"layer_{i:02d}.png")
                layer_img.save(save_path)

            # 合并预览图
            preview = Image.new("RGBA", image.size)
            for layer in output_images:
                preview = Image.alpha_composite(preview, layer)
            preview.save(os.path.join(output_dir, "preview_combined.png"))

            messagebox.showinfo("完成！", f"成功拆分 {len(output_images)} 个图层！\n保存目录：{output_dir}")

        except Exception as e:
            messagebox.showerror("处理失败", f"出错：{str(e)}")
        finally:
            self.progress.stop()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLayeredGUI(root)
    root.mainloop()
