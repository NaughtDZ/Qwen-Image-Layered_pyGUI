# Qwen-Image-Layered_pyGUI
pyGUI For Qwen-Image-Layered

使用PY脚本制作的Qwen-Image-Layered GUI 运行工具

https://huggingface.co/Qwen/Qwen-Image-Layered

首先建议新建一个目录，然后目录里用uv venv venv创建虚拟环境

然后 uv pip install pip 防止怪异bug

接下来  uv pip install git+https://github.com/huggingface/diffusers

然后 uv pip install python-pptx

然后 请到 https://pytorch.org/get-started/locally/  搞个pytorch的安装命令，如： uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

然后 uv pip install transformers

然后 uv pip install accelerate

最后，启动虚拟环境，把GUI.py放到项目目录里，拖动py文件到激活虚拟环境的cmd窗口回车运行，好好享受！

注意，初次运行会从huggingface下载模型，注意网络环境。同时额外提一句，建议各位把C盘的huggingface缓存换个盘存，然后mklink软链接回去，节约C盘空间！
