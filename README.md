# ImageLab

ImageLab 是古代碑文拓印、古籍扫描和古代手稿图像净化的第一版运行底座。

## 当前能力

- 图片上传与原图只读保存
- 古代碑文拓印、古籍扫描、古代手稿三类处理预设
- 大尺度背景估计与校正
- 轻度平滑和局部对比度增强
- 增强灰度图、文字候选图、透明背景 PNG 输出
- 原图/结果对照预览与参数调节

## 启动

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

浏览器打开 http://127.0.0.1:8000 。

## 目录

```text
app/
  main.py          FastAPI 入口和静态文件服务
  processing.py    基础图像净化流程
frontend/
  index.html       工作台界面
uploads/           运行时原图目录
outputs/           运行时结果目录
```

后续可在 `app/processing.py` 中增加 SegFormer/U-Net++ 推理，并保留当前处理器作为无模型回退路径。
