# ImageLab

ImageLab 是古代碑文拓印、古籍扫描和古代手稿图像净化的第一版运行底座。

## 当前能力

- 图片上传与原图只读保存
- 古代碑文拓印、古籍扫描、古代手稿三类处理预设
- 大尺度背景估计与校正
- 轻度平滑和局部对比度增强
- 增强灰度图、文字候选图、透明背景 PNG 输出
- 平衡/强力模式的中值滤波和连通域杂点清理
- 平衡模式保留笔画抗锯齿边缘，避免阈值裁切造成变细和锯齿
- 原图/结果对照预览与参数调节
- 文字候选多边形区域和低置信度复核入口
- 桌面版后台处理和阶段进度反馈
- 大尺寸 TIFF 导入在后台线程执行，界面保持响应
- 启动后默认最大化窗口，净化结果使用 6000 像素级无损预览

## 启动

桌面版推荐直接双击 `start_imagelab.bat` 或 `start_imagelab_desktop.bat`。两个脚本都会自动准备虚拟环境、安装依赖并打开 ImageLab 桌面窗口；网页入口仅供 API/调试使用。

## 桌面视图操作

- `Alt + 鼠标滚轮`：以光标位置为中心缩放，最低 8%，不设固定最大倍率。
- `空格 + 鼠标左键拖动`：平移画布，光标会变为手型。
- 按住鼠标中键拖动：直接平移画布。
- 点击“重置视图”：恢复适合窗口的 100% 视图。

图片导入后先显示异步生成的预览图；当鼠标在图片上使用 `Alt + 滚轮` 放大时，桌面程序会在后台载入原始高清图并自动替换预览。缩放没有固定的最大倍率，实际上限取决于图像分辨率和可用内存。

对于 Qt 无法直接解码的压缩或超大 TIFF，高清显示会由 Pillow 在后台生成有界的灰度显示缓存（按缩放级别逐级提升），避免 RGB 全图超过 Qt 的 256MB 分配限制；原始文件和处理导出结果不受影响。

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

增强灰度图默认将纸张、石材和光照背景归一化到白色，文字保持深色；“文字候选图”和“透明背景 PNG”仍使用独立的保守掩膜。

后续可在 `app/processing.py` 中增加 SegFormer/U-Net++ 推理，并保留当前处理器作为无模型回退路径。

桌面版处理结果保持原始图片的宽高和像素精度。界面先显示轻量预览，放大时异步载入高清图；原始文件和全分辨率处理结果始终保留。

## API

- `GET /api/health`：服务健康检查
- `POST /api/process`：上传图片并生成处理结果

`/api/process` 接收 `file`、`image_type`（`rubbing`、`book`、`manuscript`、`other`）、`mode`（`conservative`、`balanced`、`strong`）和 `keep_faint` 字段，返回结果文件地址、尺寸、总体置信度以及文字候选多边形。多边形只用于显示和复核，不能替代底层像素掩膜。
