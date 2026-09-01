# ImageLab

ImageLab 是古代碑文拓印、古籍扫描和古代手稿图像净化的第一版运行底座。

## 当前能力

- 图片上传与原图只读保存
- 古代碑文拓印、古籍扫描、古代手稿三类处理预设
- 大尺度背景估计与校正
- 轻度平滑和局部对比度增强
- 增强灰度图、文字候选图、透明背景 PNG 输出
- 原图/结果对照预览与参数调节
- 文字候选多边形区域和低置信度复核入口
- 桌面版后台处理和阶段进度反馈

## 启动

桌面版推荐直接双击 `start_imagelab_desktop.bat`。脚本会自动准备虚拟环境、安装依赖并打开 ImageLab 桌面窗口。`start_imagelab.bat` 保留为 API/调试用启动方式，不是最终用户入口。

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

桌面版交互处理会将超大图片缩放到最长边 4096 像素以内，以保证预览可响应；原始文件始终保留。需要文献级全分辨率导出时，应另行增加离线导出任务。

## API

- `GET /api/health`：服务健康检查
- `POST /api/process`：上传图片并生成处理结果

`/api/process` 接收 `file`、`image_type`（`rubbing`、`book`、`manuscript`、`other`）、`mode`（`conservative`、`balanced`、`strong`）和 `keep_faint` 字段，返回结果文件地址、尺寸、总体置信度以及文字候选多边形。多边形只用于显示和复核，不能替代底层像素掩膜。
