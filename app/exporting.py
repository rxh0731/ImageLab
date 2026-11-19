"""Image exports and genuine Photoshop-compatible layers (not TIFF pages)."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import os
import struct
import tempfile

from PIL import Image


FORMATS = ("tif", "png", "jpg", "psd")
LAYERED_FORMATS = {"tif", "psd"}
TIFF_LAYER_TAG = 37724
TIFF_LAYER_HEADER = b"Adobe Photoshop Document Data Block\x00"


def normalize_format(value: str) -> str:
    value = str(value).lower().lstrip(".")
    value = {"jpeg": "jpg", "tiff": "tif"}.get(value, value)
    if value not in FORMATS:
        raise ValueError(f"不支持的输出格式：{value}")
    return value


def output_path(directory: Path, stem: str, output_format: str = "tif") -> Path:
    return directory / f"{stem}.{normalize_format(output_format)}"


def on_white(image: Image.Image) -> Image.Image:
    result = Image.new("RGB", image.size, "white")
    if "A" in image.getbands():
        result.paste(image, mask=image.getchannel("A"))
    else:
        result.paste(image)
    return result


def _make_document(image: Image.Image, white: bool, dpi: tuple[float, float], text_layer_name: str):
    from psd_tools import PSDImage
    from psd_tools.constants import Compression, Resource
    from psd_tools.psd.image_resources import ImageResource, ResoulutionInfo, VersionInfo

    if max(image.size) > 30000:
        raise ValueError("PSD 图层尺寸不能超过 30000 像素，请降低统一输出 DPI 或使用单层 TIFF/PNG。")
    document = PSDImage.new("RGBA", image.size,
                            color=(0, 0, 0, 0),
                            compression=Compression.RLE)
    if white:
        with Image.new("RGB", image.size, "white") as background:
            layer = document.create_pixel_layer(background, name="白底", compression=Compression.RLE)
            layer.name = "白底"
    layer = document.create_pixel_layer(image, name=text_layer_name, compression=Compression.RLE)
    layer.name = text_layer_name
    document._update_record()
    # Supply an exact Pillow composite instead of float compositing millions
    # of pixels again. The channel data below is lossless, including alpha.
    composite = on_white(image).convert("RGBA") if white else image
    document._record.image_data.set_data([band.tobytes() for band in composite.split()], document._record.header)
    if white:
        composite.close()
    info = document._record.layer_and_mask_information.layer_info
    info.layer_count = -abs(info.layer_count)
    resolution = ResoulutionInfo(horizontal=int(round(dpi[0] * 65536)), horizontal_unit=1, width_unit=1,
                                vertical=int(round(dpi[1] * 65536)), vertical_unit=1, height_unit=1)
    document.image_resources[Resource.RESOLUTION_INFO] = ImageResource(key=Resource.RESOLUTION_INFO, data=resolution)
    document.image_resources[Resource.VERSION_INFO] = ImageResource(
        key=Resource.VERSION_INFO, data=VersionInfo(has_composite=True, writer="ImageLab", reader="ImageLab"))
    return document


def _tiff_layer_data(document) -> bytes:
    from psd_tools.psd.layer_and_mask import LayerInfoBlock
    info = document._record.layer_and_mask_information.layer_info
    data = LayerInfoBlock(layer_count=abs(info.layer_count), layer_records=info.layer_records,
                         channel_image_data=info.channel_image_data).tobytes()
    return TIFF_LAYER_HEADER + b"8BIMLayr" + struct.pack(">I", len(data)) + data + b"\x00" * (-len(data) % 4)


def read_layered_tiff(path: Path):
    """Return the embedded PSD layer tree, or None for an ordinary TIFF."""
    from psd_tools import PSDImage
    from psd_tools.psd.layer_and_mask import LayerInfoBlock
    with Image.open(path) as image:
        blob = image.tag_v2.get(TIFF_LAYER_TAG) if hasattr(image, "tag_v2") else None
        size = image.size
    if not blob or not bytes(blob).startswith(TIFF_LAYER_HEADER):
        return None
    stream = BytesIO(bytes(blob)[len(TIFF_LAYER_HEADER):])
    while len(header := stream.read(12)) == 12:
        signature, key, length = struct.unpack(">4s4sI", header)
        if signature != b"8BIM":
            raise ValueError("TIFF 图层数据无效")
        data = stream.read(length)
        if len(data) != length:
            raise ValueError("TIFF 图层数据不完整")
        if key == b"Layr":
            document = PSDImage.new("RGB", size)
            document._record.layer_and_mask_information.layer_info = LayerInfoBlock.frombytes(data)
            document._init()
            return document
        stream.seek(-length % 4, 1)
    return None


def read_text_image(path: Path) -> Image.Image:
    """Load editable text, preserving layer alpha even with a white bottom layer."""
    document = None
    if path.suffix.lower() == ".psd":
        from psd_tools import PSDImage
        document = PSDImage.open(path)
    elif path.suffix.lower() in {".tif", ".tiff"}:
        # Ordinary TIFFs need no PSD parsing or extra allocation.
        with Image.open(path) as image:
            if TIFF_LAYER_TAG in getattr(image, "tag_v2", {}):
                document = read_layered_tiff(path)
    if document is not None and len(document):
        layer = next((layer for layer in reversed(list(document)) if layer.name == "文字笔画"), document[-1])
        pixels = layer.topil(apply_icc=False)
        canvas = Image.new("RGBA", document.size, (0, 0, 0, 0))
        if pixels is not None:
            canvas.paste(pixels, (layer.left, layer.top))
            pixels.close()
        return canvas
    with Image.open(path) as image:
        return image.convert("RGBA")


def read_dpi(path: Path, fallback: float = 300.0) -> tuple[float, float]:
    if path.suffix.lower() == ".psd":
        # Read only the PSD header/resources, without decoding its layer planes.
        from psd_tools.constants import Resource
        from psd_tools.psd.header import FileHeader
        from psd_tools.psd.color_mode_data import ColorModeData
        from psd_tools.psd.image_resources import ImageResources
        with path.open("rb") as stream:
            FileHeader.read(stream)
            ColorModeData.read(stream)
            resources = ImageResources.read(stream)
        info = resources.get_data(Resource.RESOLUTION_INFO)
        return (info.horizontal / 65536, info.vertical / 65536) if info else (fallback, fallback)
    with Image.open(path) as image:
        value = image.info.get("dpi", (fallback, fallback))
    if not isinstance(value, (tuple, list)):
        value = (value, value)
    try:
        x, y = float(value[0]), float(value[1])
        return (x if 0 < x < float("inf") else fallback, y if 0 < y < float("inf") else fallback)
    except (TypeError, ValueError, IndexError):
        return fallback, fallback


def export_image(image: Image.Image, destination: Path, output_format: str = "tif",
                 white_background: bool = False, dpi: tuple[float, float] = (300, 300),
                 overwrite: bool = True, progress_callback=None) -> None:
    fmt = normalize_format(output_format)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and destination.exists():
        raise FileExistsError(destination)
    def progress(value, message):
        if progress_callback:
            progress_callback(value, message)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.stem}_", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        progress(10, "准备保存图片")
        if fmt == "jpg":
            with on_white(image) as composite:
                composite.save(temporary, format="JPEG", quality=100, subsampling=0, dpi=dpi)
        elif fmt == "png":
            image.save(temporary, format="PNG", compress_level=3, dpi=dpi)
        elif fmt == "psd":
            progress(25, "构建 Photoshop 图层")
            document = _make_document(image, white_background, dpi, destination.stem)
            progress(70, "写入 PSD 图层和合成预览")
            with temporary.open("wb") as stream:
                document._record.write(stream)
        elif white_background:
            import numpy as np
            import tifffile

            progress(25, "构建 TIFF 文字层和白底层")
            document = _make_document(image, True, dpi, destination.stem)
            layer_data = _tiff_layer_data(document)
            progress(70, "写入分层 TIFF")
            with on_white(image) as composite:
                # Photoshop interprets ImageSourceData using the TIFF byte
                # order. PSD layer records are big endian, so the containing
                # TIFF must also be big endian (Pillow writes RGB as little
                # endian and Photoshop then rejects otherwise valid layers).
                tifffile.imwrite(temporary, np.asarray(composite), byteorder=">",
                                 photometric="rgb", metadata=None,
                                 compression="deflate", compressionargs={"level": 3},
                                 resolution=dpi, resolutionunit="INCH",
                                 extratags=[(TIFF_LAYER_TAG, 7, len(layer_data), layer_data, False)])
        else:
            image.save(temporary, format="TIFF", compression="tiff_lzw", dpi=dpi)
        progress(95, "完成文件写入")
        if overwrite:
            os.replace(temporary, destination)
        else:
            # Same-directory hard-link creation atomically refuses a file
            # created by somebody else while this batch item was processing.
            os.link(temporary, destination)
            temporary.unlink()
        progress(100, "保存完成")
    finally:
        temporary.unlink(missing_ok=True)
