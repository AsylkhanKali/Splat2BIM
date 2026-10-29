#!/usr/bin/env python3
"""从视频抽取图片。需要 Python 3.9+、FFmpeg 和 ffprobe，无需第三方 Python 包。

直接运行：修改下面的“用户设置区”，填入 VIDEO_PATH，然后运行本文件。
不传命令行参数时读取用户设置区；传入命令行参数时按命令行运行。

命令行示例（路径可包含中文和空格）：
  python extract_video_frames.py "D:\\Videos\\scan.mp4"
  python extract_video_frames.py "D:\\Videos\\scan.mp4" --fps 2
  python extract_video_frames.py "D:\\Videos\\scan.mov" --interval 0.5 --format png
  python extract_video_frames.py "D:\\Videos\\scan.mp4" --every-n 30
  python extract_video_frames.py "D:\\Videos\\scan.mp4" --start 10 --end 60 --max-frames 100

默认每秒抽一张，输出到视频旁边的 <视频名>_frames 文件夹。
--output 可指定其他全新目录；已有目录不会被覆盖。
图片按 frame_00000.jpg 编号；frames.csv 记录输出编号和时间戳。
时间戳相对于第一个视频帧，来自实际解码帧的 PTS，不用帧号/平均帧率估算。
--start 包含起点，--end 不包含终点，单位均为秒。
时间采样保留每个时间区间内的第一张可用帧，不插帧或重复帧。
--every-n 从所选时间范围的第一张帧起计数；--every-n 1 导出全部帧。
保留分辨率，按视频方向元数据自动转正；JPG 为有损图片，PNG 为 8 位 RGB。
该工具仅抽图，不生成相机位姿、内参或深度，不执行 HDR 色调映射。

FFmpeg 参考：https://ffmpeg.org/ffmpeg-filters.html#select_002c-aselect
"""

# ==================== 用户设置区：修改这里即可 ====================
VIDEO_PATH = r"D:\MCP\MEP3\Old/Video_raw.mp4"                 # 必填：视频路径，例如 r"D:\Videos\scan.mp4"
OUTPUT_DIR = None                # 输出文件夹；None 表示视频旁的 <视频名>_frames

SAMPLE_MODE = "fps"              # "fps" / "interval" / "every_n"，三选一
TARGET_FPS = 4                 # fps 模式：每秒抽几张，例如 2.0
INTERVAL_SECONDS = 0.5           # interval 模式：每隔几秒抽一张
EVERY_N_FRAMES = 30              # every_n 模式：每 N 帧抽一张；1 表示全部帧

START_SECONDS = 0.0             # 从第几秒开始，包含起点
END_SECONDS = None              # 到第几秒结束，不含终点；None 表示到视频结束
MAX_FRAMES = None               # 最多抽取多少张；None 表示不限
IMAGE_FORMAT = "jpg"            # "jpg" 或 "png"
JPEG_QUALITY = 2                # JPG 质量：2–31，越小质量越高

FFMPEG_PATH = "ffmpeg"           # 通常无需修改，也可填写 ffmpeg.exe 的完整路径
FFPROBE_PATH = None             # None 表示自动查找 ffprobe
# ==================== 用户设置区结束 ====================

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的有限数字")
    return number


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("必须是大于或等于 0 的有限数字")
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def find_tool(value):
    found = shutil.which(value)
    if found:
        return str(Path(found).resolve())
    raise RuntimeError(f"找不到 {value}。请安装 FFmpeg 并加入 PATH，或用 --ffmpeg / --ffprobe 指定程序路径。")


def run_options():
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def probe_video(ffprobe, video):
    command = [ffprobe, "-v", "error", "-select_streams", "v:0",
               "-show_streams", "-show_format", "-of", "json", str(video)]
    result = subprocess.run(command, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", **run_options())
    if result.returncode:
        raise RuntimeError("无法读取视频：\n" + result.stderr[-3000:])
    info = json.loads(result.stdout)
    if not info.get("streams"):
        raise RuntimeError("文件中没有可用的视频轨道")
    return info


def extract(args):
    video = Path(args.video).expanduser().resolve()
    if not video.is_file():
        raise RuntimeError(f"找不到视频文件：{video}")
    if args.end is not None and args.end <= args.start:
        raise RuntimeError("--end 必须大于 --start")
    output = (Path(args.output).expanduser().resolve() if args.output
              else video.with_name(video.stem + "_frames"))
    if output.exists():
        raise RuntimeError(f"输出目录已存在，请用 --output 指定新目录：{output}")
    ffmpeg = find_tool(args.ffmpeg)
    sibling = Path(ffmpeg).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
    ffprobe = find_tool(args.ffprobe or (str(sibling) if sibling.is_file() else "ffprobe"))
    source_info = probe_video(ffprobe, video)

    # Decode from the beginning to retain accurate timestamps and sampling phase,
    # including variable-frame-rate video. Do not use input -ss seeking here.
    filters = ["setpts=PTS-STARTPTS"]
    trim = f"trim=start={args.start:.12g}"
    if args.end is not None:
        trim += f":end={args.end:.12g}"
    filters.append(trim)
    if args.every_n is not None:
        filters.append(f"select='not(mod(n,{args.every_n}))'")
        sampling = {"mode": "every_n", "every_n": args.every_n}
    else:
        rate = args.fps if args.fps is not None else 1.0 / (args.interval if args.interval is not None else 1.0)
        if not math.isfinite(rate) or rate <= 0:
            raise RuntimeError("抽帧频率超出可用范围")
        # Select first available frame in each time bucket, avoiding timing drift.
        bucket = f"floor((t-{args.start:.12g})*{rate:.12g}+1e-8)"
        previous = f"floor((prev_selected_t-{args.start:.12g})*{rate:.12g}+1e-8)"
        filters.append(f"select='if(isnan(prev_selected_t),1,gt({bucket},{previous}))'")
        sampling = {"mode": "time_buckets", "target_fps": rate, "interval_seconds": 1.0 / rate}
    if args.max_frames is not None:
        filters.append(f"trim=end_frame={args.max_frames}")
    filters.append("showinfo=checksum=0")

    command = [ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-n", "-loglevel", "info",
               "-i", str(video), "-map", "0:v:0", "-an", "-sn", "-dn",
               "-vf", ",".join(filters), "-fps_mode", "passthrough", "-threads", "2"]
    if args.format == "jpg":
        command += ["-c:v", "mjpeg", "-q:v", str(args.jpeg_quality), "-pix_fmt", "yuvj444p"]
    else:
        command += ["-c:v", "png", "-pix_fmt", "rgb24", "-compression_level", "3"]
    command += ["-start_number", "0", f"frame_%05d.{args.format}"]
    output.mkdir(parents=True, exist_ok=False)
    metadata_path = output / "extraction_info.json"
    metadata = {"status": "running", "source_video": str(video), "output": str(output),
                "sampling": sampling, "start_seconds": args.start, "end_seconds": args.end,
                "max_frames": args.max_frames, "format": args.format,
                "timestamp_origin": "first decoded video frame = 0 seconds",
                "csv_index": "output image index, not original video frame index",
                "auto_rotation": True, "hdr_tone_mapping": False,
                "source_probe": source_info, "ffmpeg_command": command}

    def save_metadata():
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    save_metadata()
    print(f"视频：{video}\n输出：{output}", flush=True)
    timestamps = []
    time_base = None
    time_base_re = re.compile(r"config in time_base:\s*(\d+)/(\d+)")
    frame_re = re.compile(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:")
    try:
        with (output / "ffmpeg.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=str(output), stdout=subprocess.DEVNULL,
                                       stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                       errors="replace", **run_options())
            try:
                for line in process.stderr:
                    log.write(line)
                    if "showinfo" not in line:
                        continue
                    tb_match = time_base_re.search(line)
                    if tb_match:
                        time_base = Fraction(int(tb_match[1]), int(tb_match[2]))
                    frame_match = frame_re.search(line)
                    if frame_match:
                        if time_base is None:
                            raise RuntimeError("无法解析视频时间基准，请查看 ffmpeg.log")
                        timestamps.append((int(frame_match[2]), time_base))
                        if len(timestamps) == 1 or len(timestamps) % 100 == 0:
                            print(f"已抽取 {len(timestamps)} 帧，视频时间 {float(int(frame_match[2]) * time_base):.3f} 秒", flush=True)
                code = process.wait()
                if code:
                    raise RuntimeError(f"FFmpeg 退出码 {code}；详情见 {output / 'ffmpeg.log'}")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                process.stderr.close()
        images = sorted(output.glob(f"frame_*.{args.format}"), key=lambda p: int(p.stem.split("_")[1]))
        if not images:
            raise RuntimeError("没有抽取到图片，请检查视频和 --start / --end 范围")
        if len(images) != len(timestamps):
            raise RuntimeError(f"图片数量 {len(images)} 与时间戳数量 {len(timestamps)} 不一致，请查看 ffmpeg.log")
        with (output / "frames.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["output_index", "filename", "timestamp_seconds", "pts", "time_base"])
            for i, (image, (pts, tb)) in enumerate(zip(images, timestamps)):
                if not image.stat().st_size:
                    raise RuntimeError(f"输出图片为空：{image.name}")
                writer.writerow([i, image.name, f"{float(pts * tb):.9f}", pts, str(tb)])
        metadata.update(status="complete", frame_count=len(images),
                        first_timestamp_seconds=float(timestamps[0][0] * timestamps[0][1]),
                        last_timestamp_seconds=float(timestamps[-1][0] * timestamps[-1][1]))
        save_metadata()
        print(f"完成：{len(images)} 张图片。时间戳文件：{output / 'frames.csv'}", flush=True)
    except BaseException as exc:
        metadata.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        save_metadata()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="输入视频文件路径")
    parser.add_argument("-o", "--output", help="输出到一个全新的文件夹")
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--fps", type=positive_float, help="目标每秒抽取数量，如 2；不会重复或合成帧")
    sampling.add_argument("--interval", type=positive_float, help="抽帧时间间隔（秒），默认 1")
    sampling.add_argument("--every-n", type=positive_int, help="每 N 个解码帧抽一张；1 表示全部帧")
    parser.add_argument("--start", type=nonnegative_float, default=0., help="起始秒数，默认 0")
    parser.add_argument("--end", type=nonnegative_float, help="结束秒数，不包含这个时间点")
    parser.add_argument("--max-frames", type=positive_int, help="最多导出的图片数量")
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg", help="输出图片格式，默认 jpg")
    parser.add_argument("--jpeg-quality", type=int, choices=range(2, 32), default=2,
                        help="JPG 质量参数 2–31，越小质量越高，默认 2")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg 程序路径，默认从 PATH 查找")
    parser.add_argument("--ffprobe", help="ffprobe 程序路径，默认查找 ffmpeg 同目录或 PATH")
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        if not VIDEO_PATH or not str(VIDEO_PATH).strip():
            parser.error('请先在脚本顶部的用户设置区填写 VIDEO_PATH，例如 VIDEO_PATH = r"D:\\Videos\\scan.mp4"')
        modes = {
            "fps": ("--fps", TARGET_FPS),
            "interval": ("--interval", INTERVAL_SECONDS),
            "every_n": ("--every-n", EVERY_N_FRAMES),
        }
        if SAMPLE_MODE not in modes:
            parser.error('用户设置区的 SAMPLE_MODE 必须是 "fps"、"interval" 或 "every_n"')
        sample_flag, sample_value = modes[SAMPLE_MODE]
        argv = [str(VIDEO_PATH), sample_flag, str(sample_value),
                "--start", str(START_SECONDS), "--format", str(IMAGE_FORMAT),
                "--jpeg-quality", str(JPEG_QUALITY), "--ffmpeg", str(FFMPEG_PATH)]
        for flag, value in [("--output", OUTPUT_DIR), ("--end", END_SECONDS),
                            ("--max-frames", MAX_FRAMES), ("--ffprobe", FFPROBE_PATH)]:
            if value is not None:
                argv.extend([flag, str(value)])
    args = parser.parse_args(argv)
    try:
        extract(args)
    except KeyboardInterrupt:
        print("\n已中断。已生成文件保留在输出目录中。", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
