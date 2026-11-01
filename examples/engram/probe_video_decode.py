"""Read-only NVDEC/CPU video probe. CUDA fallback is explicitly disabled."""

import argparse
import time

import av
from av.codec.hwaccel import HWAccel
from av.codec.hwaccel import hwdevices_available
import numpy as np


def decode(path, mode, count, compare_count, device):
    hw = None if mode == "cpu" else HWAccel("cuda", device=device, allow_software_fallback=False)
    start = time.monotonic()
    container = av.open(path, hwaccel=hw)
    stream = container.streams.video[0]
    if mode == "cpu":
        stream.codec_context.thread_count = 1
    images, forced, retagged, timestamps = [], [], [], []
    for frame in container.decode(stream):
        if frame.pts is None:
            raise ValueError("Decoded frame lacks a presentation timestamp")
        if not timestamps:
            print(f"{mode}_first_frame format={frame.format.name} frame_range={frame.color_range} "
                  f"stream_range={stream.codec_context.color_range} hw_active={stream.codec_context.is_hwaccel}", flush=True)
        if mode == "cuda" and not stream.codec_context.is_hwaccel:
            raise RuntimeError("CUDA decoder was not active; refusing software fallback")
        timestamps.append(float(frame.pts * frame.time_base))
        # PyAV automatically transfers GPU frames to CPU; include this cost.
        rgb = frame.to_ndarray(format="rgb24")
        if len(timestamps) <= compare_count:
            images.append(rgb)
            forced.append(frame.to_ndarray(format="rgb24", src_color_range=2, dst_color_range=2))
            frame.color_range = 2
            retagged.append(frame.to_ndarray(format="rgb24"))
        if len(timestamps) == count:
            break
    elapsed = time.monotonic() - start
    container.close()
    if len(timestamps) != count:
        raise ValueError(f"Only {len(timestamps)} frames decoded, expected {count}")
    print(f"{mode}: frames={count} wall_s={elapsed:.3f} fps={count / elapsed:.2f} shape={images[0].shape}", flush=True)
    return images, forced, retagged, timestamps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--compare-frames", type=int, default=4)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--gpu-scale", action="store_true")
    args = parser.parse_args()
    if args.frames < 1 or not 1 <= args.compare_frames <= args.frames:
        parser.error("Expected 1 <= compare-frames <= frames")
    print("compiled_hwdevices", hwdevices_available(), flush=True)
    cpu, cpu_forced, cpu_retag, cpu_pts = decode(args.video, "cpu", args.frames, args.compare_frames, args.cuda_device)
    gpu, gpu_forced, gpu_retag, gpu_pts = decode(args.video, "cuda", args.frames, args.compare_frames, args.cuda_device)
    np.testing.assert_allclose(cpu_pts, gpu_pts, atol=1e-6, rtol=0)
    for label, reference, candidate in (("gpu_default", cpu, gpu), ("gpu_forced_full", cpu, gpu_forced),
                                         ("gpu_retagged_full", cpu, gpu_retag), ("cpu_forced_full", cpu, cpu_forced)):
        errors = np.concatenate([np.abs(reference[i].astype(np.int16) - candidate[i].astype(np.int16)).ravel()
                                 for i in range(args.compare_frames)])
        print(f"CPU_vs_{label}_rgb_mae={errors.mean():.6f} p99={np.quantile(errors, 0.99):.1f} max={errors.max()}", flush=True)
    if args.gpu_scale:
        import jax
        import jax.numpy as jnp
        if not any(device.platform == "gpu" for device in jax.devices()):
            raise RuntimeError("JAX GPU device unavailable; refusing to report GPU scaling")
        @jax.jit
        def scale(image):
            left = image[:, : image.shape[1] // 2]
            return jax.image.resize(left, (360, 640, 3), method="linear")
        scale(jnp.asarray(gpu_forced[0])).block_until_ready()
        start = time.monotonic()
        for image in gpu_forced[:args.compare_frames]:
            result = np.asarray(scale(jnp.asarray(image)))
        print(f"gpu_scale_including_host_transfers count={args.compare_frames} seconds={time.monotonic() - start:.3f} shape={result.shape}", flush=True)


if __name__ == "__main__":
    main()
