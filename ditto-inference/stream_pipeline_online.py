import threading
import queue
import time
import numpy as np
import traceback
from tqdm import tqdm

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from core.atomic_components.avatar_registrar import AvatarRegistrar, smooth_x_s_info_lst
from core.atomic_components.condition_handler import ConditionHandler, _mirror_index
from core.atomic_components.audio2motion import Audio2Motion
from core.atomic_components.motion_stitch import MotionStitch
from core.atomic_components.warp_f3d import WarpF3D
from core.atomic_components.decode_f3d import DecodeF3D
from core.atomic_components.putback import PutBack
from core.atomic_components.writer import VideoWriterByImageIO
from core.atomic_components.wav2feat import Wav2Feat
from core.atomic_components.cfg import parse_cfg, print_cfg


# ---------------------------------------------------------------------------
# GPU helper: create a CUDA stream if CUDA is available, else return None.
# Called INSIDE each worker thread so the stream belongs to the correct
# CUDA context and device affinity is inherited from the parent process.
# ---------------------------------------------------------------------------
def _make_cuda_stream():
    """Return a new torch.cuda.Stream, or None if CUDA is unavailable."""
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.cuda.Stream()
    return None


def _cuda_stream_ctx(stream):
    """Context manager: use the given CUDA stream, or a no-op if None."""
    if stream is not None:
        return torch.cuda.stream(stream)
    import contextlib
    return contextlib.nullcontext()


def _autocast_ctx():
    """bfloat16 autocast for SM90a (H200) tensor cores, no-op otherwise."""
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    import contextlib
    return contextlib.nullcontext()


"""
avatar_registrar_cfg:
    insightface_det_cfg,
    landmark106_cfg,
    landmark203_cfg,
    landmark478_cfg,
    appearance_extractor_cfg,
    motion_extractor_cfg,

condition_handler_cfg:
    use_emo=True,
    use_sc=True,
    use_eye_open=True,
    use_eye_ball=True,
    seq_frames=80,

wav2feat_cfg:
    w2f_cfg, 
    w2f_type
"""


class StreamSDK:
    def __init__(self, cfg_pkl, data_root, **kwargs):

        [
            avatar_registrar_cfg,
            condition_handler_cfg,
            lmdm_cfg,
            stitch_network_cfg,
            warp_network_cfg,
            decoder_cfg,
            wav2feat_cfg,
            default_kwargs,
        ] = parse_cfg(cfg_pkl, data_root, kwargs)
        
        self.default_kwargs = default_kwargs
        
        self.avatar_registrar = AvatarRegistrar(**avatar_registrar_cfg)
        self.condition_handler = ConditionHandler(**condition_handler_cfg)
        self.audio2motion = Audio2Motion(lmdm_cfg)
        self.motion_stitch = MotionStitch(stitch_network_cfg)
        self.warp_f3d = WarpF3D(warp_network_cfg)
        self.decode_f3d = DecodeF3D(decoder_cfg)
        self.putback = PutBack()

        self.wav2feat = Wav2Feat(**wav2feat_cfg)

    def _merge_kwargs(self, default_kwargs, run_kwargs):
        for k, v in default_kwargs.items():
            if k not in run_kwargs:
                run_kwargs[k] = v
        return run_kwargs

    def setup_Nd(self, N_d, fade_in=-1, fade_out=-1, ctrl_info=None):
        # for eye open at video end
        self.motion_stitch.set_Nd(N_d)

        # for fade in/out alpha
        if ctrl_info is None:
            ctrl_info = self.ctrl_info
        if fade_in > 0:
            for i in range(fade_in):
                alpha = i / fade_in
                item = ctrl_info.get(i, {})
                item["fade_alpha"] = alpha
                ctrl_info[i] = item
        if fade_out > 0:
            ss = N_d - fade_out - 1
            ee = N_d - 1
            for i in range(ss, N_d):
                alpha = max((ee - i) / (ee - ss), 0)
                item = ctrl_info.get(i, {})
                item["fade_alpha"] = alpha
                ctrl_info[i] = item
        self.ctrl_info = ctrl_info

    def setup(self, source_path, output_path, **kwargs):

        # ======== Prepare Options ========
        kwargs = self._merge_kwargs(self.default_kwargs, kwargs)
        print("=" * 20, "setup kwargs", "=" * 20)
        print_cfg(**kwargs)
        print("=" * 50)

        # -- avatar_registrar: template cfg --
        self.max_size = kwargs.get("max_size", 1920)
        self.template_n_frames = kwargs.get("template_n_frames", -1)

        # -- avatar_registrar: crop cfg --
        self.crop_scale = kwargs.get("crop_scale", 2.3)
        self.crop_vx_ratio = kwargs.get("crop_vx_ratio", 0)
        self.crop_vy_ratio = kwargs.get("crop_vy_ratio", -0.125)
        self.crop_flag_do_rot = kwargs.get("crop_flag_do_rot", True)
        
        # -- avatar_registrar: smo for video --
        self.smo_k_s = kwargs.get('smo_k_s', 13)

        # -- condition_handler: ECS --
        self.emo = kwargs.get("emo", 4)    # int | [int] | [[int]] | numpy
        self.eye_f0_mode = kwargs.get("eye_f0_mode", False)    # for video
        self.ch_info = kwargs.get("ch_info", None)    # dict of np.ndarray

        # -- audio2motion: setup --
        self.overlap_v2 = kwargs.get("overlap_v2", 10)
        self.fix_kp_cond = kwargs.get("fix_kp_cond", 0)
        self.fix_kp_cond_dim = kwargs.get("fix_kp_cond_dim", None)  # [ds,de]
        self.sampling_timesteps = kwargs.get("sampling_timesteps", 50)
        self.online_mode = kwargs.get("online_mode", False)
        self.v_min_max_for_clip = kwargs.get('v_min_max_for_clip', None)
        self.smo_k_d = kwargs.get("smo_k_d", 3)

        # -- motion_stitch: setup --
        self.N_d = kwargs.get("N_d", -1)
        self.use_d_keys = kwargs.get("use_d_keys", None)
        self.relative_d = kwargs.get("relative_d", True)
        self.drive_eye = kwargs.get("drive_eye", None)    # None: true4image, false4video
        self.delta_eye_arr = kwargs.get("delta_eye_arr", None)
        self.delta_eye_open_n = kwargs.get("delta_eye_open_n", 0)
        self.fade_type = kwargs.get("fade_type", "")    # "" | "d0" | "s"
        self.fade_out_keys = kwargs.get("fade_out_keys", ("exp",))
        self.flag_stitching = kwargs.get("flag_stitching", True)

        self.ctrl_info = kwargs.get("ctrl_info", dict())
        self.overall_ctrl_info = kwargs.get("overall_ctrl_info", dict())
        """
        ctrl_info: list or dict
            {
                fid: ctrl_kwargs
            }

            ctrl_kwargs (see motion_stitch.py):
                fade_alpha
                fade_out_keys

                delta_pitch
                delta_yaw
                delta_roll
        """

        # only hubert support online mode
        assert self.wav2feat.support_streaming or not self.online_mode

        # ======== Register Avatar ========
        crop_kwargs = {
            "crop_scale": self.crop_scale,
            "crop_vx_ratio": self.crop_vx_ratio,
            "crop_vy_ratio": self.crop_vy_ratio,
            "crop_flag_do_rot": self.crop_flag_do_rot,
        }
        n_frames = self.template_n_frames if self.template_n_frames > 0 else self.N_d
        source_info = self.avatar_registrar(
            source_path, 
            max_dim=self.max_size, 
            n_frames=n_frames, 
            **crop_kwargs,
        )

        if len(source_info["x_s_info_lst"]) > 1 and self.smo_k_s > 1:
            source_info["x_s_info_lst"] = smooth_x_s_info_lst(source_info["x_s_info_lst"], smo_k=self.smo_k_s)

        self.source_info = source_info
        self.source_info_frames = len(source_info["x_s_info_lst"])

        # ======== Setup Condition Handler ========
        self.condition_handler.setup(source_info, self.emo, eye_f0_mode=self.eye_f0_mode, ch_info=self.ch_info)

        # ======== Setup Audio2Motion (LMDM) ========
        x_s_info_0 = self.condition_handler.x_s_info_0
        self.audio2motion.setup(
            x_s_info_0, 
            overlap_v2=self.overlap_v2,
            fix_kp_cond=self.fix_kp_cond,
            fix_kp_cond_dim=self.fix_kp_cond_dim,
            sampling_timesteps=self.sampling_timesteps,
            online_mode=self.online_mode,
            v_min_max_for_clip=self.v_min_max_for_clip,
            smo_k_d=self.smo_k_d,
        )

        # ======== Setup Motion Stitch ========
        is_image_flag = source_info["is_image_flag"]
        x_s_info = source_info['x_s_info_lst'][0]
        self.motion_stitch.setup(
            N_d=self.N_d,
            use_d_keys=self.use_d_keys,
            relative_d=self.relative_d,
            drive_eye=self.drive_eye,
            delta_eye_arr=self.delta_eye_arr,
            delta_eye_open_n=self.delta_eye_open_n,
            fade_out_keys=self.fade_out_keys,
            fade_type=self.fade_type,
            flag_stitching=self.flag_stitching,
            is_image_flag=is_image_flag,
            x_s_info=x_s_info,
            d0=None,
            ch_info=self.ch_info,
            overall_ctrl_info=self.overall_ctrl_info,
        )

        # ======== Video Writer ========
        self.output_path = output_path
        self.tmp_output_path = output_path + ".tmp.mp4"
        self.writer = VideoWriterByImageIO(self.tmp_output_path)
        self.writer_pbar = tqdm(desc="writer")

        # ======== Audio Feat Buffer ========
        if self.online_mode:
            # buffer: seq_frames - valid_clip_len
            self.audio_feat = self.wav2feat.wav2feat(np.zeros((self.overlap_v2 * 640,), dtype=np.float32), sr=16000)
            assert len(self.audio_feat) == self.overlap_v2, f"{len(self.audio_feat)}"
        else:
            self.audio_feat = np.zeros((0, self.wav2feat.feat_dim), dtype=np.float32)
        self.cond_idx_start = 0 - len(self.audio_feat)

        # ======== Setup Worker Threads ========
        # Strictly bound the pipeline depth to prevent massive multi-second latency buildup
        QUEUE_MAX_SIZE = 20
        # self.QUEUE_TIMEOUT = None

        self.worker_exception = None
        self.stop_event = threading.Event()

        self.audio2motion_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.motion_stitch_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.warp_f3d_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.decode_f3d_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.putback_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.writer_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)

        self.thread_list = [
            threading.Thread(target=self.audio2motion_worker),
            threading.Thread(target=self.motion_stitch_worker),
            threading.Thread(target=self.warp_f3d_worker),
            threading.Thread(target=self.decode_f3d_worker),
            threading.Thread(target=self.putback_worker),
            threading.Thread(target=self.writer_worker),
        ]

        for thread in self.thread_list:
            thread.start()

    def _get_ctrl_info(self, fid):
        try:
            if isinstance(self.ctrl_info, dict):
                return self.ctrl_info.get(fid, {})
            elif isinstance(self.ctrl_info, list):
                return self.ctrl_info[fid]
            else:
                return {}
        except Exception as e:
            traceback.print_exc()
            return {}

    def writer_worker(self):
        try:
            self._writer_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _writer_worker(self):
        _frame_count = 0
        _t_session_start = time.perf_counter()
        _fps_window = []   # timestamps of last 30 frames for moving-avg FPS
        _MAX_FPS_WIN = 30

        while not self.stop_event.is_set():
            _t_q_start = time.perf_counter()
            try:
                item = self.writer_queue.get(timeout=1)
            except queue.Empty:
                continue
            _q_wait_ms = (time.perf_counter() - _t_q_start) * 1000.0

            if item is None:
                break
            res_frame_rgb = item

            _t_write_start = time.perf_counter()
            self.writer(res_frame_rgb, fmt="rgb")
            _write_ms = (time.perf_counter() - _t_write_start) * 1000.0

            self.writer_pbar.update()
            _frame_count += 1

            # FPS tracking
            _now = time.perf_counter()
            _fps_window.append(_now)
            if len(_fps_window) > _MAX_FPS_WIN:
                _fps_window.pop(0)
            _fps = (_MAX_FPS_WIN - 1) / (_fps_window[-1] - _fps_window[0]) if len(_fps_window) > 1 else 0.0

            if _frame_count % 25 == 0:
                _elapsed = time.perf_counter() - _t_session_start
                _avg_fps = _frame_count / _elapsed if _elapsed > 0 else 0.0
                print(
                    f"[DITTO:WRITE] frame={_frame_count:>5} | "
                    f"q_wait={_q_wait_ms:>7.2f}ms | write={_write_ms:>6.2f}ms | "
                    f"fps_win={_fps:>5.1f} | fps_avg={_avg_fps:>5.1f}",
                    flush=True,
                )
            else:
                # Frame-level debug log
                import logging as _log
                _log.getLogger(__name__).debug(
                    f"[DITTO:WRITE] frame={_frame_count:>5} | "
                    f"q_wait={_q_wait_ms:>7.2f}ms | write={_write_ms:>6.2f}ms"
                )

    def putback_worker(self):
        try:
            self._putback_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _putback_worker(self):
        _frame_count = 0
        while not self.stop_event.is_set():
            _t_q = time.perf_counter()
            try:
                item = self.putback_queue.get(timeout=1)
            except queue.Empty:
                continue
            _q_wait_ms = (time.perf_counter() - _t_q) * 1000.0

            if item is None:
                self.writer_queue.put(None)
                break
            frame_idx, render_img = item
            frame_rgb = self.source_info["img_rgb_lst"][frame_idx]
            M_c2o = self.source_info["M_c2o_lst"][frame_idx]

            _t_put = time.perf_counter()
            res_frame_rgb = self.putback(frame_rgb, render_img, M_c2o)
            _put_ms = (time.perf_counter() - _t_put) * 1000.0

            self.writer_queue.put(res_frame_rgb)
            _frame_count += 1

            import logging as _log
            _log.getLogger(__name__).debug(
                f"[DITTO:PUT]   frame={_frame_count:>5} | "
                f"q_wait={_q_wait_ms:>7.2f}ms | putback={_put_ms:>6.2f}ms | "
                f"writer_q={self.writer_queue.qsize()}"
            )

    def decode_f3d_worker(self):
        try:
            self._decode_f3d_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _decode_f3d_worker(self):
        # Each GPU-heavy worker gets its own CUDA stream so its kernels can
        # overlap with warp_f3d and audio2motion on separate SMs (H200/SM90a).
        _cuda_stream = _make_cuda_stream()
        _frame_count = 0
        while not self.stop_event.is_set():
            _t_q = time.perf_counter()
            try:
                item = self.decode_f3d_queue.get(timeout=1)
            except queue.Empty:
                continue
            _q_wait_ms = (time.perf_counter() - _t_q) * 1000.0

            if item is None:
                self.putback_queue.put(None)
                break
            frame_idx, f_3d = item

            _t_dec = time.perf_counter()
            with _cuda_stream_ctx(_cuda_stream), _autocast_ctx():
                render_img = self.decode_f3d(f_3d)
            # Ensure stream is done before handing off to putback (CPU thread)
            if _cuda_stream is not None:
                _cuda_stream.synchronize()
            _dec_ms = (time.perf_counter() - _t_dec) * 1000.0

            self.putback_queue.put([frame_idx, render_img])
            _frame_count += 1

            import logging as _log
            _log.getLogger(__name__).debug(
                f"[DITTO:DEC]   frame={_frame_count:>5} | "
                f"q_wait={_q_wait_ms:>7.2f}ms | decode={_dec_ms:>6.2f}ms | "
                f"putback_q={self.putback_queue.qsize()}"
            )

    def warp_f3d_worker(self):
        try:
            self._warp_f3d_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _warp_f3d_worker(self):
        # Dedicated CUDA stream — overlaps with decode_f3d on second SM group.
        _cuda_stream = _make_cuda_stream()
        _frame_count = 0
        while not self.stop_event.is_set():
            _t_q = time.perf_counter()
            try:
                item = self.warp_f3d_queue.get(timeout=1)
            except queue.Empty:
                continue
            _q_wait_ms = (time.perf_counter() - _t_q) * 1000.0

            if item is None:
                self.decode_f3d_queue.put(None)
                break
            frame_idx, x_s, x_d = item
            f_s = self.source_info["f_s_lst"][frame_idx]

            _t_warp = time.perf_counter()
            with _cuda_stream_ctx(_cuda_stream), _autocast_ctx():
                f_3d = self.warp_f3d(f_s, x_s, x_d)
            if _cuda_stream is not None:
                _cuda_stream.synchronize()
            _warp_ms = (time.perf_counter() - _t_warp) * 1000.0

            self.decode_f3d_queue.put([frame_idx, f_3d])
            _frame_count += 1

            import logging as _log
            _log.getLogger(__name__).debug(
                f"[DITTO:WARP]  frame={_frame_count:>5} | "
                f"q_wait={_q_wait_ms:>7.2f}ms | warp={_warp_ms:>6.2f}ms | "
                f"dec_q={self.decode_f3d_queue.qsize()}"
            )

    def motion_stitch_worker(self):
        try:
            self._motion_stitch_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _motion_stitch_worker(self):
        _frame_count = 0
        while not self.stop_event.is_set():
            _t_q = time.perf_counter()
            try:
                item = self.motion_stitch_queue.get(timeout=1)
            except queue.Empty:
                continue
            _q_wait_ms = (time.perf_counter() - _t_q) * 1000.0

            if item is None:
                self.warp_f3d_queue.put(None)
                break

            frame_idx, x_d_info, ctrl_kwargs = item
            x_s_info = self.source_info["x_s_info_lst"][frame_idx]

            _t_stitch = time.perf_counter()
            x_s, x_d = self.motion_stitch(x_s_info, x_d_info, **ctrl_kwargs)
            _stitch_ms = (time.perf_counter() - _t_stitch) * 1000.0

            self.warp_f3d_queue.put([frame_idx, x_s, x_d])
            _frame_count += 1

            import logging as _log
            _log.getLogger(__name__).debug(
                f"[DITTO:STITCH] frame={_frame_count:>5} | "
                f"q_wait={_q_wait_ms:>7.2f}ms | stitch={_stitch_ms:>6.2f}ms | "
                f"warp_q={self.warp_f3d_queue.qsize()}"
            )

    def audio2motion_worker(self):
        try:
            self._audio2motion_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()
        
    def _audio2motion_worker(self):
        is_end = False
        seq_frames = self.audio2motion.seq_frames
        valid_clip_len = self.audio2motion.valid_clip_len
        aud_feat_dim = self.wav2feat.feat_dim
        item_buffer = np.zeros((0, aud_feat_dim), dtype=np.float32)

        res_kp_seq = None
        res_kp_seq_valid_start = None if self.online_mode else 0

        global_idx = 0   # frame idx, for template
        local_idx = 0    # for cur audio_feat
        gen_frame_idx = 0

        # ── [PROFILING] A2M timing state ────────────────────────────
        _a2m_call_count  = 0
        _a2m_total_ms    = 0.0
        _t_session_start = time.perf_counter()

        while not self.stop_event.is_set():
            _t_q = time.perf_counter()
            try:
                item = self.audio2motion_queue.get(timeout=1)    # audio feat
            except queue.Empty:
                continue
            _q_wait_ms = (time.perf_counter() - _t_q) * 1000.0

            if item is None:
                is_end = True
            else:
                item_buffer = np.concatenate([item_buffer, item], 0)

            if not is_end and item_buffer.shape[0] < valid_clip_len:
                # wait at least valid_clip_len new item
                continue
            else:
                self.audio_feat = np.concatenate([self.audio_feat, item_buffer], 0)
                item_buffer = np.zeros((0, aud_feat_dim), dtype=np.float32)

            while True:
                # print("self.audio_feat.shape:", self.audio_feat.shape, "local_idx:", local_idx, "global_idx:", global_idx)
                aud_feat = self.audio_feat[local_idx: local_idx+seq_frames]
                real_valid_len = valid_clip_len
                if len(aud_feat) == 0:
                    break
                elif len(aud_feat) < seq_frames:
                    if not is_end:
                        # wait next chunk
                        break
                    else:
                        # final clip: pad to seq_frames
                        real_valid_len = len(aud_feat)
                        pad = np.stack([aud_feat[-1]] * (seq_frames - len(aud_feat)), 0)
                        aud_feat = np.concatenate([aud_feat, pad], 0)

                aud_cond = self.condition_handler(aud_feat, global_idx + self.cond_idx_start)[None]

                # ── [PROFILING] time the diffusion forward pass ──────────
                _t_a2m = time.perf_counter()
                # audio2motion (diffusion transformer) is the heaviest GPU op.
                # Run it in autocast(bfloat16) for maximum H200 tensor core
                # throughput. The A2M worker does NOT get its own CUDA stream
                # (diffusion is sequential) but benefits from BF16 compute.
                with _autocast_ctx():
                    res_kp_seq = self.audio2motion(aud_cond, res_kp_seq)
                _a2m_ms = (time.perf_counter() - _t_a2m) * 1000.0
                _a2m_call_count  += 1
                _a2m_total_ms    += _a2m_ms
                _a2m_avg_ms       = _a2m_total_ms / _a2m_call_count

                import logging as _log
                _log.getLogger(__name__).info(
                    f"[DITTO:A2M]  call={_a2m_call_count:>4} | "
                    f"infer={_a2m_ms:>8.2f}ms | avg={_a2m_avg_ms:>8.2f}ms | "
                    f"q_wait={_q_wait_ms:>7.2f}ms | "
                    f"feat_buf={len(self.audio_feat):>4} | "
                    f"motion_q={self.motion_stitch_queue.qsize():>3} | "
                    f"mode={'online' if self.online_mode else 'offline'}"
                )

                if res_kp_seq_valid_start is None:
                    # online mode, first chunk
                    res_kp_seq_valid_start = res_kp_seq.shape[1] - self.audio2motion.fuse_length
                    d0 = self.audio2motion.cvt_fmt(res_kp_seq[0:1])[0]
                    self.motion_stitch.d0 = d0

                    local_idx += real_valid_len
                    global_idx += real_valid_len
                    continue
                else:
                    valid_res_kp_seq = res_kp_seq[:, res_kp_seq_valid_start: res_kp_seq_valid_start + real_valid_len]
                    x_d_info_list = self.audio2motion.cvt_fmt(valid_res_kp_seq)

                    for x_d_info in x_d_info_list:
                        frame_idx = _mirror_index(gen_frame_idx, self.source_info_frames)
                        ctrl_kwargs = self._get_ctrl_info(gen_frame_idx)

                        while not self.stop_event.is_set():
                            try:
                                self.motion_stitch_queue.put([frame_idx, x_d_info, ctrl_kwargs], timeout=1)
                                break
                            except queue.Full:
                                continue

                        gen_frame_idx += 1

                    res_kp_seq_valid_start += real_valid_len

                    local_idx += real_valid_len
                    global_idx += real_valid_len

                L = res_kp_seq.shape[1]
                if L > seq_frames * 2:
                    cut_L = L - seq_frames * 2
                    res_kp_seq = res_kp_seq[:, cut_L:]
                    res_kp_seq_valid_start -= cut_L

                if local_idx >= len(self.audio_feat):
                    break

            L = len(self.audio_feat)
            if L > seq_frames * 2:
                cut_L = L - seq_frames * 2
                self.audio_feat = self.audio_feat[cut_L:]
                local_idx -= cut_L

            if is_end:
                break

        # Session summary
        _elapsed = time.perf_counter() - _t_session_start
        import logging as _log
        _log.getLogger(__name__).info(
            f"[DITTO:A2M]  SESSION DONE | calls={_a2m_call_count} | "
            f"avg_infer={(_a2m_total_ms/_a2m_call_count if _a2m_call_count else 0):.1f}ms | "
            f"frames_generated={gen_frame_idx} | elapsed={_elapsed:.1f}s"
        )
        self.motion_stitch_queue.put(None)

    def close(self):
        # flush frames
        self.audio2motion_queue.put(None)
        # Wait for worker threads to finish
        for thread in self.thread_list:
            thread.join()

        try:
            self.writer.close()
            self.writer_pbar.close()
        except:
            traceback.print_exc()

        # Check if any worker encountered an exception
        if self.worker_exception is not None:
            raise self.worker_exception
        
    def run_chunk(self, audio_chunk, chunksize=(3, 5, 2)):
        # only for hubert
        aud_feat = self.wav2feat(audio_chunk, chunksize=chunksize)
        while not self.stop_event.is_set():
            try:
                self.audio2motion_queue.put(aud_feat, timeout=1)
                break
            except queue.Full:
                continue



