import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import tempfile
import subprocess
import io

import imageio_ffmpeg
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────
EMOTIONS = ["angry", "happy", "neutral", "sad"]
EMOTION_COLORS = {
    "angry":   "#FF4B4B",
    "happy":   "#FFD700",
    "neutral": "#4ECDC4",
    "sad":     "#5B9BD5",
}
EMOTION_EMOJI = {"angry": "😠", "happy": "😊", "neutral": "😐", "sad": "😢"}

MAX_FRAMES = 10
SR         = 16000
N_MELS     = 128
DURATION   = 3.0
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(__file__).parent / "best_model.pt"

VID_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Model Architecture (exact mirror of training code) ─────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


class AudioEncoder(nn.Module):
    def __init__(self, embed_dim=128, max_seq_len=50):
        super().__init__()
        self.blocks    = nn.Sequential(
            ConvBlock(1, 16), ConvBlock(16, 32), ConvBlock(32, 64)
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.proj      = nn.Linear(64, embed_dim)
        self.norm      = nn.LayerNorm(embed_dim)
        self.pos_emb   = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, x):
        x = self.blocks(x)
        x = self.freq_pool(x).squeeze(2).permute(0, 2, 1)
        x = self.proj(x)
        x = x + self.pos_emb(torch.arange(x.shape[1], device=x.device))
        return self.norm(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class VideoFrameCNN(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage1 = ResidualBlock(16, 32, stride=1)
        self.stage2 = ResidualBlock(32, 64, stride=2)
        self.stage3 = ResidualBlock(64, 128, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d((1, 1))
        self.proj   = nn.Linear(128, embed_dim)
        self.norm   = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.norm(self.proj(self.pool(x).flatten(1)))


class VideoEncoder(nn.Module):
    def __init__(self, embed_dim=128, max_frames_ceil=64):
        super().__init__()
        self.frame_cnn = VideoFrameCNN(embed_dim)
        self.pos_emb   = nn.Embedding(max_frames_ceil, embed_dim)
        self.norm      = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, T, C, H, W = x.shape
        feat = self.frame_cnn(x.view(B * T, C, H, W)).view(B, T, -1)
        feat = feat + self.pos_emb(torch.arange(T, device=x.device))
        return self.norm(feat)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=128, nhead=4, ff_dim=512, dropout=0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, nhead,
                                           dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        return x + self.ff(self.norm2(x))


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=128, nhead=4, ff_dim=512,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, nhead, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class BidirectionalCrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim=128, nhead=4, ff_dim=512, dropout=0.3):
        super().__init__()
        self.norm_a  = nn.LayerNorm(embed_dim)
        self.norm_v  = nn.LayerNorm(embed_dim)
        self.attn_av = nn.MultiheadAttention(embed_dim, nhead,
                                             dropout=dropout, batch_first=True)
        self.attn_va = nn.MultiheadAttention(embed_dim, nhead,
                                             dropout=dropout, batch_first=True)
        self.norm_ff_a = nn.LayerNorm(embed_dim)
        self.norm_ff_v = nn.LayerNorm(embed_dim)
        self.ff_a = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout),
        )
        self.ff_v = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, audio, video):
        na, nv = self.norm_a(audio), self.norm_v(video)
        a_out, w_av = self.attn_av(query=na, key=nv, value=nv)
        v_out, _    = self.attn_va(query=nv, key=na, value=na)
        audio = audio + a_out
        video = video + v_out
        audio = audio + self.ff_a(self.norm_ff_a(audio))
        video = video + self.ff_v(self.norm_ff_v(video))
        return audio, video, w_av


class FusionHead(nn.Module):
    def __init__(self, embed_dim=128, nhead=4, ff_dim=512,
                 num_classes=4, dropout=0.3):
        super().__init__()
        self.cross_attn = BidirectionalCrossAttentionBlock(
            embed_dim, nhead, ff_dim, dropout
        )
        self.norm_a = nn.LayerNorm(embed_dim)
        self.norm_v = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 64), nn.LayerNorm(64),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, audio, video):
        audio_f, video_f, attn_w = self.cross_attn(audio, video)
        pooled = torch.cat(
            [self.norm_a(audio_f.mean(1)), self.norm_v(video_f.mean(1))], dim=-1
        )
        return self.classifier(pooled), attn_w


class EmotionModel(nn.Module):
    def __init__(self, embed_dim=128, nhead=4, ff_dim=512,
                 num_layers=2, num_classes=4, dropout=0.3):
        super().__init__()
        self.audio_encoder     = AudioEncoder(embed_dim)
        self.video_encoder     = VideoEncoder(embed_dim)
        self.audio_transformer = TransformerEncoder(embed_dim, nhead, ff_dim, num_layers, dropout)
        self.video_transformer = TransformerEncoder(embed_dim, nhead, ff_dim, num_layers, dropout)
        self.fusion            = FusionHead(embed_dim, nhead, ff_dim, num_classes, dropout)

    def forward(self, audio, video):
        a = self.audio_transformer(self.audio_encoder(audio))
        v = self.video_transformer(self.video_encoder(video))
        return self.fusion(a, v)


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    m    = EmotionModel(embed_dim=128, nhead=4, ff_dim=512,
                        num_layers=2, num_classes=4, dropout=0.3)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt["model"])
    m.to(DEVICE).eval()
    return m, ckpt


# ── Processing helpers ─────────────────────────────────────────────────────────
def extract_audio(video_path: str, out_wav: str) -> bool:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(SR), "-ac", "1", out_wav,
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and os.path.exists(out_wav)


def mel_tensor_from_file(audio_path: str):
    """Return (audio_tensor (1,1,128,T), mel_db_numpy (128,T))."""
    y, _ = librosa.load(audio_path, sr=SR, duration=DURATION)
    target = int(SR * DURATION)
    y = np.pad(y, (0, max(0, target - len(y))))[:target]
    mel    = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor, mel_db


def load_video_frames(video_path: str):
    """Return (bgr_list, tensor_list, fps)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    bgr_frames, tensors = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bgr_frames.append(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensors.append(VID_TRANSFORM(rgb))
    cap.release()
    return bgr_frames, tensors, fps


def sample_uniformly(bgr_frames, tensors, n=MAX_FRAMES):
    """Uniformly sample n frames (with repetition if video is short)."""
    total = len(bgr_frames)
    indices = np.linspace(0, total - 1, n, dtype=int)
    return (
        [bgr_frames[i] for i in indices],
        torch.stack([tensors[i] for i in indices]),
    )


# ── GradCAM ────────────────────────────────────────────────────────────────────
def _run_gradcam(model, target_layer, audio, video, target_class=None):
    """
    Generic GradCAM runner. Hooks onto target_layer, runs one forward+backward,
    returns (cam_np, pred_idx, probs_np).
    cam_np shape: (T, h, w) for video layers, (H, W) for audio layers.
    """
    model.eval()
    model.zero_grad()
    activations, gradients = {}, {}

    def fwd_hook(m, i, o):
        activations["v"] = o.detach()

    def bwd_hook(m, gi, go):
        gradients["v"] = go[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    logits, _ = model(audio, video)
    pred  = int(logits.argmax(1).item()) if target_class is None else target_class
    probs = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()

    one_hot = torch.zeros_like(logits)
    one_hot[0, pred] = 1.0
    logits.backward(gradient=one_hot)
    h1.remove()
    h2.remove()

    act  = activations["v"]
    grad = gradients["v"]
    return act, grad, pred, probs


def video_gradcam(model, audio, video_tensor):
    """Returns per-frame cams (T, 112, 112), pred_idx, probs."""
    target_layer = model.video_encoder.frame_cnn.stage3
    act, grad, pred, probs = _run_gradcam(model, target_layer, audio, video_tensor)

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cams = F.relu((weights * act).sum(dim=1))          # (T, h, w)
    cams = F.interpolate(
        cams.unsqueeze(1), size=(112, 112), mode="bilinear", align_corners=False
    ).squeeze(1)                                        # (T, 112, 112)

    cams_np = cams.cpu().numpy()
    for i in range(cams_np.shape[0]):
        c = cams_np[i]
        cams_np[i] = (c - c.min()) / (c.max() - c.min() + 1e-8)
    return cams_np, pred, probs


def audio_gradcam(model, audio, video_tensor):
    """Returns audio cam (128, T_mel) normalised to [0,1]."""
    target_layer = model.audio_encoder.blocks[2]
    act, grad, _, _ = _run_gradcam(model, target_layer, audio, video_tensor)

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))      # (1, 1, H, W)
    cam = F.interpolate(cam, size=audio.shape[2:],
                        mode="bilinear", align_corners=False)    # (1, 1, 128, T)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


# ── Visualisation helpers ──────────────────────────────────────────────────────
_BG = "#0E1117"
_PANEL = "#1A202C"


def overlay_cam(bgr_frame, cam_112):
    """Blend JET heatmap over frame; returns RGB numpy array."""
    h, w = bgr_frame.shape[:2]
    cam_r = cv2.resize(cam_112, (w, h))
    heat  = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out   = cv2.addWeighted(bgr_frame, 0.55, heat, 0.45, 0)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def figure_frame_grid(bgr_frames, cams, n_cols=5):
    """
    Two-row-pair grid: top = original, bottom = GradCAM overlay.
    Layout (for 10 frames, n_cols=5): 4 rows × 5 cols.
    """
    n      = len(bgr_frames)
    n_rows = (n + n_cols - 1) // n_cols   # how many 'groups' of columns
    fig, axes = plt.subplots(
        n_rows * 2, n_cols,
        figsize=(n_cols * 2.4, n_rows * 2 * 2.0),
        facecolor=_BG,
    )
    axes = np.array(axes).reshape(n_rows * 2, n_cols)

    for idx in range(n):
        group_row = idx // n_cols          # 0 or 1 for 10-frame / 5-col
        col       = idx % n_cols
        orig_row  = group_row * 2
        cam_row   = group_row * 2 + 1

        rgb_orig = cv2.cvtColor(bgr_frames[idx], cv2.COLOR_BGR2RGB)
        rgb_cam  = overlay_cam(bgr_frames[idx], cams[idx])

        for ax, img, title in [
            (axes[orig_row, col], rgb_orig, f"Frame {idx + 1}"),
            (axes[cam_row,  col], rgb_cam,  "GradCAM"),
        ]:
            ax.imshow(img)
            ax.set_title(title, color="white", fontsize=7, pad=2)
            ax.axis("off")

    # Hide any unused axes
    for idx in range(n, n_rows * n_cols):
        g, c = idx // n_cols, idx % n_cols
        axes[g * 2,     c].axis("off")
        axes[g * 2 + 1, c].axis("off")

    # Row labels on the left
    for g in range(n_rows):
        axes[g * 2,     0].set_ylabel("Original", color="#A0AEC0",
                                      fontsize=8, labelpad=4)
        axes[g * 2 + 1, 0].set_ylabel("GradCAM",  color="#A0AEC0",
                                       fontsize=8, labelpad=4)
        axes[g * 2,     0].yaxis.label.set_visible(True)
        axes[g * 2 + 1, 0].yaxis.label.set_visible(True)

    plt.tight_layout(pad=0.4)
    return fig


def figure_audio_gradcam(mel_db, cam):
    """Side-by-side mel spectrogram and GradCAM overlay."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.2), facecolor=_BG)

    for ax in (ax1, ax2):
        ax.set_facecolor(_PANEL)
        ax.tick_params(colors="#A0AEC0", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2D3748")

    ax1.imshow(mel_db, origin="lower", aspect="auto", cmap="magma")
    ax1.set_title("Mel Spectrogram", color="white", fontsize=10, pad=6)
    ax1.set_xlabel("Time frames", color="#A0AEC0", fontsize=8)
    ax1.set_ylabel("Mel bins", color="#A0AEC0", fontsize=8)

    ax2.imshow(mel_db, origin="lower", aspect="auto", cmap="magma", alpha=0.55)
    im = ax2.imshow(cam, origin="lower", aspect="auto", cmap="jet", alpha=0.6)
    ax2.set_title("Audio GradCAM", color="white", fontsize=10, pad=6)
    ax2.set_xlabel("Time frames", color="#A0AEC0", fontsize=8)

    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#A0AEC0", labelsize=7)

    plt.tight_layout(pad=0.5)
    return fig


def figure_confidence_bars(probs):
    """Horizontal bar chart for per-class confidence."""
    fig, ax = plt.subplots(figsize=(6, 2.6), facecolor=_PANEL)
    ax.set_facecolor(_PANEL)

    colors = [EMOTION_COLORS[e] for e in EMOTIONS]
    labels = [f"{EMOTION_EMOJI[e]}  {e.title()}" for e in EMOTIONS]
    bars   = ax.barh(labels, probs * 100, color=colors, height=0.5, edgecolor="none")

    for bar, p in zip(bars, probs):
        ax.text(
            bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{p:.1%}",
            va="center", color="white", fontsize=9, fontweight="bold",
        )

    ax.set_xlim(0, 118)
    ax.set_xlabel("Confidence (%)", color="#A0AEC0", fontsize=8)
    ax.tick_params(colors="white", labelsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.xaxis.label.set_color("#A0AEC0")
    ax.invert_yaxis()

    plt.tight_layout(pad=0.4)
    return fig


# ── Page layout ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal Emotion Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0E1117; }
    .section-hdr {
        font-size: 0.8rem; font-weight: 700; letter-spacing: 2.5px;
        text-transform: uppercase; color: #718096;
        border-bottom: 1px solid #2D3748; padding-bottom: 4px;
        margin: 20px 0 10px 0;
    }
    .emotion-pill {
        display: inline-block; border-radius: 40px;
        padding: 10px 30px; font-size: 2rem; font-weight: 800;
        letter-spacing: 4px; text-transform: uppercase; margin: 6px 0;
    }
    .info-box {
        background: #1A202C; border-radius: 10px;
        padding: 14px 18px; margin-bottom: 10px;
        font-size: 0.88rem; color: #CBD5E0;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    model, ckpt = load_model()

    st.markdown("## About")
    st.markdown(
        "This demo detects emotion from short video clips by jointly analysing "
        "the speaker's **facial expressions** and **speech audio**. "
        "Gradient-weighted Class Activation Maps (GradCAM) highlight which "
        "visual regions and audio frequencies drove the prediction."
    )

    st.markdown("---")
    st.markdown("**Detectable emotions**")
    for e in EMOTIONS:
        st.markdown(
            f'<span style="color:{EMOTION_COLORS[e]};font-weight:600;">'
            f'{EMOTION_EMOJI[e]}&nbsp; {e.title()}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    col_a, col_b = st.columns(2)
    col_a.metric("Val accuracy", f"{ckpt['val_acc']:.1%}")
    col_b.metric("Device", str(DEVICE).upper())

    st.markdown("---")
    st.caption("DATA 255 · Deep Learning · SJSU Spring 2026")

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🎭 Multimodal Emotion Detection")
st.markdown(
    "Upload a short video clip. The model fuses **audio** (mel spectrogram) "
    "and **video** (facial frames) to predict one of four emotions, then "
    "explains its decision with **GradCAM** heatmaps on both modalities."
)

# ── Upload ─────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop a video file here",
    type=["mp4", "avi", "mov", "mkv"],
    help="Best results: 3–10 s clip with a visible face and speech audio.",
)

if uploaded is None:
    st.markdown(
        '<div class="info-box" style="text-align:center; padding:30px;">'
        "⬆️ &nbsp; Upload a video above to get started"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ── Show uploaded video ────────────────────────────────────────────────────────
video_bytes = uploaded.read()

col_vid, col_spacer = st.columns([1, 2])
with col_vid:
    st.markdown('<p class="section-hdr">Uploaded video</p>', unsafe_allow_html=True)
    st.video(io.BytesIO(video_bytes))

st.markdown("---")

# ── Run analysis ───────────────────────────────────────────────────────────────
with tempfile.TemporaryDirectory() as tmpdir:
    vid_path   = os.path.join(tmpdir, "input.mp4")
    audio_path = os.path.join(tmpdir, "audio.wav")

    with open(vid_path, "wb") as f:
        f.write(video_bytes)

    # --- Load video frames ---
    with st.spinner("Reading video frames…"):
        bgr_all, tens_all, fps = load_video_frames(vid_path)

    if len(bgr_all) == 0:
        st.error("Could not read any frames from the video. Try a different file.")
        st.stop()

    sampled_bgr, vid_stack = sample_uniformly(bgr_all, tens_all, MAX_FRAMES)
    video_tensor = vid_stack.unsqueeze(0).to(DEVICE)  # (1, T, 3, 112, 112)

    st.caption(
        f"Video: {len(bgr_all)} frames @ {fps:.1f} fps — "
        f"using {MAX_FRAMES} uniformly sampled frames for inference."
    )

    # --- Extract audio ---
    with st.spinner("Extracting audio…"):
        has_audio = extract_audio(vid_path, audio_path)

    if has_audio:
        audio_tensor, mel_db = mel_tensor_from_file(audio_path)
        audio_tensor = audio_tensor.to(DEVICE)
    else:
        st.warning(
            "No audio track detected — using a silent placeholder. "
            "Predictions rely on video only."
        )
        mel_db       = np.zeros((N_MELS, 94))
        audio_tensor = torch.zeros(1, 1, N_MELS, 94, device=DEVICE)

    # --- GradCAM inference ---
    with st.spinner("Running inference and computing GradCAM…"):
        cam_frames, pred_idx, probs = video_gradcam(model, audio_tensor, video_tensor)
        a_cam = audio_gradcam(model, audio_tensor, video_tensor)

# ── Results ────────────────────────────────────────────────────────────────────
emotion       = EMOTIONS[pred_idx]
badge_color   = EMOTION_COLORS[emotion]
confidence    = float(probs[pred_idx])

# ─ Prediction row ─
st.markdown('<p class="section-hdr">Prediction</p>', unsafe_allow_html=True)
col_badge, col_chart = st.columns([1, 2], gap="large")

with col_badge:
    st.markdown(
        f'<div class="emotion-pill" style="'
        f'background:{badge_color}22; color:{badge_color}; '
        f'border:2px solid {badge_color};">'
        f'{EMOTION_EMOJI[emotion]}&nbsp; {emotion.upper()}'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**Confidence:** `{confidence:.1%}`")
    if confidence < 0.5:
        st.warning("Low confidence — model is uncertain.")

with col_chart:
    fig_bars = figure_confidence_bars(probs)
    st.pyplot(fig_bars, use_container_width=True)
    plt.close(fig_bars)

st.markdown("---")

# ─ Video GradCAM ─
st.markdown(
    '<p class="section-hdr">Video Explainability using GradCAM</p>',
    unsafe_allow_html=True,
)
st.caption(
    "Warmer colours (red / yellow) highlight face regions that most influenced "
    "the prediction. Cooler colours (blue) are less important."
)

fig_grid = figure_frame_grid(sampled_bgr, cam_frames, n_cols=5)
st.pyplot(fig_grid, use_container_width=True)
plt.close(fig_grid)

st.markdown("---")

# ─ Audio GradCAM ─
st.markdown(
    '<p class="section-hdr">Audio Explainability — GradCAM on audio_encoder.blocks[2]</p>',
    unsafe_allow_html=True,
)
st.caption(
    "Left: raw mel spectrogram. Right: GradCAM overlay highlighting the "
    "time–frequency regions that drove the audio stream's contribution."
)

fig_audio = figure_audio_gradcam(mel_db, a_cam)
st.pyplot(fig_audio, use_container_width=True)
plt.close(fig_audio)

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#4A5568; font-size:0.8rem;'>"
    "DATA 255 Deep Learning — Multimodal Emotion Detection &nbsp;|&nbsp; "
    "Model trained on CREMA-D &nbsp;|&nbsp; "
    "Audio + Video Fusion with Bidirectional Cross-Attention"
    "</div>",
    unsafe_allow_html=True,
)
