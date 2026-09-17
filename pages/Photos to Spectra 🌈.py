from PIL import Image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import hsv_to_rgb
from pillow_heif import register_heif_opener
import streamlit as st
from io import BytesIO
from scipy.ndimage import gaussian_filter1d

st.set_page_config(
    page_title="Photos to Spectra", page_icon="🌈", layout="centered"
)

st.markdown(
    "<h2 style='text-align: center'>Photos to Spectra 🌈</h2>",
    unsafe_allow_html=True,
)
register_heif_opener()


def hue_to_wavelength(hue):
    hue = np.asarray(hue, dtype=float) % 360
    wavelength = np.full(hue.shape, np.nan)

    mask = (hue >= 0) & (hue < 60)
    wavelength[mask] = 700 - (hue[mask] / 60) * (700 - 580)

    mask = (hue >= 60) & (hue < 120)
    wavelength[mask] = 580 - ((hue[mask] - 60) / 60) * (580 - 530)

    mask = (hue >= 120) & (hue < 180)
    wavelength[mask] = 530 - ((hue[mask] - 120) / 60) * (530 - 500)

    mask = (hue >= 180) & (hue < 240)
    wavelength[mask] = 500 - ((hue[mask] - 180) / 60) * (500 - 460)

    mask = (hue >= 240) & (hue < 300)
    wavelength[mask] = 460 - ((hue[mask] - 240) / 60) * (460 - 400)

    mask = (hue >= 300) & (hue < 360)
    wavelength[mask] = 400 + ((hue[mask] - 300) / 60) * (700 - 400)

    return wavelength


def image_to_spectrum(
    image_path,
    wavelengths=np.linspace(390, 710, 300),
    max_side=1500,
    fwhm_nm=4.0,
):
    """
    Convert an image to a wavelength spectrum.

    Parameters
    ----------
    image_path : str
        Path to image (.jpg, .png, .heic, etc.)
    wavelengths : np.ndarray
        1D array of wavelength bins (nm), assumed sorted/evenly spaced.
    max_side : int
        Maximum length of the largest image dimension to downscale for speed.
    fwhm_nm : float
        Full-width-half-max of the instrumental blur, in nm.
        Larger = blurrier/lower resolution.
    """

    im = Image.open(image_path).convert("RGB").convert("HSV")

    # Downscale large images while preserving aspect ratio
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side))

    arr = np.array(im, dtype=np.float32)
    H, S, V = arr[..., 0], arr[..., 1], arr[..., 2]

    # Convert to conventional HSV units
    hue = 360.0 * H / 255.0
    saturation = 100.0 * S / 255.0
    value = 100.0 * V / 255.0

    # Flatten pixels
    hue_flat = hue.ravel()
    sat_flat = saturation.ravel()
    val_flat = value.ravel()

    wavelengths_tmp = hue_to_wavelength(hue_flat)

    # Keep only valid pixels
    mask = ~np.isnan(wavelengths_tmp)
    wavelengths_tmp = wavelengths_tmp[mask]
    sat_flat = sat_flat[mask]
    val_flat = val_flat[mask]

    nbins = len(wavelengths)

  
    bin_edges = (wavelengths[:-1] + wavelengths[1:]) / 2.0
    indices = np.searchsorted(bin_edges, wavelengths_tmp)

    # This reduces to a per-pixel "color" weight that lands in exactly one bin,
    # plus a per-pixel "white" weight that is identical across every bin.
    denom = 300.0 - 2.0 * sat_flat  # always in [100, 300], sat in [0,100] -> no div-by-zero
    weight_color = val_flat * sat_flat / (100.0 * denom)
    weight_white = val_flat * (100.0 - sat_flat) * 3.0 / (nbins * 100.0 * denom)

    intensities = np.bincount(indices, weights=weight_color, minlength=nbins).astype(float)
    intensities += weight_white.sum()  # uniform white contribution, same value added to every bin

    dλ = wavelengths[1] - wavelengths[0] 
    sigma_nm = fwhm_nm / 2.3548
    sigma_bins = sigma_nm / dλ
    intensities = gaussian_filter1d(intensities, sigma=sigma_bins, mode="nearest")
    intensities = intensities / np.max(intensities)

    return wavelengths, intensities

st.markdown(
    "code to generate spectra is adapted from [this repository](https://github.com/nialldeacon/espectrally_for_you)"
)

uploaded_file = st.file_uploader(
        "upload a photo", type=["jpg", "png", "heic"]
    )



darkmode = st.checkbox("plot spectrum in dark mode", value=False)



if uploaded_file is None:
    st.stop() 

image = Image.open(uploaded_file)
st.image(image, caption=uploaded_file.name, width='stretch')

uploaded_file.seek(0)   # reset before image_to_spectrum reads it again


font_path = "static/GoogleSans-Regular.ttf"
mpl.font_manager.fontManager.addfont(font_path)
font_prop = mpl.font_manager.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()

mpl.rcParams.update(
    {
        "figure.dpi": 200,
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "savefig.dpi": 300,
        "savefig.format": "png",
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "figure.autolayout": True,
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.2,
        "axes.labelcolor": "black",
        "axes.labelsize": 28,
        "axes.titlesize": 28,
        "axes.titlecolor": "black",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": True,
        "grid.color": "black",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.8,
        "xtick.top": True,
        "ytick.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.minor.width": 1,
        "ytick.minor.width": 1,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "xtick.minor.ndivs": 5,
        "ytick.minor.ndivs": 5,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "lines.color": "black",
        "mathtext.default": "regular",
        "legend.frameon": True,
        "legend.fontsize": 12,
        "legend.handlelength": 2,
        "legend.labelcolor": "black",
        "legend.facecolor": "white",
        "legend.edgecolor": "black",
        "legend.fancybox": True,
        "legend.framealpha": 1.0,
    }
)

if darkmode:
    mpl.rcParams.update(
        {
            "figure.facecolor": "black",
            "figure.edgecolor": "black",
            "savefig.facecolor": "black",
            "savefig.edgecolor": "black",
            "axes.facecolor": "black",
            "axes.edgecolor": "white",
            "axes.labelcolor": "white",
            "axes.titlecolor": "white",
            "grid.color": "snow",
            "grid.linewidth": 0.4,
            "grid.alpha": 0.8,
            "xtick.color": "white",
            "ytick.color": "white",
            "xtick.labelcolor": "white",
            "ytick.labelcolor": "white",
            "lines.color": "white",
            "mathtext.default": "regular",
            "legend.labelcolor": "white",
            "legend.facecolor": "black",
            "legend.edgecolor": "white",
        }
    )


wavelengths, intensities = image_to_spectrum(uploaded_file)

fig, ax = plt.subplots(figsize=(20, 8))
c='k'
if darkmode:
    c='w'
ax.plot(wavelengths, intensities, lw=5, c=c)

ax.set_xlabel(r"Wavelength (nm)")
ax.set_ylabel(r"Normalized Intensity")
ax.set_xlim(390, 710)
ax.set_ylim(-0.04, 1.04)
ax.grid(True, which='both')

# Leave room at the top of the figure for the colorbar band
fig.subplots_adjust(top=0.88)

# Force a draw so get_position() reflects the actual final layout
fig.canvas.draw()

pos = ax.get_position()
cax = fig.add_axes([pos.x0, pos.y1 + 0.02, pos.width, pos.height * 0.05])

norm = mpl.colors.Normalize(vmin=wavelengths.min(), vmax=wavelengths.max())
sm = mpl.cm.ScalarMappable(cmap="turbo", norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
cbar.set_ticks([])
cbar.ax.set_xticklabels([])

st.pyplot(fig)

buf = BytesIO()
fig.savefig(buf, format="png")
buf.seek(0)

st.download_button(
    label="download spectrum",
    data=buf,
    file_name=f"spec.png",
    mime="image/png",
)

