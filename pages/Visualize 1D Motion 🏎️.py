import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib as mpl
import numpy as np
import streamlit as st
from io import BytesIO
import pandas as pd
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor



st.set_page_config(page_title="Visualize 1D Motion", page_icon="🏎️", layout="centered")


def ensure_array(arr, t):
    """
    Ensure arr is a NumPy array with the same length as t.
    If arr is a scalar (int/float), fill an array with that value.
    """
    if np.isscalar(arr):
        return np.full_like(t, arr, dtype=float)
    return np.array(arr, dtype=float)


def makeplot(x_arr, v_arr, a_arr, t, darkmode):
    tval = st.slider(
        "",
        min_value=0.0,
        max_value=float(max(t)),
        value=3.142,
        step=0.001,
        format="t = %.3f s",
        help="move the slider to explore x(t), v(t), and a(t) at different times.",
    )
    idx = np.abs(t - tval).argmin()
    x = x_arr[idx]
    v = v_arr[idx]
    a = a_arr[idx]

    t = t[10:-10]
    arrs = [arr[10:-10] for arr in [x_arr, v_arr, a_arr]]

    c1 = "royalblue"
    c2 = "tab:red"
    c3 = "seagreen"
    czero = "k"
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
        c1 = "gold"
        c2 = "lightcyan"
        c3 = "palegreen"
        czero = "gainsboro"

    fig, axes = plt.subplots(
        3, 1, figsize=(8, 8), sharex=True, sharey=False, constrained_layout=True
    )
    axes = axes.flatten()
    axes[2].set_xlabel("time [s]", fontsize=14)
    ylabels = [
        "position [m]",
        "velocity [m/s]",
        r"acceleration [m/s$^2$]",
    ]
    cs = [c1, c2, c3]
    vals = [x, v, a]
    def fmt(val, decimals=3):
        s = f"{val:.{decimals}f}"
        if s.startswith("-") and float(s) == 0:
            s = s[1:]
        return s

    labels = [
        rf"x({fmt(tval)} s) = {fmt(x)} m",
        rf"v({fmt(tval)} s) = {fmt(v)} m/s",
        rf"a({fmt(tval)} s) = {fmt(a)} m/s$^2$",
    ]
    for i, (ax, ylabel, arr, c, label) in enumerate(
        zip(axes, ylabels, arrs, cs, labels)
    ):
        ax.plot(t, arr, lw=2, color=c, zorder=100)
        ax.set_ylabel(ylabel, fontsize=14)
        # ax.grid(False, which="both")
        ax.axvline(
            tval,
            color=czero,
            alpha=0.8,
            lw=1,
            linestyle="--",
            zorder=1,
            label=label,
        )
        ax.legend(
            loc="best", fontsize=10, frameon=True, handlelength=0, handletextpad=0
        )
        ymin, ymax = ax.get_ylim()
        if ymin < 0 < ymax:
            ax.axhline(
                0,
                color=czero,
                lw=2,  
                alpha=0.8,
                zorder=3, 
                linestyle="--",
            )
    return fig


def parse_function(func_str):
    """
    Parse a single function string into a sympy expression and a numpy-callable function.

    Returns:
        f: callable, f(t)
        expr: sympy expression
        error: None if successful, else error string
    """
    t_sym = sp.symbols("t")
    transformations = standard_transformations + (convert_xor,)

    func_str = func_str.strip()
    if not func_str:
        return None, None, "Empty input string"

    try:
        expr = parse_expr(
            func_str,
            transformations=transformations,
            local_dict={"t": t_sym, "e": sp.E},
        )
        f = sp.lambdify(t_sym, expr, modules=["numpy"])
        return f, expr, None
    except Exception as e:
        return None, None, f"{func_str} → {e}"


st.markdown(
    "<h1 style='text-align: center'>Visualize 1D Motion 🏎️</h1>", unsafe_allow_html=True
)


delta_t = st.number_input("Δt (s)", value=12.5663, format="%0.3f", step=0.001)


t = np.linspace(0, delta_t, int(1e6))


usr_func = st.radio(
    "specify function to define", ("a(t)", "v(t)", "x(t)"), horizontal=True
)
func_str = st.text_area(
    "",
    "sin(t)",
    help="use t as the variable, e.g. 2\*t+4, 5\*t^2, cos(t)",
    height=50,
)


x0 = 0
v0 = 0

t_sym = sp.symbols("t")

if usr_func == "x(t)":

    ufunc, expr, err = parse_function(func_str)

    v_expr = sp.diff(expr, t_sym)
    a_expr = sp.diff(v_expr, t_sym)

    x_func = sp.lambdify(t_sym, expr, modules="numpy")
    v_func = sp.lambdify(t_sym, v_expr, modules="numpy")
    a_func = sp.lambdify(t_sym, a_expr, modules="numpy")


elif usr_func == "v(t)":

    x0 = st.number_input("x₀ (m)", value=0.0, step=0.001, format="%0.3f")

    ufunc, expr, err = parse_function(func_str)

    tau = sp.symbols("tau")  # dummy integration variable to avoid clashing with t
    x_expr = sp.integrate(expr.subs(t_sym, tau), (tau, 0, t_sym)) + x0
    a_expr = sp.diff(expr, t_sym)

    x_func = sp.lambdify(t_sym, x_expr, modules="numpy")
    v_func = sp.lambdify(t_sym, expr, modules="numpy")
    a_func = sp.lambdify(t_sym, a_expr, modules="numpy")


elif usr_func == "a(t)":

    x0 = st.number_input("x₀ (m)", value=0.0, step=0.001, format="%0.3f")
    v0 = st.number_input("v₀ (m/s)", value=-1.0, step=0.001, format="%0.3f")
    ufunc, expr, err = parse_function(func_str)

    tau = sp.symbols("tau")  # dummy integration variable to avoid clashing with t
    v_expr = sp.integrate(expr.subs(t_sym, tau), (tau, 0, t_sym)) + v0
    x_expr = sp.integrate(v_expr.subs(t_sym, tau), (tau, 0, t_sym)) + x0

    x_func = sp.lambdify(t_sym, x_expr, modules="numpy")
    v_func = sp.lambdify(t_sym, v_expr, modules="numpy")
    a_func = sp.lambdify(t_sym, expr, modules="numpy")


x_arr = ensure_array(x_func(t), t)
v_arr = ensure_array(v_func(t), t)
a_arr = ensure_array(a_func(t), t)


darkmode = st.checkbox("dark mode", value=False)

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
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "axes.titlecolor": "black",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": False,
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
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
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


fig = makeplot(x_arr, v_arr, a_arr, t, darkmode)

st.pyplot(fig)

buf = BytesIO()
fig.savefig(buf, format="png")
buf.seek(0)

st.download_button(
    label="download graph", data=buf, file_name="pva.png", mime="image/png"
)


# FIGURE OUT HOW TO MAKE BELOW FASTER

# makeani = st.checkbox(
#     "make animation",
#     value=False,
#     help="visualize the motion in an animation. will take a while to generate.",
# )

# if makeani:
#     orient = st.radio(
#         "direction of motion:", ("vertical", "horizontal"), horizontal=True
#     )

#     frames = 100
#     ani_t_arr = np.linspace(0, delta_t, frames)
#     ani_x_arr = ensure_array(x_func(ani_t_arr), ani_t_arr)  # vertical position

#     if orient == "vertical":
#         fig_, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
#         color = "gold" if darkmode else "royalblue"
#         ecolor = "gainsboro" if darkmode else "k"
#         (ball,) = ax.plot(
#             [0],
#             [ani_x_arr[0]],
#             "o",
#             markersize=30,
#             color=color,
#             zorder=100,
#             markeredgecolor=ecolor,
#             markeredgewidth=3,
#         )

#         ax.set_xlim(-1, 1)
#         ax.set_ylim(np.min(ani_x_arr), np.max(ani_x_arr))

#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)
#         ax.spines["bottom"].set_visible(False)
#         ax.tick_params(axis="y", right=False, which="both")  # hide right y-axis ticks
#         ax.xaxis.set_visible(False)  # hide x-axis entirely
#         ax.set_ylabel("Position (m)", fontsize=10)
#         color = "gainsboro" if darkmode else "k"
#         ax.grid(
#             True,
#             which="both",
#             axis="y",
#             color=color,
#             alpha=0.5,
#             linestyle="--",
#             linewidth=0.5,
#         )
#     else:
#         fig_, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
#         color = "gold" if darkmode else "royalblue"
#         ecolor = "gainsboro" if darkmode else "k"
#         (ball,) = ax.plot(
#             ani_x_arr[0],
#             [0],
#             "o",
#             markersize=30,
#             color=color,
#             zorder=100,
#             markeredgecolor=ecolor,
#             markeredgewidth=3,
#         )

#         ax.set_ylim(-1, 1)
#         ax.set_xlim(np.min(ani_x_arr), np.max(ani_x_arr))

#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)
#         ax.spines["left"].set_visible(False)
#         ax.tick_params(axis="x", top=False, which="both")  # hide bottom x-axis ticks
#         ax.yaxis.set_visible(False)  # hide y-axis entirely
#         ax.set_xlabel("Position (m)", fontsize=10)
#         color = "gainsboro" if darkmode else "k"
#         ax.grid(
#             True,
#             which="both",
#             axis="x",
#             color=color,
#             alpha=0.5,
#             linestyle="--",
#             linewidth=0.5,
#         )

#     # --- animation update ---
#     def update(frame, orient=orient):
#         if orient == "vertical":
#             ball.set_data([0], [ani_x_arr[frame]])
#         else:
#             ball.set_data([ani_x_arr[frame]], [0])
#         return (ball,)

#     ani = FuncAnimation(fig_, update, frames=frames, interval=0.0000001, blit=True)

#     # --- save and show in Streamlit ---
#     ani.save("anim.mp4", writer=FFMpegWriter(fps=frames / delta_t))
#     st.video("anim.mp4")
