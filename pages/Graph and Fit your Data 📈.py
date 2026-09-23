import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import streamlit as st
from io import BytesIO
import pandas as pd
import re
from scipy.optimize import curve_fit

st.set_page_config(
    page_title="Graph and Fit your Data", page_icon="📈", layout="centered"
)

ALLOWED_FUNCS = {
    "sqrt": np.sqrt,
    "exp": np.exp,
    "ln": np.log,
    "log": np.log10,
    "log10": np.log10,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "abs": np.abs,
    "pi": np.pi,
    "e": np.e,
}
RESERVED_NAMES = set(ALLOWED_FUNCS.keys()) | {"x"}


def has_valid_xy(x, y):
    return (
        x is not None
        and y is not None
        and len(x) >= 2
        and len(y) >= 2
        and np.all(np.isfinite(x))
        and np.all(np.isfinite(y))
    )


def extract_params(formula):
    """Any identifier in the formula that isn't x or an allowed function name
    is treated as a free parameter to fit, in order of first appearance."""
    tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", formula)
    params = []
    for t in tokens:
        if t not in RESERVED_NAMES and t not in params:
            params.append(t)
    return params


def to_python_expr(formula):
    """Let users write ^ for exponentiation (as in x^2) instead of Python's **."""
    return formula.replace("^", "**")


def make_model_func(formula, params):
    """Build a callable model(x, *args) from a formula string, with args
    bound to `params` in order. Uses eval with no builtins and a locked-down
    namespace (x, the fit parameters, and ALLOWED_FUNCS only)."""
    compiled = compile(to_python_expr(formula), "<model>", "eval")

    def model(x, *args):
        local_vars = dict(zip(params, args))
        local_vars["x"] = x
        local_vars.update(ALLOWED_FUNCS)
        return eval(compiled, {"__builtins__": {}}, local_vars)

    return model


def fmt_value(v, sig=5):
    if v == 0:
        return "0"
    if abs(v) < 1e-4 or abs(v) > 1e4:
        return f"{v:.{sig - 1}e}"
    return f"{v:.{sig}g}"


st.markdown(
    "<h1 style='text-align: center'>Graph and Fit your Data 📈</h1>",
    unsafe_allow_html=True,
)

mode = st.radio(
    "choose data input method:",
    ("manually enter data", "upload data file"),
    horizontal=True,
)


if mode == "upload data file":
    uploaded_file = st.file_uploader("upload a data file", type=["csv", "txt", "xlsx"])

    if uploaded_file is not None:

        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            df = pd.read_csv(uploaded_file, sep="\s+")
        else:
            df = pd.read_csv(uploaded_file, sep=",")
        preview = st.checkbox("preview data", value=False)
        if preview:
            st.dataframe(df.head(), hide_index=True)

        describe = st.checkbox("describe data", value=False)
        if describe:
            st.dataframe(df.describe())

        x_col = st.selectbox("x column", df.columns)
        y_col = st.selectbox("y column", df.columns)

        xdata = df[x_col].values
        ydata = df[y_col].values

elif mode == "manually enter data":

    x_str = st.text_area("x values", "0 1 2 3 4 5")
    y_str = st.text_area("y values", "0 1 4 9 16 25")

    def parse_array(s):
        return np.array([float(v) for v in s.replace(",", " ").split()])

    try:
        xdata = parse_array(x_str)
        ydata = parse_array(y_str)

        if len(xdata) != len(ydata):
            st.error("x and y must have the same length.")
            st.stop()

    except ValueError:
        st.error("could not parse numbers. please check your input.")
        st.stop()


if "xdata" in locals() and "ydata" in locals():

    title = st.text_input("plot title:", "title")

    col1, col2 = st.columns(2)
    xlabel = col1.text_input("x label:", "x axis")
    ylabel = col2.text_input("y label:", "y axis")

    col1, col2, col3 = st.columns(3)
    darkmode = col1.checkbox("dark mode", value=False)
    flipx = col2.checkbox("flip x axis", value=False)
    flipy = col3.checkbox("flip y axis", value=False)

    st.markdown("---")
    fit_models = st.checkbox("fit models", value=False)

    if fit_models:
        st.caption(
            "Write each model as a function of `x` with your own parameter names, e.g. "
            "`a*x + b`, `a*x^2 + b*x + c`, `a*sqrt(x) + b`, `a*exp(-b*x)`. "
            "Any letter that isn't `x` is treated as a fit parameter. "
            "\n\nAvailable functions include: sqrt, exp, ln, log, "
            "sin, cos, tan, abs — plus constants pi and e."
        )

        if "model_formulas" not in st.session_state:
            st.session_state.model_formulas = ["a*x + b"]

        for i in range(len(st.session_state.model_formulas)):
            row_col1, row_col2 = st.columns([6, 1])
            st.session_state.model_formulas[i] = row_col1.text_input(
                f"model {i + 1}",
                value=st.session_state.model_formulas[i],
                key=f"model_input_{i}",
                label_visibility="collapsed",
                placeholder="e.g. a*sqrt(x) + b",
            )
            if len(st.session_state.model_formulas) > 1:
                if row_col2.button("remove", key=f"remove_model_{i}"):
                    st.session_state.model_formulas.pop(i)
                    st.rerun()

        if st.button("add model"):
            st.session_state.model_formulas.append("")
            st.rerun()

    st.markdown("---")

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
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
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

    c = "k"
    edgecolors = "gainsboro"
    fit_colors = [
        "dodgerblue",
        "orangered",
        "limegreen",
        "gold",
        "mediumorchid",
        "deepskyblue",
    ]

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
        c = "gainsboro"
        edgecolors = "w"
        fit_colors = ["lime", "cyan", "yellow", "magenta", "orange", "white"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xdata, ydata, s=60, c=c, edgecolors=edgecolors, lw=1, zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both")
    if flipx:
        ax.invert_xaxis()
    if flipy:
        ax.invert_yaxis()

    if fit_models and has_valid_xy(xdata, ydata):
        x_fit = np.linspace(np.min(xdata), np.max(xdata), 500)
        any_fit_plotted = False

        for i, formula in enumerate(st.session_state.model_formulas):
            formula = formula.strip()
            if not formula:
                continue

            params = extract_params(formula)
            if not params:
                st.warning(
                    f"model {i + 1} (`{formula}`) has no free parameters to fit."
                )
                continue

            try:
                model_func = make_model_func(formula, params)
                popt, pcov = curve_fit(
                    model_func, xdata, ydata, p0=np.ones(len(params)), maxfev=10000
                )
                y_fit = model_func(x_fit, *popt)

                param_lines = "\n".join(
                    f"{name} = {fmt_value(val)}" for name, val in zip(params, popt)
                )
                label = f"y = {formula}\n{param_lines}"

                color = fit_colors[i % len(fit_colors)]
                ax.plot(x_fit, y_fit, color=color, lw=2, label=label)
                any_fit_plotted = True

            except Exception as e:
                st.warning(f"could not fit model {i + 1} (`{formula}`): {e}")

        if any_fit_plotted:
            ax.legend(fontsize=10, labelspacing=1.2, loc="best", framealpha=0.8)

    st.pyplot(fig)

    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    st.download_button(
        label="download graph",
        data=buf,
        file_name=f"{title.replace(' ', '_')}.png",
        mime="image/png",
    )
