import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from typing import Union, List, Optional
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from .utils import floor_significant_digits, calculate_nb, _threshold_to_cost_benefit
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as patches
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy import stats
import warnings
import matplotlib.ticker as ticker


def train_test_lift(
        train:Union[List[List], np.ndarray],
        test:Union[List[List], np.ndarray],
        paired:bool=True,
        colors:Optional[List[str]]=None,
        labels:Optional[List[str]]=None,
        yticks_interval:Optional[int|float]=None,
        axis_range:Optional[List[Optional[int|float]]]=None,
        offset:Optional[int|float]=None
) -> tuple[Figure,Axes]:
    """
    Plot lifted histogram comparison between training and test distributions.

    Visualize two groups of data (training vs test) as bar charts, with the test
    bars lifted vertically for clear separation. Dual Y-axis ticks are drawn on
    the left and right to match each distribution’s baseline. Suitable for length
    distribution, value count, or density comparison in data analysis pipelines.

    Parameters
    ----------
    train : list of lists or np.ndarray
        Training data, either as paired [[x1, y1], ...] or separated [x_vals, y_vals].
    test : list of lists or np.ndarray
        Test data, in the same format as training data.
    paired : bool, default=True
        If True, input arrays are treated as paired points: [[x1, y1], [x2, y2], ...].
        If False, inputs are separated coordinates: ([x1, x2, ...], [y1, y2, ...]).
    colors : list of str, optional
        Two-element color list for training and test bars.
        Defaults to muted dark pink and deep blue.
    labels : list of str, optional
        Legend labels for training and test sets. Default: ["Train", "Test"].
    yticks_interval : int or float, optional
        Step interval for Y-axis ticks. If None, computed automatically from data range.
    axis_range : list of int/float/None, optional
        Axis limits in the form [X_min, X_max, Y_min, Y_max].
        Use None to auto-compute a given limit.
    offset : int or float, optional
        Vertical offset to lift test bars. If None, set to half the tick interval.

    Returns
    -------
    Figure
        The figure object containing the plot.
    Axes
        Matplotlib Axes object containing the finished plot for further styling.

    Examples
    --------
    >>> from plotcraft.draw import train_test_lift
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> train_data = np.arange(21, 100,dtype=int)
    >>> sigma, mu = 15, 60
    >>> y = np.exp(-(train_data - mu) ** 2 / (2 * sigma ** 2))
    >>> train_count = (y * 50 + 10).astype(int)
    >>> test_data = train_data.copy()
    >>> test_count = train_count.copy()
    >>> fig, ax = train_test_lift([train_data,train_count],[test_data,test_count],paired=False)
    >>> ax.set_xlabel('Length', fontsize=11)
    >>> ax.set_ylabel('Frequency', fontsize=11, labelpad=35)
    >>> plt.show()
    """
    train = np.array(train)
    test = np.array(test)
    if paired:
        train_x = train[:,0]
        train_y = train[:,1]
        test_x = test[:,0]
        test_y = test[:,1]
    else:
        train_x, train_y = train
        test_x, test_y = test

    if axis_range is None:
        X_min = min(min(train_x), min(test_x))
        X_max = max(max(train_x), max(test_x))
        Y_min = 0
        Y_max = max(max(train_y), max(test_y))
    else:
        X_min, X_max, Y_min, Y_max = axis_range
        if X_min is None:
            X_min = min(min(train_x), min(test_x))
        if X_max is None:
            X_max = max(max(train_x), max(test_x))
        if Y_min is None:
            Y_min = 0
        if Y_max is None:
            Y_max = max(max(train_y), max(test_y))

    if labels is None:
        labels = ["Train", "Test"]

    if colors is None:
        colors = ['#E0726D', '#5187B0']

    if yticks_interval is None:
        yticks_interval = floor_significant_digits((Y_max - Y_min)/4, 2)
    tick_vals = np.arange(Y_min,Y_max,yticks_interval)

    if offset is None:
        offset = yticks_interval / 2

    fig, ax = plt.subplots()

    ax.bar(train_x, train_y, alpha=0.5,
           color=colors[0], edgecolor='white', linewidth=0.5, label=labels[0])

    ax.bar(test_x, test_y, bottom=offset, alpha=0.5,
           color=colors[1], edgecolor='white', linewidth=0.5, label=labels[1])

    ax.set_xlim(X_min-1, X_max+1)
    ax.set_ylim(Y_min, Y_max + offset)
    ax.set_yticks([])

    ax.axhline(y=offset, color='#888888', linestyle='--', linewidth=1.5, dashes=(5, 2), alpha=0.8)

    blend = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    for i, v in enumerate(tick_vals):
        ax.text(-0.03, v, f'{v:.2f}', transform=blend,
                fontsize=8, color=colors[0], va='center', ha='right')
        ax.plot([-0.02, 0], [v, v], color=colors[0], linewidth=0.8,
                clip_on=False, transform=blend)

        if i:
            ax.text(0.03, v + offset, f'{v:.2f}', transform=blend,
                    fontsize=8, color=colors[1], va='center', ha='left')
            ax.plot([0, 0.02], [v + offset, v + offset], color=colors[1],
                    linewidth=0.8, clip_on=False, transform=blend)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.legend(frameon=True, fontsize=9, loc='upper right')
    plt.subplots_adjust(left=0.18)
    return fig, ax


def triangular_heatmap(
        data: pd.DataFrame | np.ndarray,
        annot: bool = True,
        annot_kws: Optional[dict] = None,
        linewidths: float | int = 1.5,
        linecolor: str = 'white',
        ticks_size: int | float = 9,
        vmin: float | int = -1,
        vmax: float | int = 1,
        cmap: str | plt.Colormap = None,
        norm: Normalize = None
) -> tuple[Figure,Axes]:
    """
    Draw a heatmap of a triangle.

    This function creates a triangular heatmap using diamond-shaped cells to visualize
    the lower triangular part of a square correlation matrix. It supports custom color
    mapping, value annotations, and styling of cell borders and labels.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Square matrix (n×n) containing correlation values. Only the lower triangular
        part of the matrix will be visualized. If a DataFrame is provided, column names
        will be used as variable labels; if a numpy array is provided, labels will be
        automatically generated as Var1, Var2, ..., Varn.

    annot : bool, default=True
        Whether to display numerical values inside each diamond cell.

    annot_kws : dict or None, default=None
        Keyword arguments for customizing the annotation text. Supported keys:
        - 'size': Font size of the annotation (default: 20)
        - 'color': Fixed text color; if not specified, text color will be white for
          values with absolute value > 0.60, otherwise dark gray (#222222)
        - 'fontweight': Font weight (default: 'bold')
        - 'fontfamily': Font family (default: None, inherits global settings)

    linewidths : float or int, default=1.5
        Width of the border lines between diamond cells.

    linecolor : str, default='white'
        Color of the border lines between diamond cells.

    ticks_size : float or int, default=9
        Font size of the variable name labels on the triangular axes.

    vmin : float or int, default=-1
        Minimum value for color normalization. Values less than or equal to vmin
        will be mapped to the bottom color of the colormap.

    vmax : float or int, default=1
        Maximum value for color normalization. Values greater than or equal to vmax
        will be mapped to the top color of the colormap.

    cmap : str or matplotlib.colors.Colormap, default=None
        Colormap used for mapping correlation values to colors. If None, 'RdBu_r'
        (red-blue reversed) will be used.

    norm : matplotlib.colors.Normalize, default=None
        Normalization object to scale data values to the [0, 1] range for colormap
        mapping. If None, a basic Normalize instance with vmin and vmax will be used.
        Other options include CenteredNorm or TwoSlopeNorm for asymmetric scaling.

    Returns
    -------
    Figure
        The figure object containing the plot.
    Axes
        Matplotlib Axes object containing the finished plot for further styling.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy import stats
    >>> from plotcraft.draw import triangular_heatmap
    >>> n_samples, n_vars = 200, 20
    >>> data = np.random.randn(n_samples, n_vars)
    >>> cols = [f"Var{i+1}" for i in range(n_vars)]
    >>> df = pd.DataFrame(data, columns=cols)
    >>> n = n_vars
    >>> corr = np.ones((n, n))
    >>> for i in range(n):
    ...     for j in range(i + 1, n):
    ...         r, _ = stats.spearmanr(df.iloc[:, i], df.iloc[:, j])
    ...         corr[i, j] = r
    ...         corr[j, i] = r
    >>> corr_df = pd.DataFrame(corr, index=cols, columns=cols)
    >>> fig, ax = triangular_heatmap(
    ...     corr_df,
    ...     annot=True,
    ...     annot_kws={'size': 7.2},
    ...     linewidths=0.5,
    ...     linecolor='white',
    ...     ticks_size=8,
    ...     vmax=1,
    ...     vmin=-1,
    ... )
    >>> plt.show()
    """

    assert vmax > vmin
    if isinstance(data, pd.DataFrame):
        columns = list(data.columns)
        corr = data.values
    else:
        corr = np.asarray(data)
        columns = [f"Var{i+1}" for i in range(corr.shape[0])]

    n = corr.shape[0]
    assert corr.shape == (n, n), "data 必须是方阵"

    _annot_kws = {'size': 20, 'fontweight': 'bold', 'fontfamily': None, 'color': None}
    if annot_kws:
        _annot_kws.update(annot_kws)

    def to_canvas(row, col):
        cx = 2 * (n - 1) - (row + col)
        cy = row - col
        return cx, cy

    half = 1.0

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    if cmap is None:
        cmap = 'RdBu_r'
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    if norm is None:
        norm_c = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm_c = norm

    for row in range(n):
        for col in range(row + 1):
            val   = corr[row, col]
            color = cmap(norm_c(val))
            cx, cy = to_canvas(row, col)

            diamond = patches.Polygon(
                [(cx, cy+half), (cx+half, cy), (cx, cy-half), (cx-half, cy)],
                closed=True,
                facecolor=color,
                edgecolor=linecolor,
                linewidth=linewidths,
                zorder=2,
            )
            ax.add_patch(diamond)

            if annot:
                if _annot_kws['color'] is not None:
                    txt_color = _annot_kws['color']
                else:
                    txt_color = 'white' if abs(val) > 0.60 else '#222222'

                txt_kws = dict(
                    ha='center', va='center', zorder=3,
                    fontsize=_annot_kws['size'],
                    color=txt_color,
                    fontweight=_annot_kws['fontweight'],
                )
                if _annot_kws['fontfamily']:
                    txt_kws['fontfamily'] = _annot_kws['fontfamily']

                ax.text(cx, cy, f'{val:.2f}', **txt_kws)

    t = n * 0.005 + 0.6
    offset = 0.18
    sq2    = np.sqrt(2)

    for i in range(n):
        cx, cy = to_canvas(i, 0)
        lx  = cx + half * t + offset / sq2
        ly  = cy + half * (1 - t) + offset / sq2
        ax.text(lx, ly, columns[i],
                ha='left', va='bottom',
                fontsize=ticks_size, rotation=45,
                rotation_mode='anchor', zorder=4)

        cx2, cy2 = to_canvas(n - 1, i)
        lx2 = cx2 - half * t - offset / sq2
        ly2 = cy2 + half * (1 - t) + offset / sq2
        ax.text(lx2, ly2, columns[i],
                ha='right', va='bottom',
                fontsize=ticks_size, rotation=-45,
                rotation_mode='anchor', zorder=4)

    sm = ScalarMappable(cmap=cmap, norm=norm_c)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.01, shrink=0.65, aspect=22)
    cbar.set_ticks(np.linspace(vmin,vmax,9))
    cbar.ax.tick_params(labelsize=8.5)
    cbar.outline.set_linewidth(0.5)

    ax.set_xlim(-half - 3.0, 2*(n-1) + half + 3.0)
    ax.set_ylim(-half - 0.5, (n-1) + half + 2.5)

    plt.tight_layout()
    return fig, ax

def enlarged_roc_curve(
        *true_score_pairs: List[List] | np.ndarray | pd.DataFrame,
        dataframe_cols: List[str] = None,
        colors:Optional[List[str]]=None,
        labels:Optional[List[str]]=None,
        paired:bool=False,
        calculate:bool=True,
        plot_kwargs:dict=None,
        enlarged:bool=False,
        to_enlarge_frame_location:List[int|float]=None,
        enlarged_frame_location:List[int|float]=None,
        enlarged_frame_xticks:List[int|float]=None,
        enlarged_frame_yticks:List[int|float]=None,
        enlarged_frame_transparent:bool=True,
        legend_kwargs:dict=None
) -> tuple[Figure,Axes]:
    """
    Plot ROC curves with optional local zoom-in functionality.

    Convenience function to draw ROC curves for one or multiple models,
    compute AUC scores, and add an inset axes to magnify a region of interest
    in the ROC space (typically low FPR, high TPR).

    Parameters
    ----------
    *true_score_pairs : sequence of array-like | dataframe
        Each argument is a pair (y_true, y_score). Multiple pairs can be
        passed to compare PR curves across models.

    dataframe_cols : list of str, default=None
        If you input "dataframe", please enter a one-dimensional list of length 2, like[true_column_name, score_column_name].
        If it is None, then the default list will be ["true", "score"].

    colors : list of str, default=None
        List of colors for each ROC curve. Length must match the number
        of model pairs provided.

    labels : list of str, default=None
        List of labels for each ROC curve. Length must match the number
        of model pairs provided.

    paired : bool, default=False
        If True, each input pair is expected to be an N x 2 array
        where each row is [y_true, score].
        If False, each input pair is interpreted as two 1D arrays:
        [y_true_array, score_array].

    calculate : bool, default=True
        Whether to compute and display AUC in the legend label.

    plot_kwargs : dict, default=None
        Keyword arguments passed to ax.plot() for ROC curves,
        e.g., linewidth, linestyle, alpha.

    enlarged : bool, default=False
        Whether to create an inset axes with a zoomed view of a subregion.

    to_enlarge_frame_location : list of float, length 4
        Region in main axes to magnify, specified as [x1, y1, x2, y2]
        in [0,1] coordinates, where (x1,y1) is lower-left and (x2,y2) upper-right.

    enlarged_frame_location : list of float, length 4
        Position of the inset axes within the main axes, in relative coordinates:
        [x1, y1, x2, y2] lower-left to upper-right.

    enlarged_frame_xticks : array-like, default=None
        Custom tick positions for the x-axis of the inset plot.

    enlarged_frame_yticks : array-like, default=None
        Custom tick positions for the y-axis of the inset plot.

    enlarged_frame_transparent : bool, default=True
        Whether to make the background of the inset plot transparent.

    legend_kwargs : dict, default=None
        Keyword arguments passed to ax.legend(), e.g., fontsize, loc.

    Returns
    -------
    Figure
        The figure object containing the plot.
    Axes
        Matplotlib Axes object containing the finished plot for further styling.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from plotcraft.draw import enlarged_roc_curve
    >>> arr = np.load('examples/data/true_score.npy')
    >>> data_list = [[arr[i], arr[i+1]] for i in range(0, arr.shape[0], 2)]
    >>> fig, ax = enlarged_roc_curve(
    ...     *data_list,
    ...     labels=[f'model{i}' for i in range(len(data_list))],
    ...     enlarged=True,
    ...     to_enlarge_frame_location=[0.01, 0.80, 0.15, 0.98],
    ...     enlarged_frame_location=[0.3, 0.5, 0.4, 0.4],
    ...     enlarged_frame_xticks=[0.045, 0.08, 0.115],
    ...     enlarged_frame_yticks=[0.9, 0.93, 0.96]
    ... )
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=(8,8))

    ax.plot([0, 1], [0, 1], color="lightgray", linestyle="--")

    fpr_list, tpr_list = [], []
    for i, true_score_pair in enumerate(true_score_pairs):
        if isinstance(true_score_pair, pd.DataFrame):
            if dataframe_cols is None:
                dataframe_cols = ['true', 'score']
            y_true, score = true_score_pair[dataframe_cols[0]].values, true_score_pair[dataframe_cols[1]].values
        else:
            true_score_pair = np.array(true_score_pair)
            if paired:
                y_true, score = true_score_pair[:, 0], true_score_pair[:, 1]
            else:
                y_true, score = true_score_pair
        fpr, tpr, _ = roc_curve(y_true, score)
        if calculate:
            roc_auc = auc(fpr, tpr)
            add_str = f"(AUC = {roc_auc:.3f})"
        else:
            add_str = ""
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        parameters = {}
        if colors is not None:
            parameters['color'] = colors[i]
        if labels is not None:
            parameters['label'] = labels[i] + add_str
        if plot_kwargs is not None:
            parameters.update(plot_kwargs)
        else:
            parameters['linewidth'] = 2

        ax.plot(fpr, tpr, **parameters)

    ax.spines[["top", "left"]].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    ax.set_xlabel("False positive rate", fontsize=22, labelpad=10)
    ax.set_ylabel("True positive rate", fontsize=22, labelpad=20)
    ax.set_title("ROC curve", fontsize=22, pad=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if labels is not None:
        if legend_kwargs is None:
            legend_kwargs = {'fontsize':12}
        ax.legend(loc="lower right",**legend_kwargs)
    ax.grid(False)

    if enlarged:
        assert to_enlarge_frame_location is not None
        assert enlarged_frame_location is not None
        x1, y1, x2, y2 = to_enlarge_frame_location
        assert 0 <= x1 < x2 <=1
        assert 0 <= y1 < y2 <=1
        axins = ax.inset_axes(enlarged_frame_location,
                              xlim=(x1, x2), ylim=(y1, y2))

        if enlarged_frame_transparent:
            axins.patch.set_alpha(0.0)

        for i, (fpr, tpr) in enumerate(zip(fpr_list, tpr_list)):
            parameters = {}
            if colors is not None:
                parameters['color'] = colors[i]
            if plot_kwargs is not None:
                parameters.update(plot_kwargs)
            else:
                parameters['linewidth'] = 2
            axins.plot(fpr, tpr, **parameters)

        axins.yaxis.tick_right()
        if enlarged_frame_xticks is not None:
            axins.set_xticks(enlarged_frame_xticks)
        if enlarged_frame_yticks is not None:
            axins.set_yticks(enlarged_frame_yticks)
        axins.grid(False)

        ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1.5)

    plt.tight_layout()
    return fig, ax

def enlarged_pr_curve(
        *true_score_pairs: List[List] | np.ndarray | pd.DataFrame,
        dataframe_cols:List[str]=None,
        colors:Optional[List[str]]=None,
        labels:Optional[List[str]]=None,
        paired:bool=False,
        calculate:bool=True,
        plot_kwargs:dict=None,
        enlarged:bool=False,
        to_enlarge_frame_location:List[int|float]=None,
        enlarged_frame_location:List[int|float]=None,
        enlarged_frame_xticks:List[int|float]=None,
        enlarged_frame_yticks:List[int|float]=None,
        enlarged_frame_transparent:bool=True,
        legend_kwargs:dict=None
) -> tuple[Figure, Axes]:
    """
    Plot PR curves with optional local zoom-in functionality.

    Convenience function to draw PR curves for one or multiple models,
    compute AUC scores, and add an inset axes to magnify a region of interest
    in the PR space (typically high Recall, high Precision).

    Parameters
    ----------
    *true_score_pairs : sequence of array-like | dataframe
        Each argument is a pair (y_true, y_score). Multiple pairs can be
        passed to compare PR curves across models.

    dataframe_cols : list of str, default=None
        If you input "dataframe", please enter a one-dimensional list of length 2, like[true_column_name, score_column_name].
        If it is None, then the default list will be ["true", "score"].

    colors : list of str, default=None
        List of colors for each PR curve. Length must match the number
        of model pairs provided.

    labels : list of str, default=None
        List of labels for each PR curve. Length must match the number
        of model pairs provided.

    paired : bool, default=False
        If True, each input pair is expected to be an N x 2 array
        where each row is [y_true, score].
        If False, each input pair is interpreted as two 1D arrays:
        [y_true_array, score_array].

    calculate : bool, default=True
        Whether to compute and display AUC in the legend label.

    plot_kwargs : dict, default=None
        Keyword arguments passed to ax.plot() for PR curves,
        e.g., linewidth, linestyle, alpha.

    enlarged : bool, default=False
        Whether to create an inset axes with a zoomed view of a subregion.

    to_enlarge_frame_location : list of float, length 4
        Region in main axes to magnify, specified as [x1, y1, x2, y2]
        in [0,1] coordinates, where (x1,y1) is lower-left and (x2,y2) upper-right.

    enlarged_frame_location : list of float, length 4
        Position of the inset axes within the main axes, in relative coordinates:
        [x1, y1, x2, y2] lower-left to upper-right.

    enlarged_frame_xticks : array-like, default=None
        Custom tick positions for the x-axis of the inset plot.

    enlarged_frame_yticks : array-like, default=None
        Custom tick positions for the y-axis of the inset plot.

    enlarged_frame_transparent : bool, default=True
        Whether to make the background of the inset plot transparent.

    legend_kwargs : dict, default=None
        Keyword arguments passed to ax.legend(), e.g., fontsize, loc.

    Returns
    -------
    Figure
        The figure object containing the plot.
    Axes
        Matplotlib Axes object containing the finished plot for further styling.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> arr = np.load('./data/true_score.npy')
    >>> data_list = [[arr[i], arr[i+1]] for i in range(0, arr.shape[0], 2)]
    >>> fig, ax = enlarged_pr_curve(*data_list,
    ...     labels=[f'model{i}' for i in range(len(datas))],
    ...     enlarged=True,
    ...     to_enlarge_frame_location=[0.82,0.75,0.97,0.93],
    ...     enlarged_frame_location=[0.3, 0.5, 0.4, 0.4],
    ...     enlarged_frame_xticks=[0.858,0.895,0.93],
    ...     enlarged_frame_yticks=[0.795, 0.84, 0.885]
    ... )
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    precision_list, recall_list = [], []
    for i, true_score_pair in enumerate(true_score_pairs):
        if isinstance(true_score_pair, pd.DataFrame):
            if dataframe_cols is None:
                dataframe_cols = ['true', 'score']
            y_true, score = true_score_pair[dataframe_cols[0]].values, true_score_pair[dataframe_cols[1]].values
        else:
            true_score_pair = np.array(true_score_pair)
            if paired:
                y_true, score = true_score_pair[:, 0], true_score_pair[:, 1]
            else:
                y_true, score = true_score_pair
        precision, recall, _ = precision_recall_curve(y_true, score)
        if calculate:
            AP = average_precision_score(y_true, score)
            add_str = f"(AUC = {AP:.3f})"
        else:
            add_str = ""
        precision_list.append(precision)
        recall_list.append(recall)
        parameters = {}
        if colors is not None:
            parameters['color'] = colors[i]
        if labels is not None:
            parameters['label'] = labels[i] + add_str
        if plot_kwargs is not None:
            parameters.update(plot_kwargs)
        else:
            parameters['linewidth'] = 2

        ax.plot(recall, precision, **parameters)

    ax.spines[["top", "right"]].set_visible(False)

    ax.set_xlabel("Recall", fontsize=22, labelpad=10)
    ax.set_ylabel("Precision", fontsize=22, labelpad=20)
    ax.set_title("PR curve", fontsize=22, pad=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if labels is not None:
        if legend_kwargs is None:
            legend_kwargs = {'fontsize': 12}
        ax.legend(loc="lower left", **legend_kwargs)
    ax.grid(False)

    if enlarged:
        assert to_enlarge_frame_location is not None
        assert enlarged_frame_location is not None
        x1, y1, x2, y2 = to_enlarge_frame_location
        assert 0 <= x1 < x2 <= 1
        assert 0 <= y1 < y2 <= 1
        axins = ax.inset_axes(enlarged_frame_location,
                              xlim=(x1, x2), ylim=(y1, y2))
        if enlarged_frame_transparent:
            axins.patch.set_alpha(0.0)

        for i, (recall, precision) in enumerate(zip(recall_list, precision_list)):
            parameters = {}
            if colors is not None:
                parameters['color'] = colors[i]
            if plot_kwargs is not None:
                parameters.update(plot_kwargs)
            else:
                parameters['linewidth'] = 2
            axins.plot(recall, precision, **parameters)

        if enlarged_frame_xticks is not None:
            axins.set_xticks(enlarged_frame_xticks)
        if enlarged_frame_yticks is not None:
            axins.set_yticks(enlarged_frame_yticks)
        axins.grid(False)

        ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1.5)

    plt.tight_layout()
    return fig, ax

def correlation_graph_between_prediction_and_reality(
        real:np.ndarray|List|pd.Series,
        pred:np.ndarray|List|pd.Series,
        correlation:Optional[callable]=None
) -> tuple[Figure, Axes]:
    """
    Scatter plot of true vs. predicted values with correlation coefficient.

    Generates a scatter plot to visualize the relationship between real (true) values and
    predicted values, with a diagonal reference line (y=x) and the correlation coefficient
    displayed in the top-left corner.

    Parameters
    ----------
    real : np.ndarray or List or pd.Series
        Ground truth (real) values. Will be flattened to 1D if input is multi-dimensional.
    pred : np.ndarray or List or pd.Series
        Predicted values. Must have the same length as `real`. Will be flattened to 1D if
        input is multi-dimensional.
    correlation : callable, default=None
        Function to compute the correlation coefficient. If None, uses `scipy.stats.pearsonr`.
        If a callable is provided, it must take two arrays (real, pred) as input and return a
        tuple (value, ...), where the first element is the correlation coefficient to display.
        The signature should match `scipy.stats.pearsonr`.

    Returns
    -------
    fig : Figure
        Matplotlib Figure object containing the plot.
    ax : Axes
        Matplotlib Axes object containing the plot.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from plotcraft.draw import correlation_graph_between_prediction_and_reality
    >>> real = np.random.randn(1000)
    >>> pred = real + np.random.randn(1000) * 0.5
    >>> fig, ax = correlation_graph_between_prediction_and_reality(real, pred)
    >>> plt.show()

    >>> real = np.random.randn(1000)
    >>> pred = real.copy()
    >>> fig, ax = correlation_graph_between_prediction_and_reality(real, pred)
    >>> plt.show()
    """
    fig = plt.figure(figsize=(8, 8))
    real = np.array(real).ravel()
    pred = np.array(pred).ravel()
    plt.scatter(real, pred, color='#3388dd', alpha=0.6, s=40, edgecolors='none')
    range_min = min(min(real), min(pred))
    range_max = max(max(real), max(pred))
    dis = range_max - range_min
    range_min -= dis * 0.05
    range_max += dis * 0.05
    plt.xlim(range_min, range_max)
    plt.ylim(range_min, range_max)
    plt.plot([range_min, range_max], [range_min, range_max], '--', color='grey')
    if correlation is None:
        correlation = stats.pearsonr
    r_value, _ = correlation(real, pred)
    r_text = f"R = {r_value:.2f}"
    ax = plt.gca()
    ax.text(0.02, 0.98, r_text,
            transform=ax.transAxes,
            fontsize=32, fontweight='bold', color='#bb2222',
            va='top', ha='left')
    plt.tight_layout()
    return fig,ax


def dca_curve(
        *dataframes: pd.DataFrame,
        dataframe_cols: List[str],
        thresholds: Optional[np.ndarray | List[str]] = None,
        confidence_intervals: Optional[float] = None,
        bootstraps: int = 500,
        policy: str = "opt-in",
        study_design: str = "cohort",
        population_prevalence: float | None = None,
        random_state: int = 42,
        model_names: List[str] = None,
        cost_benefit_axis: bool = True,
        colors: Optional[List[str]] = None,
):
    """
    Plot Decision Curve Analysis (DCA) for one or more prediction models.

    Compute and visualize the standardized net benefit (sNB) across a range
    of risk thresholds, along with the "Treat All" and "Treat None"
    reference strategies.  Optionally adds bootstrap confidence intervals
    and a secondary cost:benefit ratio axis.

    This implementation mirrors the methodology of the R ``dcurves``
    package (``decision_curve``).

    Parameters
    ----------
    *dataframes : sequence of pandas.DataFrame
        One or more DataFrames, each containing at least the outcome column
        and the predicted probability column specified in `dataframe_cols`.
        All DataFrames must share the same column names given by
        `dataframe_cols`.  When multiple DataFrames are supplied, each is
        treated as a separate model; the "Treat All" / "Treat None"
        reference curves are drawn only from the first DataFrame.

    dataframe_cols : list of str, length = 2
        Column names to use.  ``dataframe_cols[0]`` is the binary outcome
        variable (coded 0/1) and ``dataframe_cols[1]`` is the predicted
        probability of the outcome (values in [0, 1]).

    thresholds : array-like of float, default=None
        Risk-threshold grid on which net benefit is evaluated.
        Each element must lie in [0, 1].  If None, defaults to
        ``np.arange(0.01, 1.01, 0.01)``.

    confidence_intervals : float, default=None
        If not None, a value in (0, 1) giving the confidence level for
        bootstrap confidence intervals (e.g. 0.95 for 95 % CIs).
        When None, no confidence intervals are computed.

    bootstraps : int, default=500
        Number of bootstrap resamples used to estimate confidence intervals.
        Ignored when ``confidence_intervals`` is None.

    policy : {'opt-in', 'opt-out'}, default='opt-in'
        Clinical policy direction.

        - ``'opt-in'``:  patients are treated only if their predicted risk
          exceeds the threshold (the standard DCA scenario).
        - ``'opt-out'``: patients are treated by default and only opt out
          of treatment when their predicted risk falls below the threshold.

    study_design : {'cohort', 'case-control'}, default='cohort'
        Study design from which the data originate.  When
        ``'case-control'``, the ``population_prevalence`` parameter is
        required to re-calibrate net-benefit calculations.

    population_prevalence : float or None, default=None
        Known disease prevalence in the target population.
        Required when ``study_design='case-control'``; ignored (with a
        warning) when ``study_design='cohort'``.

    random_state : int, default=42
        Seed for the random number generator used in bootstrap resampling.
        Pass an int for reproducible confidence intervals across multiple
        function calls.

    model_names : list of str or None, default=None
        Display names for each model in the legend.  Must have the same
        length as the number of DataFrames.  If None, defaults to
        ``['model 0', 'model 1', ...]``.

    cost_benefit_axis : bool, default=True
        If True, a secondary x-axis is drawn showing the cost:benefit
        ratio that corresponds to each threshold value.

    colors : list of str or None, default=None
        Matplotlib-compatible color specifications for the model curves.
        If None, the current ``axes.prop_cycle`` colors are used.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Figure object containing the DCA plot.

    ax : matplotlib.axes.Axes
        The primary Axes object of the plot, which can be used for further
        customisation.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> from plotcraft.draw import dca_curve

    >>> array = np.load('./data/true_score.npy')
    >>> datas = [
    ...     pd.DataFrame(
    ...         np.array([array[i], array[i + 1]]).T,
    ...         columns=['true', 'pred'],
    ...     )
    ...     for i in range(0, array.shape[0], 2)
    ... ]

    >>> fig, ax = dca_curve(
    ...     *datas,
    ...     dataframe_cols=['true', 'pred'],
    ...     thresholds=np.arange(0.01, 0.11, 0.01),
    ... )
    >>> plt.show()

    >>> fig, ax = dca_curve(
    ...     *datas,
    ...     dataframe_cols=['true', 'pred'],
    ...     thresholds=np.arange(0.01, 1.01, 0.01),
    ... )
    >>> plt.show()

    >>> fig, ax = dca_curve(
    ...     datas[0],
    ...     dataframe_cols=['true', 'pred'],
    ...     thresholds=np.arange(0.01, 1.01, 0.01),
    ...     confidence_intervals=0.95,
    ... )
    >>> plt.show()
    """
    assert len(dataframe_cols) == 2
    real_col, score_col = dataframe_cols
    if thresholds is None:
        thresholds = np.arange(0.01, 1.01, 0.01)
    else:
        thresholds = np.array(thresholds)
        assert (0 <= thresholds).all()
        assert (thresholds <= 1).all()
    if confidence_intervals is not None:
        assert 0 < confidence_intervals < 1
    assert isinstance(bootstraps, int)
    assert policy in ("opt-in", "opt-out")
    assert study_design in ("cohort", "case-control")
    opt_in = policy == "opt-in"

    if study_design == "case-control":
        if population_prevalence is None:
            raise ValueError("In a case-control study, population prevalence needs to be provided.")
        casecontrol_rho = population_prevalence
    else:
        if population_prevalence is not None:
            warnings.warn("When study_design is set to 'cohort', the population_prevalence will be ignored.")
        casecontrol_rho = None

    if model_names is None:
        model_names = [f"model {i}" for i in range(len(dataframes))]

    rng = np.random.default_rng(random_state)

    def _calculate(real, score, thresholds, casecontrol_rho, opt_in, B_ind, confidence_intervals,
                   calculate_all_none=False):
        if not calculate_all_none:
            nb_df = calculate_nb(
                real, score, thresholds=thresholds,
                casecontrol_rho=casecontrol_rho, opt_in=opt_in,
            )
            nb_df["model"] = "pred"
            if B_ind is not None:
                alpha = 1 - confidence_intervals
                boot_snb = np.zeros((len(thresholds), bootstraps))
                boot_nb = np.zeros((len(thresholds), bootstraps))

                for b in range(bootstraps):
                    idx = B_ind[:, b]
                    real_b = real[idx]
                    score_b = score[idx]
                    try:
                        tmp = calculate_nb(
                            real_b, score_b, thresholds=thresholds,
                            casecontrol_rho=casecontrol_rho, opt_in=opt_in,
                        )
                        boot_snb[:, b] = tmp["sNB"].values
                        boot_nb[:, b] = tmp["NB"].values
                    except Exception:
                        boot_snb[:, b] = np.nan
                        boot_nb[:, b] = np.nan

                nb_df["sNB_lower"] = np.nanquantile(boot_snb, alpha / 2, axis=1)
                nb_df["sNB_upper"] = np.nanquantile(boot_snb, 1 - alpha / 2, axis=1)
                nb_df["NB_lower"] = np.nanquantile(boot_nb, alpha / 2, axis=1)
                nb_df["NB_upper"] = np.nanquantile(boot_nb, 1 - alpha / 2, axis=1)

            return nb_df
        else:
            nb_pred = _calculate(
                real, score, thresholds=thresholds,
                casecontrol_rho=casecontrol_rho, opt_in=opt_in, B_ind=B_ind, confidence_intervals=confidence_intervals
            )
            nb_pred["model"] = "pred"
            nb_all = _calculate(
                real, np.ones_like(real), thresholds=thresholds,
                casecontrol_rho=casecontrol_rho, opt_in=opt_in, B_ind=B_ind, confidence_intervals=confidence_intervals
            )
            nb_all["model"] = "All"
            nb_none = _calculate(
                real, np.zeros_like(real), thresholds=thresholds,
                casecontrol_rho=casecontrol_rho, opt_in=opt_in, B_ind=B_ind, confidence_intervals=confidence_intervals
            )
            nb_none["model"] = "None"
            return pd.concat([nb_pred, nb_all, nb_none], ignore_index=True)

    ans = []
    for i, dataframe in enumerate(dataframes):
        assert isinstance(dataframe, pd.DataFrame)
        dataframe = dataframe[dataframe_cols].copy()
        real = dataframe[real_col].values

        B_ind = None
        if confidence_intervals is not None:
            n = len(dataframe)
            B_ind = np.zeros((n, bootstraps), dtype=int)
            if study_design == "cohort":
                for b in range(bootstraps):
                    B_ind[:, b] = rng.integers(0, n, size=n)
            else:
                idx_pos = np.where(real == 1)[0]
                idx_neg = np.where(real == 0)[0]
                for b in range(bootstraps):
                    s_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
                    s_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
                    B_ind[:, b] = np.concatenate([s_pos, s_neg])

        score = dataframe[score_col].values

        if not i:
            nb_df = _calculate(
                real, score, thresholds=thresholds,
                casecontrol_rho=casecontrol_rho, opt_in=opt_in, B_ind=B_ind, confidence_intervals=confidence_intervals,
                calculate_all_none=True
            )
            nb_df["cost_benefit_ratio"] = np.tile(
                _threshold_to_cost_benefit(thresholds, policy),
                3,
            )
        else:
            nb_df = _calculate(real, score, thresholds=thresholds,
                               casecontrol_rho=casecontrol_rho, opt_in=opt_in, B_ind=B_ind,
                               confidence_intervals=confidence_intervals)
            nb_df["cost_benefit_ratio"] = np.tile(
                _threshold_to_cost_benefit(thresholds, policy),
                1,
            )
        ans.append(nb_df)

    color_map = {"All": "grey", "None": "black"}
    lw_map = {"All": 0.2, "None": 1.2}
    if colors is None:
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    else:
        default_colors = colors
    fig, ax = plt.subplots()

    for i, df in enumerate(ans):
        if not i:
            labels = ['All', 'None', 'pred']
            for label in labels:
                sub = df[df["model"] == label].sort_values("threshold")
                t = sub["threshold"].values
                snb = sub["sNB"].values
                color = color_map.get(label, default_colors[i])
                lw = lw_map.get(label, 1.0)
                ax.plot(t, snb, color=color, lw=lw, label=model_names[i] if label == 'pred' else label)
                if confidence_intervals is not None and "sNB_lower" in sub.columns and label != "None":
                    if label == "pred":
                        ax.plot(t, sub["sNB_lower"].values, color=color, lw=0.5, linestyle="-")
                        ax.plot(t, sub["sNB_upper"].values, color=color, lw=0.5, linestyle="-")
                    else:
                        ax.plot(t, sub["sNB_lower"].values, color=color, lw=0.2, linestyle="-")
                        ax.plot(t, sub["sNB_upper"].values, color=color, lw=0.2, linestyle="-")
        else:
            labels = ['pred']
            for label in labels:
                sub = df[df["model"] == label].sort_values("threshold")
                t = sub["threshold"].values
                snb = sub["sNB"].values
                color = color_map.get(label, default_colors[i])
                lw = lw_map.get(label, 1.0)
                ax.plot(t, snb, color=color, lw=lw, label=model_names[i] if label == 'pred' else label, zorder=3)
                if confidence_intervals is not None and "sNB_lower" in sub.columns and label != "None":
                    ax.plot(t, sub["sNB_lower"].values, color=color, lw=1, linestyle="-")
                    ax.plot(t, sub["sNB_upper"].values, color=color, lw=1, linestyle="-")
    ax.set_xlim(thresholds[0] - 0.005, thresholds[-1] + 0.005)
    ax.set_ylim(-0.05, 1.0)

    x_step = round((thresholds[-1] - thresholds[0]) / 4, 2)
    x_ticks = np.arange(
        round(x_step, 2),
        thresholds[-1] + 1e-9,
        x_step,
    )
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_bounds(0, 1.0)

    ax.spines["bottom"].set_position(("outward", 10))
    ax.spines["bottom"].set_bounds(x_ticks[0], x_ticks[-1])
    ax.tick_params(axis="x", direction="out", length=5, pad=8,
                   bottom=True, top=False)
    ax.set_xlabel("High Risk Threshold", fontsize=11, labelpad=12)
    ax.set_ylabel("Standardized Net Benefit", fontsize=11)
    ax.legend(loc="upper right", frameon=True,
              framealpha=0.9, edgecolor="black", fontsize=9)

    if cost_benefit_axis:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())

        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")
        ax2.spines["bottom"].set_position(("outward", 70))
        ax2.spines[["top", "right", "left"]].set_visible(False)
        ax2.spines["bottom"].set_visible(True)
        ax2.tick_params(axis="x", direction="out", length=5, pad=4,
                        bottom=True, top=False)

        t_lo, t_hi = thresholds[0], thresholds[-1]

        candidate_pts = []
        candidate_labels = []

        if policy == "opt-in":
            candidate_pts.append(1 / 2)
            candidate_labels.append(f"1:1")
            for K in range(5, 500, 5):
                pt = 1.0 / (1.0 + K)
                if t_lo <= pt <= t_hi:
                    candidate_pts.append(pt)
                    candidate_labels.append(f"1:{K}")
            for K in range(5, 500, 5):
                pt = K / (1.0 + K)
                if t_lo <= pt <= t_hi:
                    candidate_pts.append(pt)
                    candidate_labels.append(f"{K}:1")
        else:
            candidate_pts.append(1 / 2)
            candidate_labels.append(f"1:1")
            for K in range(5, 500, 5):
                pt = K / (1.0 + K)
                if t_lo <= pt <= t_hi:
                    candidate_pts.append(pt)
                    candidate_labels.append(f"1:{K}")
            for K in range(5, 500, 5):
                pt = 1.0 / (1.0 + K)
                if t_lo <= pt <= t_hi:
                    candidate_pts.append(pt)
                    candidate_labels.append(f"{K}:1")

        if candidate_pts:
            order = np.argsort(candidate_pts)
            all_pts = np.array(candidate_pts)[order]
            all_labels = [candidate_labels[i] for i in order]
        else:
            all_pts = np.array([])
            all_labels = []

        if len(all_pts) <= 5:
            sel_pts = all_pts
            sel_labels = all_labels
        else:
            left_t, right_t = all_pts[-1], all_pts[0]
            ideal_mid = np.linspace(left_t, right_t, 5)[1:-1]
            mid_pts = all_pts[1:-1]
            mid_labels_pool = all_labels[1:-1]
            chosen_idx = []
            for target in ideal_mid:
                dists = np.abs(mid_pts - target)
                for ci in chosen_idx:
                    dists[ci] = np.inf
                chosen_idx.append(int(np.argmin(dists)))
            chosen_idx.sort()
            sel_pts = np.concatenate([
                [all_pts[-1]],
                mid_pts[chosen_idx],
                [all_pts[0]],
            ])
            sel_labels = (
                    [all_labels[-1]]
                    + [mid_labels_pool[ci] for ci in chosen_idx]
                    + [all_labels[0]]
            )

        ax2.set_xticks(sel_pts)
        ax2.set_xticklabels(sel_labels, fontsize=9)
        ax2.spines["bottom"].set_bounds(sel_pts.min(), sel_pts.max())
        ax2.set_xlabel("Cost:Benefit Ratio", fontsize=11, labelpad=10)

    plt.tight_layout()

    return fig, ax



if __name__ == '__main__':
    import numpy as np
    from sklearn.model_selection import train_test_split
    real = np.random.randn(100)
    pred = np.random.randn(100)
    correlation_graph_between_prediction_and_reality(real, pred)
    plt.show()