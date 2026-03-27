import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from typing import Union, List, Optional
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from calculate.base import floor_significant_digits
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as patches
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes


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
    >>> from vizforge.draw import train_test_lift
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
    >>> from vizforge.draw import triangular_heatmap
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
    *true_score_pairs,
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
    *true_score_pairs : sequence of array-like
        Each argument is a pair (y_true, y_score). Multiple pairs can be
        passed to compare ROC curves across models.

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
    >>> from vizforge.draw import enlarged_roc_curve
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

    # 主图标签与标题
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

def enlarged_pr_curve(*true_score_pairs: List[List] | np.ndarray,
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
                       legend_kwargs:dict=None) -> tuple[Axes, Figure]:
    """
    Plot PR curves with optional local zoom-in functionality.

    Convenience function to draw PR curves for one or multiple models,
    compute AUC scores, and add an inset axes to magnify a region of interest
    in the PR space (typically high Recall, high Precision).

    Parameters
    ----------
    *true_score_pairs : sequence of array-like
        Each argument is a pair (y_true, y_score). Multiple pairs can be
        passed to compare PR curves across models.

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
    # ax.yaxis.tick_right()
    # ax.yaxis.set_label_position("right")

    # 主图标签与标题
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

        # axins.yaxis.tick_right()
        if enlarged_frame_xticks is not None:
            axins.set_xticks(enlarged_frame_xticks)
        if enlarged_frame_yticks is not None:
            axins.set_yticks(enlarged_frame_yticks)
        axins.grid(False)

        ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1.5)

    plt.tight_layout()
    return fig, ax


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from scipy import stats

    n_samples, n_vars = 200, 20
    data = np.random.randn(n_samples, n_vars)
    cols = [f"Var{i + 1}" for i in range(n_vars)]
    df = pd.DataFrame(data, columns=cols)
    n = n_vars
    corr = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = stats.spearmanr(df.iloc[:, i], df.iloc[:, j])
            corr[i, j] = r
            corr[j, i] = r
    corr_df = pd.DataFrame(corr, index=cols, columns=cols)
    ax = triangular_heatmap(
        corr_df,
        annot=True,
        annot_kws={'size': 7.2},
        linewidths=0.5,
        linecolor='white',
        ticks_size=8,
        vmax=1,
        vmin=-1,
    )
    plt.show()