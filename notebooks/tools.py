import numpy as np
from matplotlib import pyplot
import six
from bokeh.io import export_png, export_svgs
from bokeh.models import ColumnDataSource, DataTable, TableColumn


def table(data, col_width = 3.0, row_height = 0.625, font_size = 14,
          header_color = '#40466e', row_colors = ['#f1f1f2', 'w'], edge_color = 'w',
          bbox = [0, 0, 1, 1], header_columns = 0,
          ax = None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = pyplot.subplots(figsize = size)
        ax.axis('off')

    mpl_table = ax.table(cellText = data.values, bbox = bbox, colLabels = data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight = 'bold', color = 'w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return ax


def saveimage(df, path):
    src = ColumnDataSource(df)
    df_columns = [df.index.name]
    df_columns.extend(df.columns.values)
    columns_for_table = []
    for column in df_columns:
        columns_for_table.append(TableColumn(field = column, title = column))

    data_table = DataTable(source = src, columns = columns_for_table, height_policy = "auto", width_policy = "auto",
                           index_position = None)
    export_png(data_table, filename = path)
