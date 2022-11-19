from __future__ import annotations

from csv import writer, reader
from matplotlib import pyplot
import numpy as np
import six


def invert(array: list) -> list:
    val = list()
    for i in range(len(array[0])):
        aux = list()
        for j in range(len(array)):
            aux.append(array[j][i])
        val.append(aux)
    return val


class LeastSquareError(Exception):
    def __init__(self, *args):
        super(LeastSquareError, self).__init__(*args)


class LeastSquares:
    """
    :autor Edgar Alejo Ramirez
    """

    def __init__(self, data: dict | str):
        self.__array: dict
        self.__path: str

        if isinstance(data, dict):
            self.__array = data
        elif isinstance(data, str):
            self.__path = data
            self.__array = {key: value for key, value in self.__readcsv()}
        else:
            raise LeastSquareError(f'Required data types: {dict} or path of *.csv file {str}')

    def getarray(self):
        return self.__array

    def __readcsv(self):
        with open(self.__path, newline = '') as file:
            values = invert([i for i in reader(file)])
            for val in values:
                yield str(val[0]), [float(v) for v in val[1:]]

    def __writecsv(self):
        with open(self.__path, mode = 'w') as File:
            write = writer(File)
            write.writerow(self.__array)

    def m(self) -> float:
        pass
        # n = len(df)
        # sum_xy = np.sum(df["(tan α) * (I[A])"])
        # sum_x = np.sum(df["tan α"])
        # sum_y = np.sum(df["I[A]"])
        # sum_x2 = np.sum(df["(tan α)^2"])
        # sum_y2 = np.sum(df["(I[A])^2"])
        # return (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)

    def b(self) -> float:
        pass

    def table(
        self,
        decimals: int = 3,
        colwidth: int | float = 3.0,
        rowheight: int | float = 0.625,
        fontsize: int | float = 14,
        headercolor: str = '#40466e',
        rowcolors: list[str] = None,
        edgecolor: str = 'w',
        bbox: list[int] = None,
        headercolumns: int = 0,
        ax = None,
        **kwargs
    ):
        if bbox is None:
            bbox = [0, 0, 1, 1]
        if rowcolors is None:
            rowcolors = ['#f1f1f2', 'w']
        if ax is None:
            size = (np.array(self.size()) + np.array([0, 1])) * np.array([colwidth, rowheight])
            fig, ax = pyplot.subplots(figsize = size)
            ax.axis('off')

        val = np.round(self.__array.values, decimals = decimals)
        mtable = ax.table(cellText = val, bbox = bbox, colLabels = self.col(), **kwargs)

        mtable.auto_set_font_size(False)
        mtable.set_fontsize(fontsize)

        for k, cell in six.iteritems(mtable.get_celld()):
            cell.set_edgecolor(edgecolor)
            if k[0] == 0 or k[1] < headercolumns:
                cell.set_text_props(weight = 'bold', color = 'w')
                cell.set_facecolor(headercolor)
            else:
                cell.set_facecolor(rowcolors[k[0] % len(rowcolors)])
        pyplot.show()

    def graph(self) -> None:
        pass

    def size(self) -> tuple:
        row, col = 0, 0
        for v in self.__array.values():
            row = len(v)
        return row, len(self.__array)

    def col(self):
        return [k for k in self.__array.keys()]


class ErrorList(Exception):
    pass


class List(list):
    def __init__(self):
        super().__init__()

    def __add__(self, other: int | float | List):
        if isinstance(other, (int, float)):
            return [i + other for i in self]
        elif isinstance(other, List):
            if len(self) == len(other):
                return [self[i] + other[i] for i in range(len(self))]
            raise ErrorList(f"lists must be of equal size")
        else:
            raise ErrorList(f"only types are supported: {int}, {float} and {List}")

    def __sub__(self, other: int | float | List):
        if isinstance(other, (int, float)):
            return [i - other for i in self]
        elif isinstance(other, List):
            if len(self) == len(other):
                return [self[i] - other[i] for i in range(len(self))]
            raise ErrorList(f"lists must be of equal size")
        else:
            raise ErrorList(f"only types are supported: {int}, {float} and {List}")

    def __mul__(self, other: int | float | List):
        if isinstance(other, (int, float)):
            return [i * other for i in self]
        elif isinstance(other, List):
            if len(self) == len(other):
                return [self[i] * other[i] for i in range(len(self))]
            raise ErrorList(f"lists must be of equal size")
        else:
            raise ErrorList(f"only types are supported: {int}, {float} and {List}")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return [i / other for i in self]
        elif isinstance(other, List):
            if len(self) == len(other):
                return [self[i] / other[i] for i in range(len(self))]
            raise ErrorList(f"lists must be of equal size")
        else:
            raise ErrorList(f"only types are supported: {int}, {float} and {List}")

    def __pow__(self, power, modulo = None):
        if isinstance(power, (int, float)):
            return [i ** power for i in self]
        elif isinstance(power, List):
            if len(self) == len(power):
                return [self[i] ** power[i] for i in range(len(self))]
            raise ErrorList(f"lists must be of equal size")
        else:
            raise ErrorList(f"only types are supported: {int}, {float} and {List}")


if __name__ == '__main__':
    lt = List()
    lt.append(4)
    lt.append(7)
    lt.append(2)

    LIST = List()
    LIST.append(10)
    LIST.append(20)
    LIST.append(30)

    print(lt)
    a = "abb"
    LT = LIST ** lt
    print(LT)
