import sys
import time
import numpy as np
from matplotlib.backends.qt_compat import QtCore, QtWidgets


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QVBoxLayout(self._main)
        static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(static_canvas)
        self.addToolBar(NavigationToolbar(static_canvas, self))
        dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(dynamic_canvas)
        self.addToolBar(QtCore.Qt.BottomToolBarArea,
                        NavigationToolbar(dynamic_canvas, self))
        self._static_ax = static_canvas.figure.subplots()
        t = np.linspace(0, 10, 501)
        self._static_ax.plot(t, np.tan(t), ".")
        self.x = []  # 建立空的x轴数组和y轴数组
        self.y = []
        self.n = 0
        self._dynamic_ax = dynamic_canvas.figure.subplots()
        self._timer = dynamic_canvas.new_timer(
            100, [(self._update_canvas, (), {})])
        self._timer.start()

    def _update_canvas(self):
        self.n += 1
        if self.n == 200:  # 画200个点就停止，根据实际情况确定终止条件
            self._timer.stop()
        self._dynamic_ax.clear()
        self.x.append(np.pi / 100 * self.n)  # x加入一个值，后一个值比前一个大pi/100
        xx = np.array(self.x)
        # t = np.linspace(0, 10, 101)
        # Shift the sinusoid as a function of time.
        self._dynamic_ax.plot(xx, np.sin(xx))
        self._dynamic_ax.set_xlim(0, 7)
        self._dynamic_ax.set_ylim(-1, 1)
        self._dynamic_ax.figure.canvas.draw()


if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow()
    app.show()
    qapp.exec_()
