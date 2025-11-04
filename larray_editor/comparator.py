import numpy as np
import larray as la

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QListWidget, QSplitter, QHBoxLayout,
                            QLabel, QCheckBox, QLineEdit, QComboBox, QMessageBox)

from larray_editor.utils import _
from larray_editor.arraywidget import ArrayEditorWidget
from larray_editor.editor import AbstractEditorWindow, CAN_CONVERT_TO_LARRAY


class ComparatorWidget(QWidget):
    """Comparator Widget"""
    # FIXME: rtol, atol are unused, and align and fill_value are only partially used
    def __init__(self, parent=None, bg_gradient='red-white-blue', rtol=0, atol=0, nans_equal=True,
                 align='outer', fill_value=np.nan):
        QWidget.__init__(self, parent)

        layout = QVBoxLayout()
        # avoid margins around the widget
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # max diff label
        maxdiff_layout = QHBoxLayout()
        maxdiff_layout.addWidget(QLabel('maximum absolute relative difference:'))
        self.maxdiff_label = QLabel('')
        maxdiff_layout.addWidget(self.maxdiff_label)
        maxdiff_layout.addStretch()
        layout.addLayout(maxdiff_layout)

        #  arraywidget
        self.arraywidget = ArrayEditorWidget(self, data=None, readonly=True, bg_gradient=bg_gradient)
        layout.addWidget(self.arraywidget)

        self._combined_array = None
        self._array0 = None
        self._diff_below_tolerance = None
        self._bg_value = None
        self.stack_axis = None

        self.nans_equal = nans_equal
        self.align_method = align
        self.fill_value = fill_value

    # TODO: we might want to use self.align_method, etc instead of using arguments?
    def get_comparison_options_layout(self, align, atol, rtol):
        layout = QHBoxLayout()

        align_method_label = QLabel("Align:")
        # align_method_label.setToolTip(tooltip)
        layout.addWidget(align_method_label)
        align_method_combo = QComboBox()
        align_method_combo.addItems(["outer", "inner", "left", "right", "exact"])
        # align_combo.setToolTip(tooltip)
        align_method_combo.setCurrentText(align)
        align_method_combo.currentTextChanged.connect(self.update_align_method)
        self.align_method_combo = align_method_combo

        layout.addWidget(align_method_combo)

        tooltip = """Element i of two arrays are considered as equal if they satisfy the following equation:
        abs(array1[i] - array2[i]) <= (absolute_tol + relative_tol * abs(array2[i]))"""
        tolerance_label = QLabel("Tolerance:")
        tolerance_label.setToolTip(tooltip)
        # self.arraywidget.btn_layout.addWidget(tolerance_label)
        layout.addWidget(tolerance_label)
        tolerance_combobox = QComboBox()
        tolerance_combobox.addItems(["absolute", "relative"])
        tolerance_combobox.setToolTip(tooltip)
        tolerance_combobox.currentTextChanged.connect(self._update_from_combined_array)
        layout.addWidget(tolerance_combobox)
        self.tolerance_combobox = tolerance_combobox
        # We do not use a QDoubleValidator because, by default, it uses the
        # system locale (so we would need to parse the string using that
        # locale too) and does not provide any feedback to users on failure
        tolerance_line_edit = QLineEdit()
        tolerance_line_edit.setPlaceholderText("1e-8")
        tolerance_line_edit.setMaximumWidth(80)
        tolerance_line_edit.setToolTip("Press Enter to activate the new tolerance value")
        tolerance_line_edit.editingFinished.connect(self._update_from_combined_array)
        layout.addWidget(tolerance_line_edit)
        self.tolerance_line_edit = tolerance_line_edit
        if rtol > 0 and atol > 0:
            raise ValueError("Arguments 'rtol' and 'atol' cannot be used together.")
        if rtol > 0:
            tolerance_combobox.setCurrentText("relative")
            tolerance_line_edit.setText(str(rtol))
        if atol > 0:
            tolerance_combobox.setCurrentText("absolute")
            tolerance_line_edit.setText(str(atol))

        # show difference only
        diff_checkbox = QCheckBox(_('Differences Only'))
        diff_checkbox.stateChanged.connect(self._update_from_bg_value_and_diff_below_tol)
        self.diff_checkbox = diff_checkbox
        layout.addWidget(diff_checkbox)

        layout.addStretch()
        return layout

    def _get_atol_rtol(self):
        try:
            tol_str = self.tolerance_line_edit.text()
            tol = float(tol_str) if tol_str else 0
        except ValueError as e:
            # this is necessary to avoid having the error message twice, because we
            # first show it here, which makes the tolerance_line_edit lose focus,
            # which triggers its editingFinished signal, which calls update_isequal,
            # which ends up here again if tol_str did not change in-between.
            self.tolerance_line_edit.setText('')
            tol = 0
            QMessageBox.critical(self, "Error", str(e))
        is_absolute = self.tolerance_combobox.currentText() == "absolute"
        return (tol, 0) if is_absolute else (0, tol)

    # override keyPressEvent to prevent pressing Enter after changing the tolerance value
    # in associated QLineEdit to close the parent dialog box
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
            return
        QWidget.keyPressEvent(self, event)

    def set_data(self, arrays, stack_axis):
        """
        Parameters
        ----------
        arrays: list or tuple of scalar, Array, ndarray
            Arrays to compare.
        stack_axis: Axis
            Names of arrays.
        """
        assert all(np.isscalar(a) or isinstance(a, la.Array) for a in arrays)
        self.stack_axis = stack_axis
        self.arrays = arrays
        self._update_from_arrays()

    def update_align_method(self, align):
        self.align_method = align
        self._update_from_arrays()

    def _update_from_arrays(self):
        # TODO: implement align in stack instead
        stack_axis = self.stack_axis
        try:
            aligned_arrays = align_all(self.arrays,
                                       join=self.align_method,
                                       fill_value=self.fill_value)
            self._combined_array = la.stack(aligned_arrays, stack_axis)
            self._array0 = self._combined_array[stack_axis.i[0]]
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self._combined_array = la.Array([''])
            self._array0 = self._combined_array
        self._update_from_combined_array()

    def _update_from_combined_array(self):
        if self._combined_array is None:
            return

        atol, rtol = self._get_atol_rtol()
        try:
            # eq does not take atol and rtol into account
            eq = self._combined_array.eq(self._array0,
                                         nans_equal=self.nans_equal)
            isclose = self._combined_array.eq(self._array0,
                                              rtol=rtol, atol=atol,
                                              nans_equal=self.nans_equal)
        except TypeError:
            # object arrays
            eq = self._combined_array == self._array0
            isclose = eq
        self._diff_below_tolerance = isclose

        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                diff = self._combined_array - self._array0
                reldiff = diff / self._array0
            # make reldiff 0 where the values are the same than array0 even for
            # special values (0, nan, inf, -inf)
            # at this point reldiff can still contain nan and infs
            reldiff = la.where(eq, 0, reldiff)

            # 1) compute maxabsreldiff for the label
            #    this should NOT exclude nans or infs
            relmin = reldiff.min(skipna=False)
            relmax = reldiff.max(skipna=False)
            maxabsreldiff = max(abs(relmin), abs(relmax))

            # 2) compute bg_value
            # replace -inf by min(reldiff), +inf by max(reldiff)
            reldiff_for_bg = reldiff.copy()
            isneginf = reldiff == -np.inf
            isposinf = reldiff == np.inf
            isinf = isneginf | isposinf

            # given the way reldiff is constructed, it cannot contain only infs
            # (because inf/inf is nan) it can contain only infs and nans though,
            # in which case finite_relXXX will be nan, so unless the array
            # is empty, finite_relXXX should never be inf
            finite_relmin = np.nanmin(reldiff, where=~isinf, initial=np.inf)
            finite_relmax = np.nanmax(reldiff, where=~isinf, initial=-np.inf)
            # special case when reldiff contains only 0 and infs (to avoid
            # coloring the inf cells white in that case)
            if finite_relmin == 0 and finite_relmax == 0 and isinf.any():
                finite_relmin = -1
                finite_relmax = 1
            reldiff_for_bg[isneginf] = finite_relmin
            reldiff_for_bg[isposinf] = finite_relmax

            # make sure that "acceptable" differences show as white
            reldiff_for_bg = la.where(isclose, 0, reldiff_for_bg)

            # We need a separate version for bg and the label, so that when we
            # modify atol/rtol, the background color is updated but not the
            # maxreldiff label
            maxabsreldiff_for_bg = max(abs(np.nanmin(reldiff_for_bg)),
                                       abs(np.nanmax(reldiff_for_bg)))
            if maxabsreldiff_for_bg:
                # scale reldiff to range 0-1 with 0.5 for reldiff = 0
                self._bg_value = (reldiff_for_bg / maxabsreldiff_for_bg) / 2 + 0.5
            # if the only differences are nans on either side
            elif not isclose.all():
                # use white (0.5) everywhere except where reldiff is nan, so
                # that nans are grey
                self._bg_value = reldiff_for_bg + 0.5
            else:
                # do NOT use full_like as we don't want to inherit array dtype
                self._bg_value = la.full(self._combined_array.axes, 0.5)
        except TypeError:
            # str/object array
            maxabsreldiff = la.nan
            # do NOT use full_like as we don't want to inherit array dtype
            self._bg_value = la.full(self._combined_array.axes, 0.5)

        # using percents does not look good when the numbers are very small
        self.maxdiff_label.setText(str(maxabsreldiff))
        color = 'red' if maxabsreldiff != 0.0 else 'black'
        self.maxdiff_label.setStyleSheet(f"QLabel {{ color: {color}; }}")
        self._update_from_bg_value_and_diff_below_tol(self.diff_checkbox.isChecked())

    def _update_from_bg_value_and_diff_below_tol(self, diff_only):
        """
        Parameters
        ----------
        diff_only: bool
            Whether or not to show only differences.
        """
        array = self._combined_array
        bg_value = self._bg_value
        if diff_only and self._diff_below_tolerance.ndim > 0:
            row_filter = (~self._diff_below_tolerance).any(self.stack_axis)
            array = array[row_filter]
            bg_value = bg_value[row_filter]
        self.arraywidget.set_data(array, attributes={'bg_value': bg_value})


def align_all(arrays, join='outer', fill_value=la.nan):
    return arrays
    if len(arrays) > 2:
        raise NotImplementedError("aligning more than two arrays is not yet implemented")
    first_array = arrays[0]
    def is_raw(array):
        return all(axis.iswildcard and axis.name is None
                   for axis in array.axes)
    if all(is_raw(array) and array.shape == first_array.shape for array in arrays[1:]):
        return arrays
    return first_array.align(arrays[1], join=join, fill_value=fill_value)


class ArrayComparatorWindow(AbstractEditorWindow):
    """Array Comparator Dialog"""

    name = "Array Comparator"
    editable = False
    file_menu = False
    help_menu = True

    def __init__(self, data, title='', caller_info=None, parent=None,
                 bg_gradient='red-white-blue', rtol=0, atol=0, nans_equal=True,
                 align='outer', fill_value=np.nan, names=None):
        """
        Setup ArrayComparator.

        Parameters
        ----------
        data: list or tuple of Array or ndarray
            Arrays to compare.
        title: str
            Title.
        readonly: bool
            Ignored argument (comparator is always read only)
        rtol: int or float
        atol: int or float
        nans_equal: bool
        bg_gradient: str
        names: list of str
        align: str
        fill_value: Scalar
        """
        AbstractEditorWindow.__init__(self, title=title, readonly=True,
                                      caller_info=caller_info, parent=parent)
        self.setup_menu_bar()

        widget = self.centralWidget()
        arrays = [la.asarray(array) for array in data
                  if isinstance(array, CAN_CONVERT_TO_LARRAY)]
        if names is None:
            names = [f"Array{i}" for i in range(len(arrays))]

        layout = QVBoxLayout()
        widget.setLayout(layout)

        comparator_widget = ComparatorWidget(self, bg_gradient=bg_gradient,
                                             rtol=rtol, atol=atol,
                                             nans_equal=nans_equal,
                                             align=align,
                                             fill_value=fill_value)
        comparison_options_layout = (
            comparator_widget.get_comparison_options_layout(align=align,
                                                            atol=atol,
                                                            rtol=rtol))
        self.comparator_widget = comparator_widget

        layout.addLayout(comparison_options_layout)

        layout.addWidget(comparator_widget)
        comparator_widget.set_data(arrays, la.Axis(names, 'array'))
        self.set_window_size_and_geometry()


class SessionComparatorWindow(AbstractEditorWindow):
    """Session Comparator Dialog"""

    name = "Session Comparator"
    editable = False
    file_menu = False
    help_menu = True

    # except for 'names', kwargs are passed as-is to the ComparatorWidget
    def __init__(self, data, title='', caller_info=None, parent=None,
                 bg_gradient='red-white-blue', rtol=0, atol=0, nans_equal=True,
                 align='outer', fill_value=np.nan, names=None):
        """
        Setup SessionComparator.

        Parameters
        ----------
        data: list or tuple of Session
            Sessions to compare.
        title: str
            Title.
        readonly: bool
            Ignored argument (comparator is always read only)
        rtol: int or float
        atol: int or float
        nans_equal: bool
        bg_gradient: str
        names: list of str
        align: str
        fill_value: Scalar
        """
        AbstractEditorWindow.__init__(self, title=title, readonly=True,
                                      caller_info=caller_info, parent=parent)

        self.setup_menu_bar()

        widget = self.centralWidget()
        sessions = data
        if names is None:
            names = [f"Session{i}" for i in range(len(sessions))]

        assert all(isinstance(s, la.Session) for s in sessions)
        self.sessions = sessions
        self.stack_axis = la.Axis(names, 'session')

        main_layout = QVBoxLayout()
        widget.setLayout(main_layout)

        # TODO: these two fields are unused
        self.atol = atol
        self.rtol = rtol

        array_names = sorted(set.union(*[set(s.filter(kind=CAN_CONVERT_TO_LARRAY).names) for s in self.sessions]))
        self.array_names = array_names
        listwidget = QListWidget(self)
        listwidget.addItems(array_names)
        listwidget.currentItemChanged.connect(self.on_item_changed)
        self.listwidget = listwidget

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.listwidget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        comparator_widget = ComparatorWidget(self, bg_gradient=bg_gradient,
                                             rtol=rtol, atol=atol,
                                             nans_equal=nans_equal,
                                             align=align,
                                             fill_value=fill_value)
        # do not call set_data on the comparator_widget as it will be done by the setCurrentRow below
        self.comparator_widget = comparator_widget
        # FIXME:
        #  this is kinda convoluted. I am unsure the tolerance layout should be created
        #  by the comparatorWidget (but it initialize self.tolerance_combo and lineedit
        #  which are used by the widget, so extracting it entirely is probably a bit more
        #  work. I guess the widget could work with only rtol and atol fields and
        #  the
        comparison_options_layout = (
            comparator_widget.get_comparison_options_layout(align=align,
                                                            atol=atol,
                                                            rtol=rtol))
        comparator_widget.align_method_combo.currentTextChanged.connect(self.update_listwidget_colors)
        comparator_widget.tolerance_combobox.currentTextChanged.connect(self.update_listwidget_colors)
        comparator_widget.tolerance_line_edit.editingFinished.connect(self.update_listwidget_colors)

        self.update_listwidget_colors()

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(comparator_widget)
        main_splitter.setSizes([5, 95])
        main_splitter.setCollapsible(1, False)
        self.widgets_to_save_to_settings['main_splitter'] = main_splitter

        main_layout.addLayout(comparison_options_layout)
        main_layout.addWidget(main_splitter)
        self.listwidget.setCurrentRow(0)
        self.set_window_size_and_geometry()

    def update_listwidget_colors(self):
        atol, rtol = self.comparator_widget._get_atol_rtol()
        listwidget = self.listwidget
        # TODO: this functionality is super useful but can also be super slow when
        #       the sessions contain large arrays. It would be great if we
        #       could do this asynchronously
        for i, name in enumerate(self.array_names):
            align_method = self.comparator_widget.align_method
            fill_value = self.comparator_widget.fill_value
            arrays = self.get_arrays(name)
            try:
                aligned_arrays = align_all(arrays, join=align_method, fill_value=fill_value)
                first_array = aligned_arrays[0]
                all_equal = all(a.equals(first_array, rtol=rtol, atol=atol, nans_equal=True)
                                for a in aligned_arrays[1:])
            except Exception:
                all_equal = False
            item = listwidget.item(i)
            item.setForeground(Qt.black if all_equal else Qt.red)

    def get_arrays(self, name):
        return [la.asarray(s.get(name, la.nan)) for s in self.sessions]

    def on_item_changed(self, curr, prev):
        arrays = self.get_arrays(str(curr.text()))
        self.comparator_widget.set_data(arrays, self.stack_axis)

    def closeEvent(self, event):
        self.save_widgets_state_and_geometry()
        AbstractEditorWindow.closeEvent(self, event)
