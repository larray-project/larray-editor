import re

from qtpy.QtCore import Qt, QStringListModel
from qtpy.QtGui import QTextCursor
from qtpy.QtWidgets import QTextEdit, QCompleter

from larray_editor.utils import _, logger

MAX_SQL_QUERIES = 1000
SQL_CREATE_TABLE_PATTERN = re.compile(r'CREATE\s+TABLE\s+([\w_]+)\s+',
                                      flags=re.IGNORECASE)
SQL_DROP_TABLE_PATTERN = re.compile(r'DROP\s+TABLE\s+([\w_]+)',
                                    flags=re.IGNORECASE)


class SQLWidget(QTextEdit):
    SQL_KEYWORDS = [
        "SELECT", "FROM", "WHERE", "INSERT", "UPDATE", "DELETE",
        "JOIN", "LEFT", "RIGHT", "INNER", "OUTER",
        "GROUP", "BY", "ORDER", "HAVING",
        "AS", "ON", "IN", "AND", "OR", "NOT", "NULL", "IS",
        "DISTINCT", "LIMIT", "OFFSET", "UNION", "ALL",
        "CREATE", "TABLE", "DROP", "ALTER", "ADD", "INDEX", "PRIMARY",
        "KEY", "FOREIGN", "VALUES", "SET",
        "CASE", "WHEN", "THEN", "ELSE", "END"
    ]
    SQL_KEYWORDS_SET = set(SQL_KEYWORDS)

    def __init__(self, editor_window):
        import polars as pl

        # avoid a circular module dependency by having the import here
        from larray_editor.editor import MappingEditorWindow
        assert isinstance(editor_window, MappingEditorWindow)
        super().__init__()
        self.editor_window = editor_window

        msg = _("""Enter an SQL query here and press SHIFT+ENTER to execute it. 

Use the UP/DOWN arrow keys to navigate through queries you typed previously \
(including during previous sessions). 
It will only display past queries which start with the text already typed so \
far (the part before the cursor) so that one can more easily search for \
specific queries.

SQL keywords, names of variables usable as table and column names (once the \
FROM clause is known) can be autocompleted by using TAB.

The currently displayed table may be called 'self' (in addition to its real \
name).
""")
        self.setPlaceholderText(msg)
        self.setAcceptRichText(False)
        font = self.font()
        font.setFamily('Calibri')
        font.setPointSize(11)
        self.setFont(font)

        self.history = []
        self.history_index = 0

        self.completer = QCompleter([], self)
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setWidget(self)
        self.completer.activated.connect(self.insert_completion)
        self.sql_context = pl.SQLContext(eager=False)
        self.update_completer_options({})

    def update_completer_options(self, data=None, selected=None):
        if data is not None:
            data = self._filter_data_for_sql(data)
            if selected is not None and self._handled_by_polars_sql(selected):
                data['self'] = selected
            self.data = data
            self.sql_context.register_many(data)
        else:
            data = self.data

        if 'self' in data:
            table_names_to_fetch_columns = ['self']
        else:
            table_names_to_fetch_columns = []

        table_names = [k for k, v in data.items()]

        # extract table names from the current FROM clause
        query_text = self.toPlainText()
        m = re.search(r'\s+FROM\s+(\S+)', query_text, re.IGNORECASE)
        if m:
            after_from = m.group(1)
            # try any identifier found in the query after the FROM keyword
            # there will probably be false positives if a column has the same
            # name as another table but that should be rare
            from_tables = [word for word in after_from.split()
                           if word not in self.SQL_KEYWORDS_SET and word in data]
            if from_tables:
                table_names_to_fetch_columns = from_tables

        # add column names from all the used tables or self, if present
        col_names_set = set()
        for table_name in table_names_to_fetch_columns:
            col_names_set.update(set(data[table_name].collect_schema().names()))
        col_names = sorted(col_names_set)

        logger.debug(f"available columns for SQL queries: {col_names}")
        logger.debug(f"available tables for SQL queries: {table_names}")
        completions = col_names + table_names + self.SQL_KEYWORDS
        model = QStringListModel(completions, self.completer)
        self.completer.setModel(model)

    def _filter_data_for_sql(self, data):
        return {k: v for k, v in data.items()
                if self._handled_by_polars_sql(v)}

    def _handled_by_polars_sql(self, obj):
        import polars as pl
        SUPPORTED_TYPES = (pl.DataFrame, pl.LazyFrame, pl.Series)
        # We purposefully do not support pandas and pyarrow objects here, even if
        # polars SQL can sort of handle them, because Polars does that by
        # converting the object to their Polars counterpart first and that
        # can be slow (e.g. >1s for pd_df_big)

        # if 'pandas' in sys.modules:
        #     import pandas as pd
        #     SUPPORTED_TYPES += (pd.DataFrame, pd.Series)
        # if 'pyarrow' in sys.modules:
        #     import pyarrow as pa
        #     SUPPORTED_TYPES += (pa.Table, pa.RecordBatch)
        return isinstance(obj, SUPPORTED_TYPES)

    def insert_completion(self, completion):
        cursor = self.textCursor()
        cursor.select(QTextCursor.WordUnderCursor)
        cursor.removeSelectedText()
        # Insert a space if the cursor is at the end of the text
        at_end = cursor.position() == len(self.toPlainText())
        cursor.insertText(completion + (' ' if at_end else ''))
        self.setTextCursor(cursor)
        self.update_completer_options()

    def keyPressEvent(self, event):
        completer_popup = self.completer.popup()
        if completer_popup.isVisible():
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Tab):
                # Insert the currently highlighted completion
                current_index = completer_popup.currentIndex()
                if not current_index.isValid():
                    # Default to the first item if none is highlighted
                    current_index = completer_popup.model().index(0, 0)
                completion = current_index.data()
                self.insert_completion(completion)
                completer_popup.hide()
                return
            elif event.key() == Qt.Key.Key_Escape:
                completer_popup.hide()
                return

        if (event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return) and
                event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            query_text = self.toPlainText().strip()
            if query_text:
                self.append_to_history(query_text)
            self.execute_sql(query_text)
            return
        elif event.key() == Qt.Key.Key_Tab:
            prefix = self.get_word_prefix()
            self.completer.setCompletionPrefix(prefix)
            if self.completer.completionCount() == 1:
                completion = self.completer.currentCompletion()
                self.insert_completion(completion)
                return
            else:
                self.show_autocomplete_popup()
                return

        cursor = self.textCursor()
        # for plaintext QTextEdit, blockNumber gives the line number
        line_num = cursor.blockNumber()
        if event.key() == Qt.Key.Key_Up:
            if line_num == 0:
                if self.search_and_recall_history(direction=-1):
                    return
        elif event.key() == Qt.Key.Key_Down:
            total_lines = self.document().blockCount()
            if line_num == total_lines - 1:
                if self.history_index < len(self.history) - 1:
                    if self.search_and_recall_history(direction=1):
                        return
                else:
                    self.history_index = len(self.history)
                    self.clear()
                return
        super().keyPressEvent(event)
        self.update_completer_options()
        # we need to compute the prefix *after* the keypress event has been
        # handled so that the prefix contains the last keystroke
        prefix = self.get_word_prefix()
        if prefix:
            self.completer.setCompletionPrefix(prefix)
            num_completion = self.completer.completionCount()
            # we must show the popup even if it is already visible, because
            # the number of completions might have changed and the popup size
            # needs to be updated
            if 0 < num_completion <= self.completer.maxVisibleItems():
                self.show_autocomplete_popup()
            elif num_completion == 0 and completer_popup.isVisible():
                completer_popup.hide()

    def show_autocomplete_popup(self):
        # create a new cursor and move it to the start of the word, so that
        # we can position the popup correctly
        word_start_cursor = self.textCursor()
        word_start_cursor.movePosition(QTextCursor.StartOfWord)
        rect = self.cursorRect(word_start_cursor)
        completer_popup = self.completer.popup()
        popup_scrollbar = completer_popup.verticalScrollBar()
        popup_scrollbar_width = popup_scrollbar.sizeHint().width() + 10
        rect.setWidth(completer_popup.sizeHintForColumn(0)
                      + popup_scrollbar_width)
        self.completer.complete(rect)

    def get_word_prefix(self):
        text = self.toPlainText()
        if not text:
            return ''
        cursor = self.textCursor()
        cursor_pos = cursor.position()
        # <= len(text) (instead of <) because cursor can be at the end
        assert 0 <= cursor_pos <= len(text), f"{cursor_pos=} {len(text)=}"
        word_start = cursor_pos
        while (word_start > 0 and
               text[word_start - 1].isalnum() or text[word_start - 1] == '_'):
            word_start -= 1
        return text[word_start:cursor_pos]

    def search_and_recall_history(self, direction: int):
        if not self.history:
            return False
        cursor = self.textCursor()
        query_text = self.toPlainText()
        cursor_pos = cursor.position()
        prefix = query_text[:cursor_pos]
        index = self.history_index + direction
        while 0 <= index <= len(self.history) - 1:
            if self.history[index].startswith(prefix):
                self.history_index = index
                self.setPlainText(self.history[self.history_index])
                self.update_completer_options()
                cursor.setPosition(cursor_pos)
                self.setTextCursor(cursor)
                return True
            index += direction
        # no matching prefix found, do not change history_index
        return False

    def append_to_history(self, sql_text):
        history = self.history
        if not history or history[-1] != sql_text:
            history.append(sql_text)
        if len(history) > MAX_SQL_QUERIES:
            # keep the last N entries
            history = history[-MAX_SQL_QUERIES:]
        self.history_index = len(history)

    def _fetch_table(self, table_name):
        return self.sql_context.execute(f"SELECT * FROM {table_name}",
                                        eager=False)

    def execute_sql(self, sql_text: str):
        """Execute SQL query and display result"""
        editor_window = self.editor_window
        sql_context = self.sql_context
        logger.debug(f"Executing SQL query:\n{sql_text}")
        result = sql_context.execute(sql_text, eager=False)

        # To determine whether we have added or dropped tables, comparing the
        # resulting SQL context to what we had before would be more reliable
        # than this regex-based solution but is not currently possible using
        # Polars public API
        new_table_name = None
        m = SQL_CREATE_TABLE_PATTERN.match(sql_text)
        if m is not None:
            new_table_name = m.group(1)
        dropped_table_name = None
        m = SQL_DROP_TABLE_PATTERN.match(sql_text)
        if m is not None:
            dropped_table_name = m.group(1)
        if new_table_name or dropped_table_name:
            # data might be a Session, make sure we have a dict copy
            new_data = dict(editor_window.data.items())
            if new_table_name:
                new_data[new_table_name] = self._fetch_table(new_table_name)
                logger.debug(f'added table {new_table_name} to session')
            if dropped_table_name:
                del new_data[dropped_table_name]
                logger.debug(f'dropped table {dropped_table_name} from session')
            editor_window.update_mapping_and_varlist(new_data)
        if new_table_name:
            editor_window.select_list_item(new_table_name)
        elif not dropped_table_name:
            editor_window.arraywidget.set_data(result)

    def save_to_settings(self, settings):
        settings.setValue('queries', self.history)

    def load_from_settings(self, settings):
        self.history = settings.value('queries', [], type=list)
        self.history_index = len(self.history)
