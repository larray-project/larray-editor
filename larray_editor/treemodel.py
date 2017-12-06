from qtpy.QtCore import Qt, QModelIndex, QAbstractItemModel


class SimpleTreeNode(object):
    __slots__ = ['parent', 'data', 'children']

    def __init__(self, parent, data):
        self.parent = parent
        self.data = data
        self.children = []


class SimpleLazyTreeModel(QAbstractItemModel):
    def __init__(self, root, parent=None):
        super(SimpleLazyTreeModel, self).__init__(parent)
        assert isinstance(root, SimpleTreeNode)
        self.root = root

    def columnCount(self, index):
        node = index.internalPointer() if index.isValid() else self.root
        return len(node.data)

    def data(self, index, role):
        if not index.isValid():
            return None

        if role != Qt.DisplayRole:
            return None

        node = index.internalPointer()
        return node.data[index.column()]

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags

        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.root.data[section]
        return None

    def index(self, row, column, parent_index):
        if not self.hasIndex(row, column, parent_index):
            return QModelIndex()

        parent_node = parent_index.internalPointer() if parent_index.isValid() else self.root
        child_node = parent_node.children[row]
        return self.createIndex(row, column, child_node)

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()

        child_node = index.internalPointer()
        parent_node = child_node.parent

        if parent_node == self.root:
            return QModelIndex()

        grand_parent = parent_node.parent
        parent_row = grand_parent.children.index(parent_node)
        return self.createIndex(parent_row, 0, parent_node)

    def rowCount(self, index):
        if index.column() > 0:
            return 0

        node = index.internalPointer() if index.isValid() else self.root
        return len(node.children)
