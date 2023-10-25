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


## Extra functions for AMECO tree
class Ameco_SimpleTreeNode:
    def __init__(self, data, parent=None):
        self.data = data
        self.parent = parent
        self.children = []
        if parent:
            parent.children.append(self)

def Ameco_parse_tree_structure(lines):
    root = Ameco_SimpleTreeNode("Root")
    prev_node = root
    prev_level = -1
    
    for line in lines:
        level = line.count('>')
        name = line.strip('>').strip()
        node = Ameco_SimpleTreeNode(name)
        
        if level > prev_level:
            node.parent = prev_node
            prev_node.children.append(node)
        elif level == prev_level:
            node.parent = prev_node.parent
            prev_node.parent.children.append(node)
        else:
            diff = prev_level - level
            higher_node = prev_node.parent
            for _ in range(diff):
                higher_node = higher_node.parent
            node.parent = higher_node
            higher_node.children.append(node)
        
        prev_node = node
        prev_level = level
    
    return root