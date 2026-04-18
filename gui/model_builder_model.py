from PyQt6.QtGui import QStandardItemModel, QStandardItem, QIcon, QBrush, QColor
from PyQt6.QtCore import Qt, pyqtSignal

class NodeType:
    """Enum-like constants for Node Types"""
    PROJECT = "Project"
    GROUP = "Group"
    SURROGATE_MODELER = "Surrogate Modeler"
    CLIMATE_GUARDIAN = "Climate Guardian"
    ASSET_MANAGER = "Asset Manager"
    OPTIMIZER = "Optimizer"
    RUN = "Run"
    # Generic leaf nodes
    PARAMETER = "Parameter"
    VARIABLE = "Variable"
    RESULT_TABLE = "Result Table"
    
    # --- New Backend Integrations ---
    FFD_MORPHER = "FFD Hull Morpher"
    OMNIVERSE = "Omniverse"
    POINTNET = "PointNet Agent"
    
class SmartNodeItem(QStandardItem):
    """
    Custom Node Item that holds data about its type and associated logic.
    """
    def __init__(self, name, node_type):
        super().__init__(name)
        self.node_type = node_type
        # Store node type in UserRole for easy access
        self.setData(node_type, Qt.ItemDataRole.UserRole)
        self.setEditable(False)
        
        # Set Icons based on type (using text chars/emojis as placeholders like in original)
        self.setup_appearance()

    def setup_appearance(self):
        icon_text = ""
        if self.node_type == NodeType.PROJECT:
            font = self.font()
            font.setBold(True)
            self.setFont(font)
            icon_text = "📁"
        elif self.node_type == NodeType.GROUP:
            icon_text = "📂"
        elif self.node_type == NodeType.SURROGATE_MODELER:
            icon_text = "🧠" 
        elif self.node_type == NodeType.CLIMATE_GUARDIAN:
            icon_text = "🌍"
        elif self.node_type == NodeType.ASSET_MANAGER:
            icon_text = "💼"
        elif self.node_type == NodeType.OPTIMIZER:
            icon_text = "⚖️"
        elif self.node_type == NodeType.RUN:
            icon_text = "▶️"
        elif self.node_type == NodeType.PARAMETER:
            icon_text = "P"
        
        # In a real app, load QIcon("path/to/icon.png")
        # Here we just prefix the name or leave it to the View if using delegates.
        # But standard QTreeView shows text. Let's prepend emoji to text if not using actual icons
        # The prompt asked for "Icon and Data Assign", assuming actual QIcons or similar.
        # Since I don't have icon files, I'll rely on the text being descriptive as per previous code.
        self.setText(f"{icon_text} {self.text()}")

class SmartModel(QStandardItemModel):
    """
    The Model managing the tree structure.
    """
    def __init__(self):
        super().__init__()
        self.root_item = self.invisibleRootItem()
        self.setup_initial_structure()

    def setup_initial_structure(self):
        """Builds the default tree hierarchy."""
        # Root: SmartCAPEX Project
        self.project_node = SmartNodeItem("SmartCAPEX Project", NodeType.PROJECT)
        self.root_item.appendRow(self.project_node)

        # 1. Global Definitions
        global_def = SmartNodeItem("Global Definitions", NodeType.GROUP)
        global_def.appendRow(SmartNodeItem("Parameters", NodeType.PARAMETER))
        global_def.appendRow(SmartNodeItem("Variables", NodeType.VARIABLE))
        self.project_node.appendRow(global_def)

        # 2. Component 1
        comp1 = SmartNodeItem("Agents", NodeType.GROUP)
        
        # Default Children for Agents
        comp1.appendRow(SmartNodeItem("Surrogate Modeler", NodeType.SURROGATE_MODELER))
        comp1.appendRow(SmartNodeItem("Climate Guardian", NodeType.CLIMATE_GUARDIAN))
        comp1.appendRow(SmartNodeItem("Asset Manager", NodeType.ASSET_MANAGER))
        
        self.project_node.appendRow(comp1)

        # 3. Cases
        cases = SmartNodeItem("Cases", NodeType.GROUP)
        cases.appendRow(SmartNodeItem("Multi-Objective Optimizer", NodeType.OPTIMIZER))
        cases.appendRow(SmartNodeItem("Compute", NodeType.RUN))
        self.project_node.appendRow(cases)

        # 4. Results
        results = SmartNodeItem("Results", NodeType.GROUP)
        results.appendRow(SmartNodeItem("Datasets", NodeType.RESULT_TABLE))
        results.appendRow(SmartNodeItem("Reports", NodeType.RESULT_TABLE))
        self.project_node.appendRow(results)

    def add_node_smart(self, parent_index, node_type):
        """
        Adds a node of `node_type` under the node at `parent_index`.
        If parent_index is invalid or not suitable, finds a suitable parent.
        """
        if not parent_index.isValid():
            parent_item = self.project_node # Default to project root
        else:
            parent_item = self.itemFromIndex(parent_index)
        
        # Logic: If selected item is a leaf, add to its parent.
        # If selected item is a Group or Root, add to it.
        
        # Check if parent allows children (conceptually)
        # For simplicity, if it's a "Component" or "Group" or "Project", we add to it.
        # If it's a specific functional node (like Surrogate), we might add sub-features.
        # Here, let's assume we append to the parent if it's a container, 
        # or append to the parent's parent if the selected one is a leaf.
        
        node_data = parent_item.data(Qt.ItemDataRole.UserRole)
        
        target_parent = parent_item
        if node_data in [NodeType.SURROGATE_MODELER, NodeType.OPTIMIZER, NodeType.RUN, NodeType.PARAMETER]:
            # It's a leaf or specialized node, add to its parent (sibling)
            # UNLESS we are adding a sub-feature. Let's assume we are adding top-level components for now.
             if parent_item.parent():
                 target_parent = parent_item.parent()
        
        new_node = SmartNodeItem(node_type, node_type)
        target_parent.appendRow(new_node)
        return new_node
