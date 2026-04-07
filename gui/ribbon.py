from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QToolBar, 
                             QToolButton, QLabel, QFrame, QHBoxLayout)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QIcon, QAction

class RibbonWidget(QWidget):
    # Signal to broadcast which action was triggered (e.g., "Add\nComponent")
    action_triggered = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setFixedHeight(120)  # Fixed height for ribbon feel
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.tabs = QTabWidget()
        # Styles moved to global QSS
        
        # Define Tabs based on COMSOL image
        self.tab_names = ["File", "Home", "Definitions", "Geometry", "Omniverse", "Materials", "Physics", "Mesh", "Study", "Results", "Developer"]
        
        for name in self.tab_names:
            toolbar = self.create_toolbar_for_tab(name)
            self.tabs.addTab(toolbar, name)
            
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def create_toolbar_for_tab(self, tab_name):
        """Creates a specialized toolbar for each tab to mimic ribbon groups"""
        container = QWidget()
        # container.setStyleSheet("background-color: #f5f6f7;") # Removed for Dark Theme
        h_layout = QHBoxLayout()
        h_layout.setContentsMargins(5, 5, 5, 5)
        h_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        # Placeholder buttons generator based on tab name
        actions = []
        if tab_name == "Home":
            actions = [("Application\nBuilder", "A"), ("Model\nManager", "M"), ("Add\nComponent", "+"), ("Parameters", "P"), ("Variables", "x="), ("Functions", "f(x)")]
        elif tab_name == "Geometry":
            actions = [("Import", "I"), ("Block", "⬜"), ("Cylinder", "⚪"), ("Sphere", "⭕"), ("Booleans", "⋂")]
        elif tab_name == "Mesh":
            actions = [("Build All", "🔨"), ("Free Tet", "△"), ("Swept", "▱"), ("Boundary\nLayers", "≡")]
        elif tab_name == "Omniverse":
            actions = [("Retrofit\nWizard", "⚙️"), ("USD\nLOD 0", "H"), ("USD\nLOD 1", "M"), ("USD\nLOD 2", "L"), ("Swap\nY-Z Axis", "⇅"), ("Optimize\nMesh", "⚡")]
        else:
            actions = [("Feature 1", "1"), ("Feature 2", "2"), ("Options", "⚙")]

        for label_text, icon_text in actions:
            btn_layout = QVBoxLayout()
            btn_layout.setSpacing(2)
            
            # Using simple styles to mimic icons for now
            btn = QToolButton()
            btn.setText(icon_text) 
            btn.setFixedSize(40, 40)
            # Removed inline button style for global QSS
            
            # Connect the signal
            # We use a default argument (name=label_text) to capture the value in the lambda closure
            btn.clicked.connect(lambda checked, name=label_text: self.action_triggered.emit(name))
            
            lbl = QLabel(label_text)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            # Removed inline label style for global QSS
            
            btn_layout.addWidget(btn, 0, Qt.AlignmentFlag.AlignHCenter)
            btn_layout.addWidget(lbl, 0, Qt.AlignmentFlag.AlignHCenter)
            
            wrapper = QWidget()
            wrapper.setLayout(btn_layout)
            h_layout.addWidget(wrapper)
            
            # Add a separator occasionally
            if len(actions) > 3 and actions.index((label_text, icon_text)) == 2:
                line = QFrame()
                line.setFrameShape(QFrame.Shape.VLine)
                line.setFrameShadow(QFrame.Shadow.Sunken)
                h_layout.addWidget(line)

        container.setLayout(h_layout)
        return container
