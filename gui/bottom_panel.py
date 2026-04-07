from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QTextEdit, 
                             QProgressBar, QLabel, QHBoxLayout)
from PyQt6.QtCore import Qt

class BottomPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(150) # Initial height
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.South)
        # Styles moved to global QSS
        
        # 1. Messages Tab
        self.msg_widget = QTextEdit()
        self.msg_widget.setReadOnly(True)
        self.msg_widget.setPlaceholderText("System messages will appear here...")
        self.tabs.addTab(self.msg_widget, "Messages")
        
        # 2. Progress Tab
        progress_container = QWidget()
        p_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_lbl = QLabel("Ready")
        p_layout.addWidget(self.progress_lbl)
        p_layout.addWidget(self.progress_bar)
        p_layout.addStretch()
        progress_container.setLayout(p_layout)
        self.tabs.addTab(progress_container, "Progress")
        
        # 3. Log Tab
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setStyleSheet("font-family: Consolas; font-size: 10pt;")
        self.tabs.addTab(self.log_widget, "Log")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def update_log(self, text):
        self.log_widget.append(text)
        # Scroll to bottom
        self.log_widget.verticalScrollBar().setValue(self.log_widget.verticalScrollBar().maximum())
