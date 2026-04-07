# gui/styles.py

DARK_THEME_QSS = """
/* =======================================================
   MODERN PREMIUM DARK THEME - SmartCAPEX AI
   ======================================================= */

:root {
    --bg-darker: #0d1117;
    --bg-main: #161b22;
    --bg-panel: #21262d;
    --accent-blue: #58a6ff;
    --accent-glow: rgba(88, 166, 255, 0.2);
    --text-main: #c9d1d9;
    --text-dim: #8b949e;
    --border-dim: #30363d;
    --success: #238636;
    --danger: #da3633;
}

QWidget {
    background-color: #161b22;
    color: #c9d1d9;
    font-family: 'Segoe UI', 'Roboto', 'Inter', sans-serif;
    font-size: 10pt;
}

/* =======================================================
   RIBBON & TOOLBARS (GLASSMORPHISM INSPIRED)
   ======================================================= */
RibbonWidget {
    background-color: #0d1117;
    border-bottom: 2px solid #30363d;
}

QTabWidget::pane {
    border: 1px solid #30363d;
    background: #161b22;
}

QTabBar::tab {
    background: #0d1117;
    color: #8b949e;
    padding: 10px 20px;
    border: none;
    border-bottom: 2px solid transparent;
}

QTabBar::tab:selected {
    color: #58a6ff;
    border-bottom: 3px solid #58a6ff;
    background: #161b22;
    font-weight: bold;
}

QTabBar::tab:hover:!selected {
    background: #21262d;
    color: #f0f6fc;
}

/* =======================================================
   PANEL HEADERS
   ======================================================= */
QLabel#HeaderLabel {
    background-color: #0d1117;
    color: #58a6ff;
    padding: 8px 15px;
    font-weight: bold;
    font-size: 11pt;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 1px solid #30363d;
}

/* Tree & Model Explorer */
QTreeWidget {
    background-color: #0d1117;
    border: none;
    outline: none;
    padding: 5px;
}

QTreeWidget::item {
    padding: 8px;
    border-radius: 4px;
    margin: 2px 5px;
}

QTreeWidget::item:hover {
    background-color: #21262d;
}

QTreeWidget::item:selected {
    background-color: #1f6feb;
    color: white;
}

/* =======================================================
   INPUTS & FORMS
   ======================================================= */
QGroupBox {
    border: 1px solid #30363d;
    border-radius: 8px;
    margin-top: 25px;
    background-color: #0d1117;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 15px;
    padding: 0 5px;
    color: #58a6ff;
    font-weight: bold;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 6px;
    color: #f0f6fc;
}

QLineEdit:focus, QSpinBox:focus {
    border: 1px solid #58a6ff;
    background-color: #161b22;
}

QPushButton {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 8px 16px;
    color: #c9d1d9;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #30363d;
    border-color: #8b949e;
}

QPushButton:pressed {
    background-color: #161b22;
}

/* Primary Action Blue Button */
QPushButton#btn_primary, QPushButton#btn_compute, QPushButton#btn_train {
    background-color: #238636;
    border: 1px solid rgba(240, 246, 252, 0.1);
    color: white;
    font-weight: bold;
}

QPushButton#btn_primary:hover, QPushButton#btn_compute:hover, QPushButton#btn_train:hover {
    background-color: #2ea043;
}

/* =======================================================
   SCROLLBARS (MINIMALIST)
   ======================================================= */
QScrollBar:vertical {
    background: #0d1117;
    width: 10px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #30363d;
    min-height: 30px;
    border-radius: 5px;
    margin: 2px;
}

QScrollBar::handle:vertical:hover {
    background: #484f58;
}

QScrollBar::add-line, QScrollBar::sub-line {
    height: 0px;
}

/* =======================================================
   STATUS & LOGS
   ======================================================= */
QTextEdit#LogViewer {
    background-color: #0d1117;
    color: #7ee787; /* Github Green-ish for logs */
    font-family: 'Cascadia Code', 'Consolas', monospace;
    font-size: 9pt;
    border: 1px solid #30363d;
    border-radius: 4px;
}

/* Graphics Stack */
QStackedWidget#GraphicsStack {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    margin: 5px;
}

/* =======================================================
   SPLITTER (Panel Ayırıcı)
   ======================================================= */
QSplitter::handle {
    background-color: #30363d;
}

QSplitter::handle:horizontal {
    width: 5px;
    margin: 20px 0px;
    border-radius: 2px;
}

QSplitter::handle:vertical {
    height: 5px;
    margin: 0px 20px;
    border-radius: 2px;
}

QSplitter::handle:hover {
    background-color: #58a6ff;
}

QSplitter::handle:pressed {
    background-color: #1f6feb;
}
"""
