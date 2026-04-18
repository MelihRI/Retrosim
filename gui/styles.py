# gui/styles.py

DARK_THEME_QSS = """
/* =======================================================
   STITCH MCP INFLUENCED MODERN PREMIUM DARK THEME
   Applies Theme: Dark Mode, Space Grotesk, Round_Eight, Accent: #13a4ec
   ======================================================= */

:root {
    --bg-darker: #0b0d10;
    --bg-main: #111418;
    --bg-panel: #1a1e24;
    --accent-primary: #13a4ec;
    --accent-glow: rgba(19, 164, 236, 0.2);
    --text-main: #e2e8f0;
    --text-dim: #94a3b8;
    --border-dim: #334155;
    --success: #10b981;
    --danger: #ef4444;
}

QWidget {
    background-color: #111418;
    color: #e2e8f0;
    font-family: 'Space Grotesk', 'Segoe UI', 'Inter', sans-serif;
    font-size: 10.5pt;
}

/* =======================================================
   RIBBON & TOOLBARS
   ======================================================= */
RibbonWidget {
    background-color: #0b0d10;
    border-bottom: 1px solid #334155;
}

QTabWidget::pane {
    border: 1px solid #334155;
    background: #111418;
    border-radius: 8px;
}

QTabBar::tab {
    background: #0b0d10;
    color: #94a3b8;
    padding: 12px 24px;
    border: none;
    border-bottom: 2px solid transparent;
    font-weight: 500;
}

QTabBar::tab:selected {
    color: #13a4ec;
    border-bottom: 3px solid #13a4ec;
    background: #111418;
    font-weight: bold;
}

QTabBar::tab:hover:!selected {
    background: #1a1e24;
    color: #f8fafc;
}

/* =======================================================
   PANEL HEADERS
   ======================================================= */
QLabel#HeaderLabel {
    background-color: #0b0d10;
    color: #13a4ec;
    padding: 12px 18px;
    font-weight: bold;
    font-size: 12pt;
    letter-spacing: 0.5px;
    border-bottom: 1px solid #334155;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}

/* Tree & Model Explorer */
QTreeWidget {
    background-color: #0b0d10;
    border: 1px solid #334155;
    border-radius: 8px;
    outline: none;
    padding: 8px;
}

QTreeWidget::item {
    padding: 10px;
    border-radius: 6px;
    margin: 3px 6px;
}

QTreeWidget::item:hover {
    background-color: #1a1e24;
}

QTreeWidget::item:selected {
    background-color: #13a4ec;
    color: #ffffff;
    font-weight: bold;
}

/* =======================================================
   INPUTS & FORMS
   ======================================================= */
QGroupBox {
    border: 1px solid #334155;
    border-radius: 8px;
    margin-top: 28px;
    background-color: #1a1e24;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 18px;
    padding: 0 8px;
    color: #13a4ec;
    font-weight: bold;
    font-size: 11pt;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #0b0d10;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 8px 10px;
    color: #f8fafc;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #13a4ec;
    background-color: #111418;
}

QPushButton {
    background-color: #1a1e24;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 10px 18px;
    color: #e2e8f0;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #334155;
    border-color: #94a3b8;
}

QPushButton:pressed {
    background-color: #111418;
}

/* Primary Action Blue Button */
QPushButton#btn_primary, QPushButton#btn_compute, QPushButton#btn_train {
    background-color: #13a4ec;
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #ffffff;
    font-weight: bold;
    border-radius: 8px;
}

QPushButton#btn_primary:hover, QPushButton#btn_compute:hover, QPushButton#btn_train:hover {
    background-color: #0ea5e9;
}

QPushButton#btn_primary:pressed {
    background-color: #0284c7;
}

/* =======================================================
   SCROLLBARS (MINIMALIST)
   ======================================================= */
QScrollBar:vertical {
    background: #0b0d10;
    width: 12px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #334155;
    min-height: 30px;
    border-radius: 6px;
    margin: 2px;
}

QScrollBar::handle:vertical:hover {
    background: #475569;
}

QScrollBar::add-line, QScrollBar::sub-line {
    height: 0px;
}

/* =======================================================
   STATUS & LOGS
   ======================================================= */
QTextEdit#LogViewer {
    background-color: #0b0d10;
    color: #38bdf8;
    font-family: 'Cascadia Code', 'Consolas', monospace;
    font-size: 9.5pt;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 8px;
}

/* Graphics Stack */
QStackedWidget#GraphicsStack {
    background-color: #0b0d10;
    border: 1px solid #334155;
    border-radius: 12px;
    margin: 8px;
}

/* =======================================================
   SPLITTER
   ======================================================= */
QSplitter::handle {
    background-color: #334155;
}

QSplitter::handle:horizontal {
    width: 6px;
    margin: 24px 0px;
    border-radius: 3px;
}

QSplitter::handle:vertical {
    height: 6px;
    margin: 0px 24px;
    border-radius: 3px;
}

QSplitter::handle:hover {
    background-color: #13a4ec;
}

QSplitter::handle:pressed {
    background-color: #0ea5e9;
}
"""
