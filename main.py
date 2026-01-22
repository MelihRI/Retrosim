"""
SmartCAPEX AI - Intelligent Multi-Agent Architecture for Maritime Retrofit Decisions
===================================================================================

Author: AI Assistant
Description: Desktop application for intelligent maritime retrofit decision support
Literature Base: "AI-Based Decision Support Systems for Retrofit" (Bocaneala et al.)

Agents:
1. Surrogate Modeler (EANN Core) - Physics-Informed Surrogate Model
2. Multi-Objective Optimizer - Pareto Optimality for Cost vs Performance
3. Climate Guardian - Temporal Projection for 2025-2050 scenarios
4. Asset Manager - UI and Data Validation Layer
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.main_window import SmartCAPEXMainWindow


def main():
    """Main entry point for SmartCAPEX AI application"""
    
    # Create main application window
    root = tk.Tk()
    
    # Configure application
    root.title("SmartCAPEX AI - Maritime Retrofit Decision Support")
    root.geometry("1400x900")
    root.minsize(1200, 800)
    
    # Set application icon (if available)
    try:
        from PIL import Image, ImageTk
        # icon = ImageTk.PhotoImage(Image.open("assets/icon.png"))
        # root.iconphoto(False, icon)
    except:
        pass
    
    # Create and run main application
    app = SmartCAPEXMainWindow(root)
    
    # Handle application closure
    def on_closing():
        if messagebox.askokcancel("Exit", "Do you want to exit SmartCAPEX AI?"):
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Run application
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
        root.destroy()


if __name__ == "__main__":
    main()
