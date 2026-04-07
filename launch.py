#!/usr/bin/env python3
"""
SmartCAPEX AI Launcher
======================

Simple launcher script for the SmartCAPEX AI desktop application
"""

import sys
import os

# Ensure we're in the correct directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add current directory to path
sys.path.insert(0, '.')

def main():
    """Main launcher function"""
    print("SmartCAPEX AI - Maritime Retrofit Decision Support")
    print("=" * 55)
    print()
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return
    
    print(f"✓ Python version: {sys.version.split()[0]}")
    
    # Check required libraries
    required_libs = [
        ('PyQt6', 'PyQt6'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('scikit-learn', 'sklearn'),
        ('scipy', 'scipy'),
        ('torch', 'torch')
    ]
    
    missing_libs = []
    for lib_name, import_name in required_libs:
        try:
            __import__(import_name)
            print(f"✓ {lib_name}")
        except ImportError:
            print(f"❌ {lib_name}")
            missing_libs.append(lib_name)
    
    if missing_libs:
        print()
        print("Missing libraries:")
        for lib in missing_libs:
            print(f"  - {lib}")
        print()
        print("Please install missing libraries using:")
        print("  pip install " + " ".join(missing_libs))
        return
    
    print()
    print("🚀 Launching SmartCAPEX AI...")
    print()
    print("Note: The first time you run the application, you may need to")
    print("      train the surrogate model, which can take 5-10 minutes.")
    print()
    
    try:
        # Import and launch main application
        from main_gui import main_gui as app_main
        app_main()
    except KeyboardInterrupt:
        print("\n👋 Application terminated by user")
    except Exception as e:
        print(f"\n❌ Error launching application: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
