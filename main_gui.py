import os
import sys

# Windows CMD Unicode Emoji Fix
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# PyTorch backend (no special DLL workaround needed)
try:
    import torch
    print(f"   -> PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
except ImportError:
    print("   -> PyTorch not found, CPU mode only.")

import traceback # Hata izini sürmek için

print("--- BAŞLANGIÇ: Program Tetiklendi ---")

try:
    # 1. Klasör Yolu Kontrolü
    current_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_path)
    print(f"1. Çalışma Dizini Ayarlandı: {current_path}")

    # 2. PyQt6 Kontrolü
    print("2. PyQt6 kütüphanesi aranıyor...")
    try:
        from PyQt6.QtWidgets import QApplication
        print("   -> BAŞARILI: PyQt6 bulundu.")
    except ImportError:
        print("   -> HATA: PyQt6 kurulu değil. pip install PyQt6")
        input("Çıkmak için Enter'a bas...")
        sys.exit()

    # 3. GUI Dosyası Kontrolü
    print("3. gui/main_window.py dosyası aranıyor...")
    try:
        from gui.main_window import SmartCAPEXMainWindow
        print("   -> BAŞARILI: GUI dosyası import edildi.")
    except ImportError as e:
        print(f"   -> HATA: GUI dosyası bulunamadı veya içinde hata var. Detay: {e}")
        traceback.print_exc()
        input("Çıkmak için Enter'a bas...")
        sys.exit()

    # 4. Uygulama Başlatma
    print("4. QApplication oluşturuluyor...")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    print("5. Ana Pencere (Window) nesnesi üretiliyor...")
    window = SmartCAPEXMainWindow()
    
    print("6. Pencere gösteriliyor (window.show)...")
    window.show()

    print("7. Uygulama döngüsüne (Event Loop) giriliyor. Pencere açılmalı...")
    exit_code = app.exec()
    print(f"--- BİTİŞ: Uygulama {exit_code} koduyla kapandı ---")
    sys.exit(exit_code)

except:
    traceback.print_exc()
    sys.exit(1)