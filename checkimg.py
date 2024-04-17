import os
from PIL import Image

def check_images(directory):
    # 檢查目錄是否存在
    if not os.path.isdir(directory):
        print("指定的目錄不存在。")
        return
    
    # 列出目錄中的所有檔案
    files = os.listdir(directory)
    
    # 檢查每個檔案是否是圖像文件
    for file in files:
        try:
            # 嘗試打開圖像文件
            image_path = os.path.join(directory, file)
            img = Image.open(image_path)
            img.verify()  # 驗證圖像文件是否完整
            print(f"{file}: OK")
        except Exception as e:
            print(f"{file}: 檢測到問題 - {e}")

# 指定要檢查的資料夾路徑
folder_path = "./data/masks/"

# 執行檢查
check_images(folder_path)
