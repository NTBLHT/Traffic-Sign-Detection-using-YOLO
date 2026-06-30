import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN INPUT / OUTPUT
# ==========================================
INPUT_IMAGE_DIR = r"C:\Users\Dell\Desktop\DPL302m\train\images"
INPUT_LABEL_DIR = r"C:\Users\Dell\Desktop\DPL302m\train\labels"

OUTPUT_IMAGE_DIR = r"C:\Users\Dell\Desktop\DPL302m\train\images_processed"
OUTPUT_LABEL_DIR = r"C:\Users\Dell\Desktop\DPL302m\train\labels_processed"

TARGET_SIZE = 640 

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# Khai báo tên 56 lớp để hiển thị lên biểu đồ cột
CLASS_NAMES = {0: "Cam_o_to", 1: "Cam_quay_dau", 2: "Nguy_hiem_duong_giao", 3: "Chi_dan"}
for i in range(56):
    if i not in CLASS_NAMES: CLASS_NAMES[i] = f"Class_{i}"

# ==========================================
# 2. CÁC HÀM XỬ LÝ TOÁN HỌC & ĐỒ HỌA
# ==========================================
def letterbox_image(image, labels, target_size=640):
    """ Resize ảnh về khung vuông, thêm viền xám chống méo biển báo """
    h_orig, w_orig = image.shape[:2]
    scale = min(target_size / w_orig, target_size / h_orig)
    new_w, new_h = int(w_orig * scale), int(h_orig * scale)
    
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    processed_image = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    processed_image[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_image
    
    new_labels = []
    for label in labels:
        class_id, x_txt, y_txt, w_txt, h_txt = label
        x_pixel = (x_txt * w_orig * scale) + pad_x
        y_pixel = (y_txt * h_orig * scale) + pad_y
        w_pixel = w_txt * w_orig * scale
        h_pixel = h_txt * h_orig * scale
        
        new_labels.append([
            class_id, 
            np.clip(x_pixel / target_size, 0.0, 1.0),
            np.clip(y_pixel / target_size, 0.0, 1.0),
            np.clip(w_pixel / target_size, 0.0, 1.0),
            np.clip(h_pixel / target_size, 0.0, 1.0)
        ])
    return processed_image, new_labels

def apply_clahe_color(image):
    """ Cân bằng sáng cục bộ chống lóa/tối bằng hệ màu LAB """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

# ==========================================
# 3. QUY TRÌNH CHẠY CHÍNH (MAIN PROCESS)
# ==========================================
def main():
    image_files = glob.glob(os.path.join(INPUT_IMAGE_DIR, "*.*"))
    image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"🚀 Bắt đầu xử lý tổng lực cho {len(image_files)} ảnh...")
    
    all_annotations = [] # Mảng lưu thông tin để vẽ biểu đồ
    corrupted_count, mismatch_count = 0, 0
    
    for img_path in tqdm(image_files, desc="Processing Images"):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(INPUT_LABEL_DIR, base_name + ".txt")
        
        if not os.path.exists(lbl_path): mismatch_count += 1; continue
        image = cv2.imread(img_path)
        if image is None: corrupted_count += 1; continue
            
        labels = []
        with open(lbl_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                    
        # --- BƯỚC THỰC THI TIỀN XỬ LÝ ẢNH ---
        boxed_image, updated_labels = letterbox_image(image, labels, target_size=TARGET_SIZE)
        final_image = apply_clahe_color(boxed_image)
        
        # --- LƯU KẾT QUẢ ẢNH VÀ NHÃN MỚI ---
        cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, os.path.basename(img_path)), final_image)
        with open(os.path.join(OUTPUT_LABEL_DIR, base_name + ".txt"), 'w') as f:
            for lbl in updated_labels:
                f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")
                
                # Đồng thời nạp tọa độ MỚI vào mảng phân tích để vẽ biểu đồ chuẩn xác sau xử lý
                all_annotations.append({
                    "class_name": CLASS_NAMES.get(lbl[0], f"Class_{lbl[0]}"),
                    "x_center": lbl[1], "y_center": lbl[2], "width": lbl[3], "height": lbl[4]
                })

    print("\n✅ Đã tiền xử lý ảnh xong! Bắt đầu tạo biểu đồ tự động...")
    
    # --- BƯỚC TỰ ĐỘNG VẼ BIỂU ĐỒ ---
    df = pd.DataFrame(all_annotations)
    sns.set_theme(style="whitegrid")
    
    # Biểu đồ 1: Phân phối lớp (Top 30)
    plt.figure(figsize=(14, 6))
    sns.barplot(x=df['class_name'].value_counts().values[:30], y=df['class_name'].value_counts().index[:30], palette="flare")
    plt.title("Phân Phối Số Lượng Bounding Box Theo Từng Lớp (Sau Gộp Dữ Liệu)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("chart_1_class_distribution.png", dpi=300)
    plt.close()
    
    # Biểu đồ 2: Heatmap mật độ vị trí tâm biển báo
    plt.figure(figsize=(6, 6))
    sns.kdeplot(x=df['x_center'], y=df['y_center'], cmap="Blues", fill=True, thresh=0.05)
    plt.title("Heatmap Vị Trí Tâm Biển Báo Trên Khung Hình Chuẩn 640x640", fontsize=12, fontweight='bold')
    plt.xlim(0, 1); plt.ylim(1, 0) # Đảo trục Y theo ma trận ảnh
    plt.tight_layout()
    plt.savefig("chart_2_location_heatmap.png", dpi=300)
    plt.close()

    # Biểu đồ 3: Biểu đồ điểm kích thước (Width vs Height)
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x='width', y='height', alpha=0.3, color='purple')
    plt.title("Đồ Thị Phân Phối Kích Thước Hộp Giới Hạn Biển Báo", fontsize=12, fontweight='bold')
    plt.xlim(0, 0.4); plt.ylim(0, 0.4)
    plt.tight_layout()
    plt.savefig("chart_3_size_scatter.png", dpi=300)
    plt.close()

    print("==================================================")
    print("🎉 TẤT CẢ ĐÃ SẴN SÀNG!")
    print(f"• Ảnh và nhãn sạch (640x640 + CLAHE): Lưu tại `{OUTPUT_IMAGE_DIR}`")
    print("• 3 file biểu đồ (.png) chất lượng cao đã xuất ra thư mục chạy code!")
    print("==================================================")

if __name__ == "__main__":
    main()