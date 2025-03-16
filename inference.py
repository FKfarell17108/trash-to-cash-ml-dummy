import cv2
import numpy as np
import pickle
import os

# Load model dari file pickle
with open('trash_classifier_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Fungsi preprocessing gambar
def preprocess_image(image_path, target_size=(64, 64)):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.flatten()
    return img

# Fungsi prediksi untuk klasifikasi biner
def predict_image(image_path, target_size=(64, 64)):
    img = preprocess_image(image_path, target_size)
    if img is None:
        return {"error": f"Gambar {image_path} tidak dapat dibaca"}
    img = img.reshape(1, -1)
    prediction = model.predict(img)[0]
    return {"prediction": "Anorganik" if prediction == 0 else "Organik"}

# Fungsi prediksi untuk sampah campuran
def predict_mixed_waste(image_path, grid_size=(2, 2)):
    original_img = cv2.imread(image_path)
    if original_img is None:
        return {"error": f"Gambar {image_path} tidak dapat dibaca"}, None
    
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    height, width, _ = original_img.shape
    grid_h, grid_w = height // grid_size[0], width // grid_size[1]
    
    predictions = []
    boxes = []
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            y_start = i * grid_h
            y_end = (i + 1) * grid_h
            x_start = j * grid_w
            x_end = (j + 1) * grid_w
            
            grid_img = original_img[y_start:y_end, x_start:x_end]
            temp_path = f'temp_grid_{i}_{j}.jpg'  # Nama unik untuk setiap grid
            cv2.imwrite(temp_path, cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
            
            grid_data = preprocess_image(temp_path)
            if grid_data is not None:
                grid_data = grid_data.reshape(1, -1)
                pred = model.predict(grid_data)[0]
                predictions.append(pred)
                boxes.append((x_start, y_start, x_end, y_end, pred))
            os.remove(temp_path)  # Hapus file sementara
    
    total_grids = len(predictions)
    if total_grids == 0:
        return {"error": f"Tidak ada prediksi untuk {image_path}"}, None
    
    organik_count = predictions.count(1)
    anorganik_count = predictions.count(0)
    organik_percent = (organik_count / total_grids) * 100
    anorganik_percent = (anorganik_count / total_grids) * 100
    
    # Gambar kotak penanda
    marked_img = original_img.copy()
    for box in boxes:
        x_start, y_start, x_end, y_end, pred = box
        color = (0, 255, 0) if pred == 1 else (255, 0, 0)  # Hijau: organik, Merah: anorganik
        cv2.rectangle(marked_img, (x_start, y_start), (x_end, y_end), color, 2)
        label = "Organik" if pred == 1 else "Anorganik"
        cv2.putText(marked_img, label, (x_start, y_start-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Simpan gambar yang ditandai dengan nama berdasarkan file asli
    output_filename = f"marked_{os.path.basename(image_path)}"
    cv2.imwrite(output_filename, cv2.cvtColor(marked_img, cv2.COLOR_RGB2BGR))
    
    return {
        "organik_percent": organik_percent,
        "anorganik_percent": anorganik_percent,
        "grid_predictions": ["Anorganik" if p == 0 else "Organik" for p in predictions]
    }, output_filename

# Fungsi untuk memproses semua gambar di folder images
def process_images_in_folder(folder_path="images"):
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} tidak ditemukan!")
        return
    
    results = []
    for image_file in os.listdir(folder_path):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, image_file)
            
            # Klasifikasi biner
            binary_result = predict_image(image_path)
            print(f"Hasil klasifikasi biner untuk {image_file}: {binary_result}")
            
            # Klasifikasi campuran
            mixed_result, marked_image_path = predict_mixed_waste(image_path)
            print(f"Hasil klasifikasi campuran untuk {image_file}: {mixed_result}")
            print(f"Gambar ditandai disimpan di: {marked_image_path}")
            
            results.append({
                "image": image_file,
                "binary": binary_result,
                "mixed": mixed_result,
                "marked_image": marked_image_path
            })
    return results

# Contoh penggunaan
if __name__ == "__main__":
    # Proses semua gambar di folder images
    results = process_images_in_folder("images")