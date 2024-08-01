#PNG
import asyncio
import nest_asyncio
import gxipy as gx
import os
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import models, transforms

# Fotoğrafların kaydedileceği ana dizin
output_dir = 'C:/Users/ADMIN/Desktop/METO_PROJE/h-deneme/output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Kamera dizinlerini oluştur
camera_dirs = {1: os.path.join(output_dir, 'kamera1'), 2: os.path.join(output_dir, 'kamera2')}
for camera_dir in camera_dirs.values():
    if not os.path.exists(camera_dir):
        os.makedirs(camera_dir)

# Birleştirilmiş görsellerin kaydedileceği dizin
merged_output_dir = os.path.join(output_dir, 'merged')
if not os.path.exists(merged_output_dir):
    os.makedirs(merged_output_dir)

# Çıkartılmış alanın kaydedileceği dizin
extracted_output_dir = os.path.join(output_dir, 'extracted')
if not os.path.exists(extracted_output_dir):
    os.makedirs(extracted_output_dir)

# Kesilmiş nesnelerin kaydedileceği dizin
cropped_output_dir = os.path.join(output_dir, 'cropped')
if not os.path.exists(cropped_output_dir):
    os.makedirs(cropped_output_dir)

# Konturlu görüntülerin kaydedileceği dizin
contoured_output_dir = os.path.join(output_dir, 'contoured')
if not os.path.exists(contoured_output_dir):
    os.makedirs(contoured_output_dir)

# PNG dosyalarının kaydedileceği dizin
png_output_dir = os.path.join(output_dir, 'png')
if not os.path.exists(png_output_dir):
    os.makedirs(png_output_dir)

def get_next_index(camera_dir):
    # Belirtilen dizindeki JPG dosyalarının sayısını alır ve bir sonraki dosya için indeks döndürür
    files = [f for f in os.listdir(camera_dir) if f.endswith('.jpg')]
    return len(files) + 1

def save_image(image, filename):
    # Görüntüyü belirtilen dosya adıyla kaydeder
    image.save(filename)
    print(f"Resim kaydedildi: {filename}")

def capture_image(camera, index, camera_dir):
    # Kameradan görüntü alır ve kaydeder
    raw_image = camera.data_stream[0].get_image()
    if raw_image is None:
        print(f"Kameradan görüntü alınamadı: {index}")
        return None
    
    # Ham görüntüyü RGB'ye dönüştür
    rgb_image = raw_image.convert("RGB")
    if rgb_image is None:
        print(f"Görüntü dönüştürülemedi: {index}")
        return None
    
    # Görüntüyü numpy array'e dönüştür ve ardından PIL görüntüsüne çevir
    numpy_image = rgb_image.get_numpy_array()
    pil_image = Image.fromarray(numpy_image)
    
    # Zaman damgası oluştur ve dosya adını belirle
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(camera_dir, f"kamera_{index}_{timestamp}.jpg")
    
    # Görüntüyü kaydet
    save_image(pil_image, filename)
    return filename

def merge_images(image1_path, image2_path, output_path):
    # İlk ve ikinci görüntüyü oku ve döndür
    image1 = Image.open(image1_path).rotate(270, expand=True)  # İlk görüntüyü 270 derece döndür
    image2 = Image.open(image2_path).rotate(-90, expand=True)  # İkinci görüntüyü 90 derece döndür

    # Görüntülerin genişlik ve yüksekliklerini al
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Görüntülerin aynı boyutta olduğundan emin olun
    new_height = max(height1, height2)
    image1 = image1.resize((width1, new_height))
    image2 = image2.resize((width2, new_height))
    
    # Yeni görüntü genişliği ve yüksekliği
    total_width = width1 + width2
    max_height = new_height
    
    # Yeni görüntüyü oluştur
    new_image = Image.new('RGB', (total_width, max_height))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (width1, 0))
    
    # Yeni görüntüyü kaydet
    new_image.save(output_path)
    print(f"Birleştirilmiş resim kaydedildi: {output_path}")

# Derin öğrenme modeli ile nesne algılama
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

def detect_objects(image_path, score_threshold=0.01):  # Skor eşiği daha da düşürüldü
    # Görüntüyü yükle ve tensöre dönüştür
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)

    # Model ile nesne algılama
    with torch.no_grad():
        outputs = model(image_tensor)

    # Yalnızca belirli bir skorun üzerindeki algılamaları döndür
    filtered_outputs = []
    for output in outputs:
        high_score_indices = [i for i, score in enumerate(output['scores']) if score > score_threshold]
        filtered_outputs.append({
            'boxes': output['boxes'][high_score_indices],
            'labels': output['labels'][high_score_indices],
            'scores': output['scores'][high_score_indices]
        })

    # Algılanan nesneleri ve skorlarını yazdır
    for i, output in enumerate(filtered_outputs):
        print(f"Output {i}:")
        for j in range(len(output['boxes'])):
            print(f"  Box: {output['boxes'][j].numpy()}, Score: {output['scores'][j].item()}")

    return filtered_outputs

def draw_contours_and_save_crops(image_path, outputs):
    # Görüntüyü yükle
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    for output in outputs:
        for box in output['boxes']:
            x1, y1, x2, y2 = map(int, box)
            
            width_padding = int((x2 - x1) * 0.02)
            height_padding = int((y2 - y1) * 0.02)

            x1 = max(0, x1 - width_padding)
            y1 = max(0, y1 - height_padding)
            x2 = min(image.shape[1], x2 + width_padding)
            y2 = min(image.shape[0], y2 + height_padding)
            
            cropped_image = image[y1:y2, x1:x2]
            gray_cropped_image = gray_image[y1:y2, x1:x2]
            
            # Kenar algılama için bulanıklaştırma ve eşikleme işlemleri
            blurred = cv2.GaussianBlur(gray_cropped_image, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            edges = cv2.Canny(thresh, 50, 150)
            
            # Kontur bulma
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Konturları çiz
            contoured_image = cropped_image.copy()
            cv2.drawContours(contoured_image, contours, -1, (0, 255, 0), 2)
            
            # Kesilmiş görüntüyü kaydet
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            crop_filename = os.path.join(cropped_output_dir, f"crop_{timestamp}.jpg")
            contoured_filename = os.path.join(contoured_output_dir, f"contoured_{timestamp}.jpg")
            
            cv2.imwrite(crop_filename, cropped_image)
            cv2.imwrite(contoured_filename, contoured_image)
            print(f"Nesne kesilip kaydedildi: {crop_filename}")
            print(f"Konturlu resim kaydedildi: {contoured_filename}")
            
            # Konturları kullanarak PNG formatında görüntüyü kaydet
            save_cropped_png(cropped_image, contours, timestamp)
            break

def save_cropped_png(cropped_image, contours, timestamp):
    mask = np.zeros(cropped_image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    result = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
    
    b, g, r = cv2.split(result)
    alpha = mask
    rgba = cv2.merge([b, g, r, alpha])
    
    png_filename = os.path.join(png_output_dir, f"cropped_{timestamp}.png")
    cv2.imwrite(png_filename, rgba)
    print(f"Konturlu PNG görüntü kaydedildi: {png_filename}")

# Asenkron resim çekme fonksiyonu
async def async_capture_image(camera, index, camera_dir):
    # Asenkron olarak resim çeker ve kaydeder
    return await loop.run_in_executor(None, capture_image, camera, index, camera_dir)

async def main_async():
    # Cihaz yöneticisini başlatır ve cihaz listesini günceller
    device_manager = gx.DeviceManager()
    dev_num, dev_info_list = device_manager.update_device_list()
    if dev_num == 0:
        print("Cihaz bulunamadı")
        return

    if dev_num < 2:
        print("En az iki kamera gereklidir")
        return

    cam1 = None
    cam2 = None
    image1_path = None
    image2_path = None

    try:
        # İki kamerayı da açar
        cam1 = device_manager.open_device_by_index(2)
        cam2 = device_manager.open_device_by_index(1)

        # Kameraların veri akışını başlatır
        cam1.stream_on()
        cam2.stream_on()

        # Her iki kamera için de sonraki dosya indeksini alır
        next_index_cam1 = get_next_index(camera_dirs[1])
        next_index_cam2 = get_next_index(camera_dirs[2])

        # Asenkron olarak iki kameradan da görüntü çeker
        image1_path, image2_path = await asyncio.gather(
            async_capture_image(cam1, next_index_cam1, camera_dirs[1]),
            async_capture_image(cam2, next_index_cam2, camera_dirs[2])
        )

        # Kameraların veri akışını durdurur
        cam1.stream_off()
        cam2.stream_off()

    except Exception as e:
        # Herhangi bir hata durumunda hata mesajını yazdırır
        print(f"Kameralarla ilgili bir hata oluştu: {e}")

    finally:
        # Kameraları kapatır
        if cam1:
            cam1.close_device()
        if cam2:
            cam2.close_device()

    # İki görüntü de başarıyla alındıysa birleştirir
    if image1_path and image2_path:
        # Birleştirilmiş görüntünün dosya yolunu belirler
        merged_image_path = os.path.join(merged_output_dir, f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        merge_images(image1_path, image2_path, merged_image_path)

        # Algılanan nesneler için çıktı dosya yolunu belirler
        outputs = detect_objects(merged_image_path)
        draw_contours_and_save_crops(merged_image_path, outputs)

if __name__ == "__main__":
    # Asenkron olay döngüsünü başlatır
    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main_async())
