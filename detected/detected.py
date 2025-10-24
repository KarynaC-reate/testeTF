import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageDraw, ImageFont

# --- 1. Configurações e Carregamento do Modelo ---

# URL para um modelo rápido e pré-treinado do TensorFlow Hub (SSD MobileNet V2)
MODEL_URL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2" 

print("Carregando o modelo do TensorFlow Hub...")
detector = hub.load(MODEL_URL)
print("Modelo carregado com sucesso!")

# Mapeamento de Classes COCO (Simplificado)
COCO_CLASSES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 
    5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 
    16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 
    20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra', 
    24: 'giraffe', 25: 'backpack', 27: 'handbag', 31: 'skiboard', 
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 
    77: 'cell phone', 84: 'book', 86: 'scissors', 
}


# --- 2. Função de Visualização Simples (Corrigida e Atualizada) ---
def draw_boxes(image_np, boxes, classes, scores, category_index, min_score_thresh=0.5):
    # Converte o array NumPy do OpenCV para uma imagem PIL
    image_pil = Image.fromarray(image_np)
    draw = ImageDraw.Draw(image_pil)
    im_width, im_height = image_pil.size
    
    # Carrega a fonte padrão
    try:
        # Tenta carregar uma fonte comum e define o tamanho
        font = ImageFont.truetype("arial.ttf", 20) 
    except IOError:
        # Usa a fonte padrão se Arial não for encontrada
        font = ImageFont.load_default() 

    for i in range(boxes.shape[0]):
        if scores[i] >= min_score_thresh:
            # As caixas estão em coordenadas normalizadas (0 a 1)
            ymin, xmin, ymax, xmax = boxes[i]
            
            # Converte para coordenadas de pixel
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width, 
                                          ymin * im_height, ymax * im_height)

            class_id = int(classes[i])
            class_name = category_index.get(class_id, f'Class {class_id}')
            score = scores[i]
            
            display_str = f"{class_name}: {int(100*score)}%"
            
            # --- CORREÇÃO AQUI: Usando draw.textbbox() ---
            padding = 5
            # Obtém a caixa delimitadora do texto: (l, t, r, b)
            # O ponto de ancoragem do texto é (left, top_da_caixa)
            bbox = draw.textbbox((left + padding, top - 20), display_str, font=font)
            
            # Extrai largura e altura da caixa para o fundo
            text_w = bbox[2] - bbox[0] + padding * 2
            text_h = bbox[3] - bbox[1] + padding * 2
            
            # Ajusta a posição vertical para desenhar o fundo do texto acima da caixa
            text_top = top - text_h
            # ----------------------------------------------------------------------
            
            # Desenha o retângulo (caixa delimitadora)
            draw.rectangle([(left, top), (right, bottom)], outline="red", width=3)
            
            # Desenha o fundo do texto (rótulo)
            draw.rectangle([
                (left, text_top), 
                (left + text_w, top)
            ], fill="red")
            
            # Desenha o texto
            draw.text((left + padding, text_top + padding/2), display_str, fill="white", font=font)

    # Converte a imagem PIL de volta para o array NumPy (formato OpenCV)
    return np.array(image_pil)


# --- 3. Loop da Webcam ---
# (Este bloco permanece o mesmo)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Não foi possível abrir a webcam.")

while True:
    ret, frame_np = cap.read()
    
    if ret:
        rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        
        input_tensor = tf.convert_to_tensor(rgb_frame, dtype=tf.uint8)
        input_tensor = tf.expand_dims(input_tensor, 0)

        detections = detector(input_tensor)

        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy()
        
        frame_with_boxes = draw_boxes(
            frame_np, 
            boxes, 
            classes, 
            scores, 
            COCO_CLASSES,
            min_score_thresh=0.5
        )
        
        cv2.imshow('Detecção de Objetos em Tempo Real (TF Hub)', frame_with_boxes)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
