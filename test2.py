import cv2
from ultralytics import YOLO

# --- 1. Configurações e Carregamento do Modelo ---

# Carrega o modelo YOLOv8. 
# 'yolov8n.pt' é o modelo nano, o mais rápido para tempo real.
# Ele será baixado automaticamente se não for encontrado localmente.
try:
    # O YOLOv8 pode usar modelos pré-treinados em datasets de fogo (ex: yolov8n-fire.pt)
    # Se você tiver um modelo customizado treinado para fogo, substitua 'yolov8n.pt'
    # Por agora, usaremos o modelo padrão e focaremos nas caixas delimitadoras.
    model = YOLO('yolov8n.pt') 
    print("Modelo YOLOv8 carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo YOLO: {e}")
    print("Verifique sua instalação da biblioteca ultralytics.")
    exit()


# --- 2. Loop da Webcam ---

# Inicia a captura de vídeo (0 geralmente é a webcam padrão)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Não foi possível abrir a webcam.")

print("Iniciando detecção de fogo em tempo real. Pressione 'q' para sair.")

while True:
    # Captura frame-a-frame
    ret, frame = cap.read()
    
    if ret:
        # Redimensiona o frame para processamento mais rápido (opcional)
        frame = cv2.resize(frame, (640, 480))
        
        # --- Executa a Inferência ---
        # A função predict executa a detecção e desenha as caixas automaticamente.
        # 'conf=0.4' define o limite de confiança para a detecção (40%)
        # 'verbose=False' desativa a impressão de resultados no console.
        results = model.predict(source=frame, conf=0.4, verbose=False)
        
        # O objeto 'results' contém todas as informações, incluindo o frame com as detecções desenhadas
        
        # Pega o frame processado
        processed_frame = results[0].plot()

        # --- Lógica para Alerta (Opcional) ---
        fire_detected = False
        
        # Itera sobre as detecções (boxes)
        for box in results[0].boxes:
            # Obtém o ID da classe (0 para 'person', 2 para 'car', etc. no COCO)
            class_id = int(box.cls.cpu().numpy()[0])
            class_name = model.names[class_id]
            
            # **AQUI VOCÊ PRECISARIA DA CLASSE 'FIRE'**
            # Como estamos usando um modelo COCO genérico, o fogo pode ser classificado 
            # como 'bowl' (tigela) ou 'cup' (copo) de forma errada, ou mesmo 'frisbee'.
            # Se você usar um modelo treinado em fogo, troque 'person' por 'fire'.
            # Para fins de demonstração, vamos simular a detecção de um objeto específico.
            
            # EXEMPLO: Se estivéssemos detectando "pessoa"
            if class_name == 'person': 
                cv2.putText(processed_frame, f"ALERTA! Pessoa Detectada!", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                fire_detected = True # Simulação de alerta
                
            # Se você tiver um modelo customizado para FOGO, a condição seria:
            # if class_name == 'fire':
            #     cv2.putText(processed_frame, "ALERTA: FOGO!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            #     fire_detected = True
            
        
        cv2.imshow('Detecção de Objetos com YOLOv8', processed_frame)
    
    # Sai do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpeza
cap.release()
cv2.destroyAllWindows()