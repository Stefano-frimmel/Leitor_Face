# Requisitos: pip install opencv-python mediapipe

import cv2
import mediapipe as mp

# Inicializa o detector de rosto do mediapipe
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Carrega o classificador Haar Cascade para objetos (exemplo: rosto)
object_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')  # Remova ou comente esta linha

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Conversão para RGB para o mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    # Desenha retângulos nos rostos detectados
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Rosto', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Detecção de objetos (exemplo: rosto)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    objects = object_cascade.detectMultiScale(gray, 1.1, 1)
    for (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, 'Objeto', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    # Detecção de carros
    # cars = car_cascade.detectMultiScale(gray, 1.1, 1)  # Remova ou comente esta linha
    # for (x, y, w, h) in cars:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #     cv2.putText(frame, 'Carro', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    cv2.imshow('Detecção de Rostos e Objetos', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

cap.release()
cv2.destroyAllWindows()