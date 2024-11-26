# import cv2
# from ultralytics import YOLO

# # Cargar el modelo entrenado
# model = YOLO('best.pt')

# # Leer la imagen
# image_path = 'DETER.jpg'
# image = cv2.imread(image_path)

# # Verificar si la imagen se cargó correctamente
# if image is None:
#     print(f"Error al cargar la imagen: {image_path}")
#     exit()

# # Realizar la inferencia
# results = model(image)

# # Mostrar los resultados
# if len(results) == 1:
#     results[0].show()
# else:
#     for result in results:
#         result.show()


import cv2
from ultralytics import YOLO

# Cargar el modelo entrenado
model = YOLO('best.pt')
# Inicializar la captura de video (0 para la cámara por defecto)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

while True:
    # Leer un fotograma de la cámara
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el fotograma.")
        break

    # Realizar la inferencia
    results = model(frame)

    # Dibujar las detecciones en el fotograma
    annotated_frame = results[0].plot()

    # Mostrar el fotograma anotado
    cv2.imshow('Detección en tiempo real', annotated_frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
