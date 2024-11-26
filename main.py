import cv2
from ultralytics import YOLO

# Cargar el modelo entrenado
model = YOLO('best.pt')

# Leer la imagen
image_path = 'prueba3.jpg'
image = cv2.imread(image_path)

# Verificar si la imagen se cargó correctamente
if image is None:
    print(f"Error al cargar la imagen: {image_path}")
    exit()

# Realizar la inferencia
results = model(image)

# Mostrar los resultados
if len(results) == 1:
    results[0].show()
else:
    for result in results:
        result.show()
