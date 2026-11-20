
# Procesamiento de imagenes con OpenCV
# Ian Gutierrez NC 0091
# CBTIS 128
# Especialidad: Inteligencia Artificial

# Importamos las librerias
import cv2
import os

# Ruta de la imagen
ruta_imagen = os.path.join("imagenes", "amongus.png")

# Creamos la carpeta de resultados
os.makedirs("resultados", exist_ok=True)

# 1. Leer y mostrar la imagen
image = cv2.imread(ruta_imagen)

cv2.imshow("Imagen original Ian gutierrez 0091", image)
cv2.waitKey(0)

# 2. Convertir imagen a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Imagen gris Ian gutierrez 0091", gray_image)
cv2.imwrite("resultados/amongus_gris.png", gray_image)
cv2.waitKey(0)

# 3. Cambiar el tamaño de la imagen
resized_image = cv2.resize(image, (300, 200))

cv2.imshow("Imagen redimensionada Ian gutierrez 0091", resized_image)
cv2.imwrite("resultados/amongus_redimensionada.png", resized_image)
cv2.waitKey(0)

# 4. Rotar la imagen
(h, w) = image.shape[:2]
center = (w // 2, h // 2)

matrix = cv2.getRotationMatrix2D(center, 90, 1.0)
rotated_image = cv2.warpAffine(image, matrix, (w, h))

cv2.imshow("Imagen rotada Ian gutierrez 0091", rotated_image)
cv2.imwrite("resultados/amongus_rotada.png", rotated_image)
cv2.waitKey(0)

# 5. Aplicar desenfoque Gaussiano
blurred_image = cv2.GaussianBlur(image, (15, 15), 0)

cv2.imshow("Desenfoque Gaussiano Ian gutierrez 0091", blurred_image)
cv2.imwrite("resultados/amongus_desenfocada.png", blurred_image)
cv2.waitKey(0)

# 6. Detectar bordes con Canny
edges = cv2.Canny(gray_image, 100, 200)

cv2.imshow("Deteccion de bordes Ian gutierrez 0091", edges)
cv2.imwrite("resultados/amongus_bordes.png", edges)
cv2.waitKey(0)

# 7. Dibujar figuras y agregar texto
output_image = image.copy()

cv2.rectangle(output_image, (50, 50), (200, 200), (255, 0, 0), 3)

cv2.putText(output_image, "OpenCV 0091", (50, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("Figuras y texto Ian gutierrez 0091", output_image)
cv2.imwrite("resultados/amongusfiguras_texto.png", output_image)
cv2.waitKey(0)

# 8. Aplicar umbralizacion
_, thresh_image = cv2.threshold(
    gray_image, 127, 255, cv2.THRESH_BINARY
)

cv2.imshow("Umbralizacion Ian gutierrez 0091", thresh_image)
cv2.imwrite("resultados/amongus_umbralizacion.png", thresh_image)
cv2.waitKey(0)

# 9. Detectar contornos
contours, _ = cv2.findContours(
    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

contour_image = image.copy()

cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

cv2.imshow("Contornos Ian gutierrez 0091", contour_image)
cv2.imwrite("resultados/amongus_contornos.png", contour_image)
cv2.waitKey(0)

# Cerrar todas las ventanas
cv2.destroyAllWindows()

print("Procesamiento de imagenes terminado correctamente")
print("Ian Gutierrez NC 0091")
