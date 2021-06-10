import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

# Desative a notação científica para maior clareza
np.set_printoptions(suppress=True)

# Carregue o modelo
model = tensorflow.keras.models.load_model('keras_model.h5')

# Crie a matriz da forma certa para alimentar o modelo keras
# O 'comprimento' ou número de imagens que você pode colocar no array é
# determinado pela primeira posição na tupla de forma, neste caso 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Substitua pelo caminho para a sua imagem
image = Image.open('foto.jpg')

#redimensione a imagem para 224x224 com a mesma estratégia do TM2:
#redimensionar a imagem para pelo menos 224x224 e, em seguida, cortar a partir do centro
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#transformar a imagem em uma matriz numpy
image_array = np.asarray(image)

#mostra a imagem redimensionada
image.show()


normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

#Carregue a imagem no array
data[0] = normalized_image_array

prediction = model.predict(data)
print(prediction)
