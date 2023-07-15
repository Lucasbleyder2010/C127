import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from matplotlib import pyplot
from matplotlib.image import imread

#Definir CNN
import tensorflow as tf
model = tf.keras.models.Sequential([
    
    # Primeira camada de Convolução e Pooling
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(180, 180, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Segunda camada de Convolução e Pooling
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Terceira camada de Convolução e Pooling
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Quarta camada de Convolução e Pooling
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Achatar (flatten) os resultados para alimentar em uma camada densa
    
    #descomente o código correto para achatar os resultados
    tf.keras.layers.Flatten(),
    #tf.keras.Layers.Flatten(),
    #tf.keras.Layers.flatten(),
    #tf.Keras.layers.Flatten(),
    
    tf.keras.layers.Dropout(0.5),

    # Camada de classificação
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
])

training_damaged_image =  "/Users/CLIENTE/Downloads/C127/C127/C127_Pneumothorax/training_dataset"

# carregue os pixels da imagem
image = imread(training_damaged_image)

pyplot.title("danificado: Imagem 1")

# plote dados brutos de pixel
pyplot.imshow(image)

# exiba a imagem
pyplot.show()

# Aumento aleatório de dados (Redimensionamento, Rotação, Inversões, Zoom, Deslocamentos) usando ImageDataGenerator  
training_data_generator = ImageDataGenerator(
    rescale = 1.0/255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')


# Diretório de Imagens
training_image_directory = "/content/PRO_1-4_C127_PneumothoraxImageDataset/train"

# Gere arquivos de imagem aumentada
training_augmented_images = training_data_generator.flow_from_directory(
    training_image_directory,
    target_size=(180,180))

#Dados de validação
# Aumento aleatório de dados (redimensionamento) usando ImageDataGenerator
validation_data_generator = ImageDataGenerator(rescale = 1.0/255)

# Diretório de Imagens
validation_image_directory = "/content/PRO_1-4_C127_PneumothoraxImageDataset/validate"

# Gere dados aumentados pré-processados
validation_augmented_images = validation_data_generator.flow_from_directory(
    validation_image_directory,
    target_size=(180,180))

#etiquetas das classes
training_augmented_images.class_indices

#Compilar o modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Ajustar e salvar
#descomente o código correto para ajustar e salvar o modelo
history = model.fit(training_augmented_images, 
            epochs=20, validation_data = validation_augmented_images, verbose=True)

#history = model.fit(validation_augmented_images, 
           # epochs=20, validation_data = training_augmented_images, verbose=False)

#history = model.fit(training_Augmented_images, 
           # epochs=20, validation_data = validation_augmented_images, verbose=true)
            
#history = model.fit(validation_augmented_images, 
          #  epochs=20, validation_data = training_Augmented_images, verbose=False)


#model.Save("Hurricane_damage.H5")
#model1.save("Hurricane_damage.h5")
model.save("Hurricane_damage.h5")
#model1.Save("Hurricane_damage.H5")