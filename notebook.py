# 1. MODE: Classificació binària (gats vs. gossos)
class_mode = 'binary'

# 2. DEFINICIÓ DEL MODEL LLEUGER

# Importem les capes necessàries
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

# Creem un model seqüencial (capa rere capa)
model = models.Sequential([

    # Primera capa convolucional amb 8 filtres de mida 3x3 i activació ReLU
    layers.Conv2D(8, (3, 3), activation='relu', input_shape=(100, 100, 3)),

    # Reducció de dimensions amb max pooling (2x2)
    layers.MaxPooling2D(2, 2),

    # Segona capa convolucional amb 16 filtres de mida 3x3
    layers.Conv2D(16, (3, 3), activation='relu'),

    # Max pooling altra vegada per reduir la mida
    layers.MaxPooling2D(2, 2),

    # Aplanem la sortida per passar a les capes completament connectades
    layers.Flatten(),

    # Capa densa amb 32 neurones i activació ReLU
    layers.Dense(32, activation='relu'),

    # Capa de sortida amb 1 neurona i activació sigmoid per classificació binària
    layers.Dense(1, activation='sigmoid')
])

# Compilem el model amb pèrdua binària, optimitzador Adam i mètrica d'exactitud
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 3. ENTRENAMENT DEL MODEL

# Entrenem el model amb el generador d'entrenament i validació durant 10 èpoques
history = model.fit(
    train_generator,                      # Dades d'entrenament
    epochs=10,                            # Nombre d’èpoques
    validation_data=validation_generator # Dades de validació
)

# 4. EXPORTACIÓ DEL MODEL EN FORMAT LLEUGER

# Guardem l'arquitectura del model en format JSON
model_json = model.to_json()

# Obrim un fitxer per escriure-hi l'estructura del model
with open("model_gats_gossos.json", "w") as json_file:
    json_file.write(model_json)

# Guardem només els pesos del model en format H5 (lleuger)
model.save_weights("model_gats_gossos.weights.h5")

# 5. DESCÀRREGA DELS FITXERS PER PUJAR A GITHUB (només a Google Colab)
from google.colab import files

# Descarreguem el fitxer JSON amb l'estructura del model
files.download("model_gats_gossos.json")

# Descarreguem el fitxer H5 amb els pesos del model
files.download("model_gats_gossos.weights.h5")

# 6. GRÀFICA DE LA PRECISIÓ DURANT L’ENTRENAMENT

# Extraiem la precisió d'entrenament per cada època
acc = history.history['accuracy']

# Extraiem la precisió de validació per cada època
val_acc = history.history['val_accuracy']

# Creem un rang d’èpoques per fer el gràfic
epochs_range = range(len(acc))

# Configurem la mida de la figura del gràfic
plt.figure(figsize=(8, 6))

# Dibuixem la corba de precisió en entrenament (vermella)
plt.plot(epochs_range, acc, 'r', label='Entrenament')

# Dibuixem la corba de precisió en validació (blava)
plt.plot(epochs_range, val_acc, 'b', label='Validació')

# Títol del gràfic
plt.title('Evolució de la Precisió')

# Mostrem la llegenda
plt.legend()

# Mostrem el gràfic
plt.show()
