import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Input 
from tensorflow.keras.callbacks import EarlyStopping 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay) 

# Función validadora para que el bucle por teclado no rompa el programa
def pedir_numero(mensaje, es_entero=True, opciones=None):
    while True:
        try:
            valor = int(input(mensaje)) if es_entero else float(input(mensaje))
            if opciones and valor not in opciones:
                print(f"Por favor, elegí una opción válida: {opciones}")
                continue
            return valor
        except ValueError:
            print("Entrada inválida. Por favor, ingresá un número correcto.")


#Limpiar datos------------------
#Cargar datset
df = pd.read_csv("train.csv")

# Eliminar columnas innecesarias
df = df.drop(
    columns=["PassengerId", "Name", "Ticket", "Cabin"]
)

#Convertir texto a numeros
#sex
df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})

#Embarked
df["Embarked"] = df["Embarked"].map({
    "S": 0,
    "C": 1,
    "Q": 2
})

# Separar características y objetivo
X = df.drop("Survived", axis=1)
y = df["Survived"]


# División en Entrenamiento, Validación y Test
# 70% entrenamiento, 30% restante se divide entre validación y test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# 15% validación y 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

# Tratamiento estricto de nulos después del split (Evita Data Leakage)
mediana_edad = X_train["Age"].median()
moda_embarked = X_train["Embarked"].mode()[0]

X_train["Age"] = X_train["Age"].fillna(mediana_edad)
X_val["Age"] = X_val["Age"].fillna(mediana_edad)
X_test["Age"] = X_test["Age"].fillna(mediana_edad)

X_train["Embarked"] = X_train["Embarked"].fillna(moda_embarked)
X_val["Embarked"] = X_val["Embarked"].fillna(moda_embarked)
X_test["Embarked"] = X_test["Embarked"].fillna(moda_embarked)


# Normalizar datos
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# Modelo secuencial - Red Neuronal
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(16, activation="relu"),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.summary()

# Compilar
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Configuración de Early Stopping
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True
)

# Entrenar
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=50,
    validation_data=(X_val, y_val),
    callbacks=[early_stop], 
    verbose=1
)

# Métricas finales de entrenamiento 
train_acc = history.history["accuracy"][-1]
val_acc = history.history["val_accuracy"][-1]

print("\nAccuracy entrenamiento:", train_acc)
print("Accuracy validación:", val_acc)

#Perdida-------------------------------
print("\nÚltima loss entrenamiento:",
      history.history["loss"][-1])

print("Última loss validación:",
      history.history["val_loss"][-1])
# ----------------------------------------------------------------------


# Guardar gráficos de Pérdida y Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.savefig("accuracy.png")
plt.close()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.savefig("loss.png")
plt.close()


# Evaluar con datos de prueba
loss, acc = model.evaluate(X_test, y_test, verbose=0)

# Predicciones sobre TEST 
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int)

# Métricas
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== RESULTADOS FINALES SOBRE TEST =====")
print("Accuracy:", acc)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Reporte completo
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Matriz de Confusión Gráfica
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Sobrevive", "Sobrevive"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión - Test")
plt.savefig("matriz_confusion.png") 
plt.close()


# Predicciones por teclado
while True:
    print("\n--- NUEVA PREDICCIÓN ---")

    pclass = pedir_numero("Clase (1,2,3): ", es_entero=True, opciones=[1, 2, 3])
    sex = pedir_numero("Sexo (0=hombre, 1=mujer): ", es_entero=True, opciones=[0, 1])
    age = pedir_numero("Edad: ", es_entero=False)
    sibsp = pedir_numero("Hermanos/esposo a bordo: ", es_entero=True)
    parch = pedir_numero("Padres/hijos a bordo: ", es_entero=True)
    fare = pedir_numero("Tarifa: ", es_entero=False)
    embarked = pedir_numero("Puerto (0=S, 1=C, 2=Q): ", es_entero=True, opciones=[0, 1, 2])

    persona_datos = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked
    }])
    persona_escalada = scaler.transform(persona_datos)

    probabilidad = model.predict(persona_escalada, verbose=0)

    print("\nProbabilidad de sobrevivir:", probabilidad[0][0])

    if probabilidad[0][0] >= 0.5:
        print("Resultado: SOBREVIVE")
    else:
        print("Resultado: NO SOBREVIVE")

    seguir = input("\n¿Desea hacer otra consulta? (s/n): ")
    if seguir.lower() != "s":
        break

