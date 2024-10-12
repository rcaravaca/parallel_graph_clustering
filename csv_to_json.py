import csv
import json
from collections import defaultdict

# Nombre del archivo CSV de entrada
csv_file = 'digits_values_10000.csv'
# Nombre del archivo JSON de salida
json_file = 'digits_values_10000.json'

# Usamos defaultdict para agrupar los eventos
event_dict = defaultdict(list)

print(f"Inciando {csv_file}")
# Abrir el archivo CSV y leerlo
with open(csv_file, mode='r') as file:
    csv_reader = csv.DictReader(file)
    
    # Iterar sobre las filas del archivo CSV
    for row in csv_reader:
        event = int(row["Event"])
        # Crear un diccionario para "digits" con row, col, y energy
        digit = {
            "row": int(row["row"]),
            "col": int(row["col"]),
            "energy": float(row["energy"])
        }
        # Añadir el "digit" al evento correspondiente
        event_dict[event].append(digit)

# Crear la estructura JSON con la agrupación por eventos
data = [
    {
        "Event": event,
        "digits": digits
    }
    for event, digits in event_dict.items()
]

# Escribir los datos en un archivo JSON
with open(json_file, mode='w') as file:
    json.dump(data, file, indent=4)

print(f"Los datos se han convertido y guardado en {json_file}")
