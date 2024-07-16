
# TimeSeriesForecasting

## Übersicht

**TimeSeriesForecasting** ist ein End-to-End-Projekt zur Analyse und Vorhersage von Zeitreihen mit Python. Das Projekt nutzt Jupyter Notebooks zur Datenanalyse, Python-Skripte zur Datenverarbeitung und Modellierung sowie Docker zur Bereitstellung einer konsistenten Entwicklungsumgebung.

## Projektstruktur

```
TimeSeriesForecasting/
├── data/
├── notebooks/
├── scripts/
├── models/
├── results/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

- `data/`: Enthält die Rohdaten.
- `notebooks/`: Jupyter Notebooks für Datenanalyse und Modellierung.
- `scripts/`: Python-Skripte für verschiedene Aufgaben.
- `models/`: Gespeicherte Modelle.
- `results/`: Ergebnisse und Berichte.
- `Dockerfile`: Definiert die Docker-Umgebung.
- `docker-compose.yml`: Orchestriert Docker-Container.
- `requirements.txt`: Liste der benötigten Python-Pakete.
- `.gitignore`: Dateien und Verzeichnisse, die von Git ignoriert werden sollen.
- `README.md`: Projektbeschreibung und Anleitung.

## Installation

### Voraussetzungen

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Schritte

1. Klone das Repository:

   ```bash
   git clone <repository-url>
   cd TimeSeriesForecasting
   ```

2. Baue und starte den Docker-Container:

   ```bash
   docker-compose up --build
   ```

3. Öffne den Jupyter-Notebook-Server im Browser:

   ```
   http://localhost:8888
   ```

## Nutzung

1. Lege deine Rohdaten im `data/` Verzeichnis ab.
2. Erstelle und bearbeite Jupyter Notebooks im `notebooks/` Verzeichnis.
3. Füge neue Python-Skripte im `scripts/` Verzeichnis hinzu.
4. Speichere Modelle im `models/` Verzeichnis.
5. Speichere Ergebnisse und Berichte im `results/` Verzeichnis.

## Autor

Sangeeths Chandrakumar
