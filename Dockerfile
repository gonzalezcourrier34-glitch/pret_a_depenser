# Image Python légère
FROM python:3.12-slim

# Empêche Python de générer des fichiers .pyc
ENV PYTHONDONTWRITEBYTECODE=1

# Force l'affichage direct des logs Python dans le terminal
ENV PYTHONUNBUFFERED=1

# Dossier de travail dans le conteneur
WORKDIR /app

# Installation des dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers de dépendances en premier
# Cela permet de profiter du cache Docker si le code change mais pas les dépendances
COPY pyproject.toml uv.lock ./

# Installation de uv
RUN pip install --no-cache-dir uv

# Installation des dépendances du projet
RUN uv sync --frozen --no-dev

# Copie du reste du projet
COPY . .

# Exposition du port de l'API
EXPOSE 8000

# Commande de lancement de l'API
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]