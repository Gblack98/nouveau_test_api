# Étape de construction
FROM python:3.11-slim as builder

# Installer les dépendances système + outils d'archives
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    unzip \
    tar \
    gzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Étape finale
FROM python:3.11-slim

# Installer les outils d'extraction uniquement
RUN apt-get update && apt-get install -y \
    tar \
    gzip \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    APP_HOME=/app \
    MODELS_DIR=/app/models

WORKDIR $APP_HOME

# Copier l'environnement Python
COPY --from=builder /opt/venv /opt/venv

# Copier le code et l'archive des modèles
COPY models.tar.gz .
COPY main.py .

# Commandes d'initialisation
RUN mkdir -p $MODELS_DIR && \
    tar -xzf models.tar.gz -C $MODELS_DIR && \
    rm models.tar.gz && \
    mkdir -p /tmp/pestai_crops && \
    chmod -R a+r $APP_HOME && \
    find $APP_HOME -type d -exec chmod a+x {} \; && \
    # Nettoyer les outils d'extraction après usage
    apt-get remove -y tar gzip && \
    apt-get autoremove -y

ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

