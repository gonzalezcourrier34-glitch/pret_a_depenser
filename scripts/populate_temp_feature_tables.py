"""
Script de remplissage des tables temporaires d'agrégation de features.

Ce module alimente les tables temporaires déjà créées à partir
des tables RAW du projet Home Credit, via des insertions par batch.

Objectif
--------
Séparer la création de structure du remplissage afin de :
- faciliter le débogage
- isoler les erreurs SQL
- rendre le pipeline plus lisible

Notes
-----
- Les tables temporaires doivent déjà exister.
- Les insertions sont réalisées par batch de clés.
- Ce script ne crée pas les tables, il les remplit.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# =============================================================================
# Chargement des variables d'environnement
# =============================================================================

load_dotenv()


# =============================================================================
# Configuration
# =============================================================================

DATABASE_URL = os.getenv("DATABASE_URL")
BATCH_SIZE = 500

if not DATABASE_URL:
    raise ValueError("DATABASE_URL n'est pas défini dans les variables d'environnement")