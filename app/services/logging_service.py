"""
Middleware de logging HTTP technique.

Ce middleware journalise chaque requête HTTP entrante avec :
- un identifiant unique de requête
- la méthode HTTP
- le chemin appelé
- les query params
- le code de statut
- la latence d'exécution
- l'hôte client si disponible

Objectif
--------
Fournir une traçabilité technique minimale et structurée
de tous les appels API, indépendamment du logging métier
stocké en base PostgreSQL.

Notes
-----
- ce middleware n'enregistre pas les corps de requête
  afin de limiter le risque de fuite de données sensibles
- il ajoute automatiquement le header `X-Request-ID`
  dans la réponse
- en cas d'erreur non gérée, il journalise une exception
  avec le temps écoulé avant réémission
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


logger = logging.getLogger("app.http")


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware de logs HTTP techniques.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Intercepte une requête HTTP, mesure son temps d'exécution
        et produit un log structuré.

        Parameters
        ----------
        request : Request
            Requête HTTP entrante.
        call_next : callable
            Fonction Starlette permettant de transmettre la requête.

        Returns
        -------
        Response
            Réponse HTTP enrichie avec un `X-Request-ID`.
        """
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        method = request.method
        path = request.url.path
        query_params = str(request.query_params)
        client_host = request.client.host if request.client else None

        logger.info(
            "HTTP request started",
            extra={
                "extra_data": {
                    "event": "http_request_start",
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "query_params": query_params,
                    "client_host": client_host,
                }
            },
        )

        try:
            response = await call_next(request)
            latency_ms = round((time.perf_counter() - start) * 1000, 2)

            logger.info(
                "HTTP request completed",
                extra={
                    "extra_data": {
                        "event": "http_request_success",
                        "request_id": request_id,
                        "method": method,
                        "path": path,
                        "query_params": query_params,
                        "status_code": response.status_code,
                        "latency_ms": latency_ms,
                        "client_host": client_host,
                    }
                },
            )

            response.headers["X-Request-ID"] = request_id
            return response

        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)

            logger.exception(
                "Unhandled HTTP error",
                extra={
                    "extra_data": {
                        "event": "http_request_exception",
                        "request_id": request_id,
                        "method": method,
                        "path": path,
                        "query_params": query_params,
                        "latency_ms": latency_ms,
                        "client_host": client_host,
                        "error": str(exc),
                    }
                },
            )
            raise