# 6. Gestion des erreurs et journalisation

## 6.1 Stratégie de gestion des erreurs

La gestion des erreurs repose sur une stratégie de try/except localisée au niveau de chaque outil et validation au niveau des entrées utilisateur. L'objectif est d'éviter que des erreurs ponctuelles remontent à l'orchestrateur sous forme d'exception non gérée, qui arrêterait l'exécution ; à la place, les erreurs sont capturées, loggées, et retournées comme dictionnaires de résultat structurés avec champ `error`.

### 6.1.1 Validation des entrées

La fonction `validate_niche_input(niche: str)` applique les règles suivantes :
- **Longueur** : 2 à 200 caractères (niche vide ou abusivement long rejetée).
- **Caractères interdits** : `< > { } \ `` (prévention injection de commande et erreurs de parsing).
- **Trimming** : whitespace enlevé des extrémités.

Toute entrée invalide déclenche une ValueError avec message explicite à l'utilisateur.

### 6.1.2 Catégories d'erreurs et réponses

| Erreur | Source | Réponse | Logging |
|---|---|---|---|
| Dépassement quota API Gemini | CrewAI / manager LLM | Backoff exponentiel (2s, 4s, 8s), max 3 retries, puis fail gracieux | `tool_error` + message |
| Clé API Gemini invalide | Startup `.env` | Exit immédiat avec instruction setup `.env` | `crew_error` (global) |
| Fichier modèle RoBERTa absent | `sentiment_tool` startup | Vérifier `MODEL_PATH`, fournir lien de téléchargement | `tool_error` + instruction |
| Erreur inférence RoBERTa | `sentiment_tool` call-time | Retourner `{"sentiment": "UNKNOWN", "confidence": 0.0, "error": "inference failed"}` | `tool_error` |
| Prédiction confidence < 0.6 | `sentiment_tool` | Retourner `label_used="UNCERTAIN"` (logique normale, non erreur) | `tool_called` avec label `UNCERTAIN` |
| Timeout recherche DuckDuckGo | `web_search_tool` | Retourner résultats partiels ou liste vide, log warning | `tool_error` + warning |
| Catégorie avis inconnue | `review_loader_tool` | Retourner liste des catégories disponibles, prompt utilisateur | `tool_error` + suggestions |
| Entrée avis vide | `review_loader_tool` | Retourner dictionnaire erreur, continuer pipeline | `tool_error` |
| Sortie JSON LLM malformée | CrewAI / agent response parsing | Parser avec fallback regex, logger l'erreur brute | `tool_error` + raw output |
| Exception inattendue (global) | N'importe quel point | Try/catch global en main.py, log traceback complet, exit code 1 | `crew_error` |

Chaque outil enveloppe son cœur fonctionnel dans un bloc try/except et retourne un dictionnaire d'erreur cohérent plutôt que lever une exception. Ceci permet aux agents CrewAI de continuer et d'ajuster leur stratégie si un outil a échoué.

## 6.2 Système de journalisation structurée

Toute exécution du système génère un fichier de log JSONL unique dans le répertoire `logs/`, nommé `run_<timestamp>.jsonl` où timestamp = `YYYYMMDD_HHMMSS`. Chaque ligne du fichier est un objet JSON valide représentant un événement dans le pipeline.

### 6.2.1 Types d'événements

Le système enregistre les 8 types d'événements suivants :

| Type | Agent/Phase | Données loggées |
|---|---|---|
| `crew_start` | Manager | Niche requêtée, config (model, LLM), run_id, timestamp |
| `task_delegated` | Manager | Task ID, description, agent assigné, ordre d'exécution |
| `tool_called` | Tout agent | Nom outil, input params, output, duration_ms, status="success" |
| `tool_error` | Tout agent | Nom outil, input params, error message, traceback, status="error" |
| `hitl_checkpoint` | Manager | Résumé préliminaire affiché, réaction utilisateur (approve/feedback) |
| `agent_response` | Tout agent | Agent name, task description, output final texte/JSON, duration |
| `report_generated` | Manager | Chemin fichier rapport, word count, format (Markdown) |
| `crew_complete` | Manager | Statut final (success/failure), durée totale pipeline, erreurs agrégées |

### 6.2.2 Format et structure

Chaque entrée JSONL suit ce schéma :

```json
{
  "timestamp": "2026-05-XX_HH:MM:SS.sss",
  "run_id": "run_20260505_143000",
  "event_type": "tool_called",
  "agent": "sentiment_analyst",
  "action": "analyze_sentiment",
  "input": {
    "review_text": "This product is amazing!"
  },
  "output": {
    "sentiment": "POSITIVE",
    "confidence": 0.97,
    "label_used": "POSITIVE",
    "scores": {
      "POSITIVE": 0.97,
      "NEUTRAL": 0.02,
      "NEGATIVE": 0.01
    }
  },
  "duration_ms": 145,
  "status": "success",
  "error": null
}
```

### 6.2.3 Implémentation : classe AgentLogger

La classe `AgentLogger` (fichier `src/utils/logger.py`) fournit une interface thread-safe pour l'enregistrement. Les caractéristiques principales sont :

- **Thread-safe** : verrou (Lock) protège l'accès concurrent au fichier JSONL ; plusieurs agents peuvent loguer simultanément sans race condition.
- **Méthodes spécialisées** : chaque type d'événement a une méthode pratique (`crew_start()`, `tool_called()`, `tool_error()`, `hitl_checkpoint()`, `agent_response()`, `report_generated()`, `crew_complete()`, `task_delegated()`) acceptant les paramètres requis et construisant l'objet JSON.
- **Unicité run_id** : génération automatique du run_id via timestamp au premier log, invariant pour la durée d'une exécution.
- **Encoding UTF-8** : logs supportent les caractères accentués et caractères spéciaux (POSITIVE / NÉGATIF, etc.).

Exemple d'utilisation dans un agent :

```python
logger = AgentLogger("logs/agent_20260503T200000Z.jsonl")
logger.tool_called(
    tool_name="analyze_sentiment",
    input_data={"review_text": "..."},
    output_data=result,
    duration_ms=elapsed,
    agent="sentiment_analyst",
)
```

### 6.2.4 Post-traitement et audit

Les fichiers JSONL accumulés peuvent être post-traités pour :
- Audit : filtrer par `event_type="tool_error"` pour identifier les pannes.
- Métriques : agréger `duration_ms` par outil pour identifier les goulots d'étranglement.
- Traçabilité : reconstituer la séquence chronologique complète d'une exécution.
- Validation : vérifier que chaque `task_delegated` a un `agent_response` correspondant.

Un script Python d'analyse des logs est fourni (non documenté ici) permettant la visualisation d'exécution type.
