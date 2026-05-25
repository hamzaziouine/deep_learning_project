# 4. Conception détaillée des agents et des outils

## 4.1 Vue d'ensemble de l'équipe d'agents

Le système repose sur trois entités logiques : deux agents spécialisés et un orchestrateur implicite créé par CrewAI lorsque le processus est configuré en mode hiérarchique. Chaque agent dispose d'un rôle précis, d'un objectif mesurable, d'une narration (backstory) qui guide son raisonnement, et d'un sous-ensemble d'outils qu'il est seul autorisé à invoquer. Cette ségrégation des responsabilités évite les conflits d'attribution et facilite l'audit des décisions par lecture du journal d'exécution.

| Agent | Rôle (CrewAI) | Outils assignés | Sortie attendue |
|---|---|---|---|
| Market Research Agent | Analyste de marché | `web_search_tool` | Liste structurée de concurrents, prix, tendances |
| Sentiment Analyst Agent | Analyste de sentiment client | `review_loader_tool`, `sentiment_tool` | Distribution des sentiments, top thèmes |
| Manager (implicite) | Orchestrateur hiérarchique | aucun (délégation pure) | Synthèse finale soumise à validation HITL |

Le manager n'exécute aucun outil ; sa seule responsabilité est de planifier, déléguer aux agents spécialisés, vérifier la cohérence des sorties intermédiaires, et formuler la synthèse soumise à l'utilisateur.

## 4.2 Market Research Agent

L'agent Market Research a pour mission d'établir la cartographie concurrentielle d'une niche produit donnée. Son `goal` est formulé pour cibler trois axes : identifier les principaux acteurs du marché, estimer la fourchette de prix observée, et repérer les tendances récentes (innovations, retournements, controverses) susceptibles d'influer sur une stratégie de lancement.

L'outil unique dont il dispose, `web_search_tool`, encapsule l'API DuckDuckGo via la bibliothèque `duckduckgo-search`. Le tool applique trois protections : limitation de débit (rate limiting à 1 requête par seconde via un verrou en mémoire), repli exponentiel en cas d'erreur transitoire (retries jusqu'à 3 tentatives avec backoff 2-4-8 s), et borne sur la taille des snippets retournés (1500 caractères par résultat) pour éviter la saturation du contexte de l'agent. Le tool retourne une liste de dictionnaires JSON contenant `title`, `url`, `snippet` — un format que CrewAI peut transmettre directement à l'agent comme observation tool-use.

Le backstory positionne l'agent comme un consultant expérimenté en intelligence économique, l'incitant à formuler des requêtes de recherche concises et à filtrer les résultats non pertinents au lieu de tout retransmettre brut au manager.

## 4.3 Sentiment Analyst Agent

L'agent Sentiment Analyst opère sur les avis clients du dataset Amazon Reviews 2018. Son flux interne est en deux temps : chargement des avis filtrés sur la niche, puis classification par le modèle RoBERTa-base fin-tuné.

Le premier outil, `review_loader_tool`, lit le fichier CSV échantillon (`data/sample/electronics_100.csv` par défaut, configurable via la variable d'environnement `DATA_PATH`). Une particularité d'implémentation mérite mention : la résolution du chemin se fait au moment de l'appel et non au moment de l'import, car CrewAI invoque automatiquement `load_dotenv` au boot, ce qui modifie l'environnement et peut écraser la valeur par défaut figée à l'import. Le tool retourne une liste d'avis avec `review_id`, `text`, `stars` (label de référence) et `verified_purchase` (utile pour pondérer la confiance).

Le second outil, `sentiment_tool`, encapsule l'inférence du modèle RoBERTa-base. Le modèle est chargé en mode lazy (singleton process-level) afin d'éviter le coût de chargement répété (~3 s par appel) ; après le premier appel, le modèle reste en mémoire pour la durée du processus. Le tool retourne pour chaque avis un dictionnaire `{label, confidence}`, où `label ∈ {NEGATIVE, NEUTRAL, POSITIVE, UNCERTAIN}`. Le label `UNCERTAIN` est attribué dès que la probabilité maximale prédite par le modèle tombe sous le seuil 0,6, garantissant que les classifications ambiguës n'altèrent pas les agrégations statistiques.

Le backstory de l'agent l'oriente vers une lecture qualitative : non seulement compter les sentiments, mais aussi extraire les thèmes récurrents (qualité, prix, livraison, durabilité, fonctionnalités spécifiques) qui donnent du sens aux pourcentages.

## 4.4 Orchestrateur hiérarchique (manager)

CrewAI matérialise l'orchestrateur lorsqu'on instancie une `Crew` avec `process=Process.hierarchical` et `manager_llm` défini sur un LLM autonome. Le manager n'est pas un agent au sens classique du framework (il n'est pas listé dans `agents=[…]`), mais un acteur planificateur que CrewAI génère en interne. Il reçoit en entrée la liste des tâches et celle des agents disponibles, et décide pour chaque tâche : (a) à quel agent la confier, (b) avec quel contexte issu des étapes précédentes, (c) si une étape de validation HITL doit être insérée.


## 4.5 Définition des tâches CrewAI

Trois tâches sont déclarées dans `config/tasks.yaml`, dans l'ordre exécutionnel suivant :

1. **`market_research_task`** — assignée à l'agent Market Research. Description : « Cartographier la concurrence sur la niche \<niche\>, identifier 3 à 5 acteurs principaux, leur positionnement prix, et 2 à 3 tendances récentes. » Format de sortie attendu : Markdown structuré (sections concurrents, prix, tendances).
2. **`sentiment_analysis_task`** — assignée à l'agent Sentiment Analyst. Description : « Charger les avis filtrés sur la niche \<niche\>, exécuter la classification RoBERTa, agréger la distribution des sentiments et extraire les 5 thèmes récurrents les plus saillants. » Format de sortie : JSON avec `total_reviews`, `distribution`, `themes_positifs`, `themes_negatifs`, `count_uncertain`.
3. **`synthesis_task`** — assignée au manager, avec `human_input=True`. Description : « Synthétiser les sorties des deux tâches précédentes en un rapport d'intelligence de marché de 600 à 900 mots, incluant un encadré 'recommandations actionnables'. » Le `human_input=True` déclenche le checkpoint HITL décrit en §5.

Chaque tâche dispose d'un `expected_output` explicite, ce qui aide le LLM à structurer sa réponse et facilite la validation par le manager.

## 4.6 Outils transverses : journalisation et configuration

Tous les agents partagent un journal d'exécution unique implémenté par la classe `AgentLogger` (`src/utils/logger.py`). Le logger émet huit types d'événements (`crew_start`, `agent_start`, `tool_call_start`, `tool_call_end`, `agent_finish`, `human_input_request`, `human_input_response`, `crew_end`) au format JSON Lines, avec un horodatage ISO 8601 et un identifiant de session. Le fichier de journal est créé dans `logs/agent_<timestamp>.jsonl`. Le logger est conçu thread-safe (verrou interne sur l'écriture) afin de tolérer les exécutions concurrentes que CrewAI peut déclencher en mode hiérarchique.

La configuration centrale est concentrée dans `config/settings.py`, qui expose les chemins du dataset, le seuil de confiance du sentiment, le nombre maximum de retries du web_search, et le nom du modèle Gemini. Toute modification de comportement passe par ce module unique, évitant les constantes magiques disséminées dans le code.

## 4.7 Flux d'exécution complet

Lorsque l'utilisateur lance `python -m src.main --niche "wireless earbuds"`, le flux suivant s'enchaîne :

1. Initialisation : `AgentLogger` crée un nouveau fichier de log, `Crew` est instancié avec les agents et tâches définis.
2. Manager planifie : choisit `market_research_task` comme première tâche.
3. Market Research Agent appelle `web_search_tool` une à trois fois, agrège les résultats, retourne sa sortie au manager.
4. Manager planifie : choisit `sentiment_analysis_task`.
5. Sentiment Analyst Agent appelle `review_loader_tool` (1 fois) puis `sentiment_tool` (1 fois sur l'ensemble des avis chargés), retourne la sortie agrégée.
6. Manager appelle `synthesis_task` avec `human_input=True`. Pause HITL : l'utilisateur valide ou commente.
7. Manager produit la synthèse finale, sauvegardée dans `outputs/<niche>_<timestamp>.md`.
8. Logger émet `crew_end` ; le programme se termine en code 0.

Ce flux concrétise la spécification : trois composantes, deux agents spécialisés, un orchestrateur, un mécanisme HITL, et une journalisation traçable de bout en bout.
