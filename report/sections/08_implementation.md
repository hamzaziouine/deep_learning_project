# 8. Implémentation

## 8.1 Vue d'ensemble du dépôt

Le code source du projet est organisé en sept dossiers principaux, chacun avec une responsabilité unique. Cette structure découple les configurations (modifiables sans toucher au code), les outils (chacun isolé dans un fichier), les agents (factories paramétrées), le modèle d'apprentissage profond (entraînement, évaluation, inférence), les utilitaires partagés, et les tests.

```
config/         settings.py + agents.yaml + tasks.yaml
src/agents/     market_researcher.py, sentiment_analyst.py, manager.py
src/tools/      web_search_tool.py, sentiment_tool.py, review_loader_tool.py
src/models/     train.py, evaluate.py, predict.py
src/utils/      logger.py
src/main.py     CLI et orchestration de la pipeline complète
tests/          30 tests pytest (unitaires, structurels, smoke E2E)
```

Le dépôt comprend également `data/` (scripts de téléchargement et de prétraitement, plus splits sauvegardés), `report/` (sections du rapport, diagrammes UML, cheat sheets de soutenance), `docs/` (rapports de spike, évaluation, comparaisons d'ablation, revue de council), `models/` (checkpoint du modèle entraîné, gitignored), et `outputs/` / `logs/` (artefacts d'exécution, gitignored).

## 8.2 Configuration externalisée (YAML)

Deux fichiers YAML constituent la configuration du système. `config/agents.yaml` définit trois rôles (`market_researcher`, `sentiment_analyst`, `manager`) avec pour chacun les champs `role`, `goal`, `backstory`, `allow_delegation`, `verbose`, et `max_iter`. Le manager est volontairement isolé dans une entrée séparée car il sera passé à CrewAI via le paramètre `manager_agent=` (et non `manager_llm=`), pour garantir que la backstory explicite « never answer questions yourself » est appliquée.

`config/tasks.yaml` définit trois tâches dans l'ordre exécutionnel : `market_research_task`, `sentiment_analysis_task` (avec `context_tasks: [market_research_task]` pour passer la sortie du premier en contexte au second), et `synthesis_task` (avec `human_input: true` pour déclencher le checkpoint HITL natif de CrewAI). Chaque tâche déclare `description`, `expected_output` et `agent`.

Le choix d'externaliser dans YAML plutôt que de hardcoder dans le code Python répond à deux contraintes : (i) la spec encourage la séparation config/code pour faciliter la modification de comportement sans recompilation ; (ii) l'externalisation permet aux trois sections (agents, tasks, paramètres globaux) d'être versionnées indépendamment et auditées par le superviseur.

## 8.3 Factory pattern pour les agents

Les fichiers `src/agents/{market_researcher,sentiment_analyst,manager}.py` exposent chacun une fonction factory `build_*(llm) -> Agent`. La factory charge la configuration YAML correspondante, instancie l'objet `crewai.Agent` avec les paramètres et la liste d'outils appropriés, et retourne l'agent prêt à l'emploi. Ce pattern présente trois avantages :

1. **Testabilité** : on peut instancier un agent en test unitaire avec un mock LLM sans toucher à l'orchestrateur.
2. **Réutilisabilité** : si le projet évolue vers d'autres workflows (par exemple un crew séquentiel pour un usage simplifié), la même factory peut être appelée depuis plusieurs entry points.
3. **Cohérence avec le YAML** : toute modification du backstory ou des paramètres `max_iter` se fait dans le YAML, sans toucher au code Python.

L'agent Market Research reçoit `tools=[web_search_tool.search_market]`, l'agent Sentiment Analyst reçoit `tools=[review_loader_tool.load_product_reviews, sentiment_tool.analyze_sentiment]`, et le manager reçoit `tools=[]` (il ne consomme aucun outil — il délègue).

## 8.4 Outils CrewAI

Les trois outils sont décorés avec `@tool("nom_logique")` du module `crewai.tools` et exposent une fonction Python prenant des arguments simples et retournant un dictionnaire structuré. Cette signature est consommée directement par les agents : le LLM appelle l'outil par son nom logique, CrewAI parse les arguments depuis la sortie LLM, exécute la fonction Python, et passe le résultat (stringifié) à l'agent comme observation.

**Web Search Tool** (`web_search_tool.py`) — encapsule l'API DDGS avec trois protections : `_MIN_INTERVAL_S = 2.0` (pause forcée entre requêtes), `_BACKOFF_DELAYS = (2, 4, 8)` (retry exponentiel sur `RatelimitException`), et un wrapping try/except qui retourne un dict d'erreur plutôt que de propager. La fonction `set_logger(agent_logger)` est exposée pour permettre à `src.main` d'injecter l'instance partagée d'`AgentLogger` au démarrage du crew.

**Sentiment Tool** (`sentiment_tool.py`) — encapsule l'inférence RoBERTa-base avec un cache module-level (lazy load) et une logique de seuil de confiance τ=0.6 produisant l'étiquette `UNCERTAIN`. La fonction `_load_model()` détecte CUDA et déplace le modèle sur GPU si disponible (renforcement A7 du council day 3). Au call-time, les inputs tokenisés sont déplacés vers le device avant `model.forward()`.

**Review Loader Tool** (`review_loader_tool.py`) — charge un CSV d'avis filtrés par catégorie. La résolution du chemin `DATA_PATH` est faite au call-time, pas à l'import, car CrewAI 1.14.4 appelle `load_dotenv()` automatiquement au boot et peut écraser une valeur figée à l'import.

## 8.5 Pipeline de la classe `AgentLogger`

`src/utils/logger.py` implémente une classe singleton-style instanciée une fois par exécution dans `src/main.py`. Elle expose huit méthodes (`crew_start`, `task_delegated`, `tool_called`, `tool_error`, `hitl_checkpoint`, `agent_response`, `report_generated`, `crew_complete`) qui écrivent chacune une ligne JSON valide dans le fichier `logs/agent_<timestamp>.jsonl`. Un `threading.Lock` protège l'accès concurrent (deux agents pouvant loguer simultanément en exécution hiérarchique).

## 8.6 Entry point `src/main.py`

Le module principal expose trois entry points distincts, sélectionnables par CLI :

| CLI | Fonction appelée | Utilité |
|---|---|---|
| `--smoke` | `run_smoke_crew()` | Test rapide d'un agent unique pour valider la connexion à Gemini |
| `--validate-key` | `_validate_gemini_key()` | Préflight : un appel `models.list()` qui échoue immédiatement si la clé est invalide |
| `--niche "<niche>"` | `run_market_intelligence_crew(niche, category)` | Pipeline complète |

Au démarrage, le module charge `.env`, synchronise `GEMINI_API_KEY` et `GOOGLE_API_KEY` (le SDK google-genai consomme les deux et avertit en cas de divergence), et active `LITELLM_CACHE=disk` pour que les requêtes LLM identiques soient mises en cache (résilience au quota gratuit de 20 req/jour).

La fonction `run_market_intelligence_crew()` réalise les étapes suivantes :

1. Crée le fichier de log `logs/agent_<timestamp>.jsonl`.
2. Instancie l'`AgentLogger` et l'injecte dans les trois outils via `set_logger()`.
3. Instancie le LLM Gemini.
4. Construit les trois agents via leurs factories.
5. Charge les définitions de tâches du YAML et matérialise trois objets `Task` avec interpolation `{niche}` et `{category}`.
6. Instancie un objet `Crew(process=Process.hierarchical, manager_agent=manager)`.
7. Appelle `crew.kickoff()` qui exécute la pipeline avec délégations dynamiques.
8. Écrit le rapport final en Markdown dans `outputs/<niche>_<timestamp>.md`.
9. Émet les événements `report_generated` et `crew_complete` dans le log.

## 8.7 Pipeline d'apprentissage profond

Trois scripts coexistent dans `src/models/` :

**`train.py`** réalise le fine-tuning. Il charge les CSV de `data/processed/`, tokenise via `AutoTokenizer`, instancie un modèle via `AutoModelForSequenceClassification` (donc model-agnostic — `roberta-base`, `distilbert-base-uncased`, `microsoft/deberta-v3-base` fonctionnent tous via le même chemin), configure le `Trainer` HuggingFace avec early stopping et FP16, puis sauvegarde le checkpoint `eval_loss`-minimum dans `models/sentiment_<arch>/`. Une sous-classe `WeightedTrainer` permet de passer des poids de classe via le CLI `--class-weights "1.0,2.5,1.0"` (utilisée pour l'ablation négative documentée en §7.5).

**`evaluate.py`** charge un checkpoint et évalue sur le jeu de test. Il produit `docs/evaluation.md` (métriques, matrice de confusion, exemples corrects et erronés) plus trois figures PNG (matrice de confusion normalisée, métriques par classe, courbes d'entraînement). Le script accepte `--model-dir` pour évaluer un checkpoint quelconque, ce qui a été utilisé pour l'ablation comparative DistilBERT vs RoBERTa.

**`predict.py`** expose `predict_sentiment(text)` pour usage programmatique en dehors de la pipeline CrewAI. Il maintient un cache thread-safe singleton et déplace correctement le modèle sur GPU. Cette fonction n'est pas appelée par le pipeline (qui passe par `sentiment_tool.analyze_sentiment` côté CrewAI) mais reste disponible pour les notebooks d'exploration et les scripts ad hoc.

## 8.8 Reproductibilité

Trois mécanismes garantissent la reproductibilité :

1. **Graine 42** fixée dans `train.py` pour `random`, `numpy`, `torch`, et `cudnn.deterministic=True` (au coût d'un léger ralentissement, jugé acceptable).
2. **Liste des `review_id`** retenus dans chaque split sauvegardée dans `data/processed/split_indices.json` (4 MB, committé), permettant de re-générer exactement les mêmes CSV à partir du dataset brut.
3. **Versions épinglées** dans `requirements.txt` pour les dépendances critiques (`crewai==1.14.4`, `torch==2.6.0+cu124`, `transformers`).

L'exécution de `train.py` puis `evaluate.py` sur la même machine doit reproduire 80.83 % à ±0.1 pp.

## 8.9 Choix de robustesse

Plusieurs décisions de robustesse méritent d'être nommées :

- **Caching LLM disque** (`LITELLM_CACHE=disk`) : élimine les appels Gemini redondants, critique pour le quota 20 req/jour.
- **Synchronisation GEMINI_API_KEY ↔ GOOGLE_API_KEY** : le SDK google-genai consomme les deux. Le main.py les met automatiquement en miroir au démarrage pour éviter une erreur silencieuse.
- **Chargement YAML lazy** : les YAML sont relus à chaque appel de factory plutôt que cachés au boot, ce qui permet de modifier un backstory sans redémarrer un service.
- **Tools logger-injectable** : chaque outil expose `set_logger()` pour permettre à `main.py` de partager l'instance d'`AgentLogger`. Sans `set_logger()`, l'outil fonctionne mais ne loggue pas.
- **Validation préflight** : `--validate-key` permet de détecter une clé révoquée ou un quota épuisé avant de lancer un run complet.
