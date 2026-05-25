# 9. Plan de tests

## 9.1 Stratégie globale

La stratégie de validation s'organise sur trois niveaux complémentaires : (i) tests unitaires sur chaque outil et utilitaire, en isolation, avec dépendances externes simulées (mocks) ; (ii) tests structurels sur la cohérence des configurations YAML et l'intégrité du câblage des agents ; (iii) tests d'intégration end-to-end sur le pipeline complet, exécutés avec une clé API Gemini active. À l'état actuel, les niveaux (i) et (ii) sont automatisés via pytest et exécutés à chaque commit ; le niveau (iii) est planifié pour la phase A9 et exécuté manuellement par l'opérateur.

Le framework retenu est **pytest 8.x**. Les mocks reposent sur `unittest.mock` (bibliothèque standard) plus quelques helpers spécifiques (`_make_stub_model` qui crée un modèle DistilBERT fictif renvoyant des logits contrôlés). Aucune dépendance externe (Gemini, DuckDuckGo, GPU) n'est requise pour exécuter les tests des niveaux (i) et (ii).

L'exécution se fait via :

```
PYTHONPATH=. python -m pytest tests/ -q
```

et produit le résultat **30 tests passants** (1 test smoke supplémentaire est gated par le marker `@pytest.mark.smoke` opt-in, afin de ne pas consommer le quota Gemini à chaque exécution).

## 9.2 Inventaire des tests

| ID | Fichier | Couvre | Stratégie |
|---|---|---|---|
| T01 | `test_crew_smoke.py` | Câblage CrewAI hiérarchique + appel Gemini réel | E2E live |
| T02-T06 | `test_web_search_tool.py` | DDGS rate limit, backoff exponentiel, erreur réseau, format de retour | Unit + mocks |
| T07-T13 | `test_sentiment_tool.py` | Scores POSITIVE haute confiance, NEUTRAL faible (UNCERTAIN), NEGATIVE haute confiance, entrée vide, whitespace, schéma de sortie, somme softmax ≈ 1 | Unit + mocks |
| T14-T17 | `test_review_loader_tool.py` | Chargement CSV valide, catégorie inconnue, fichier absent, max_reviews respecté | Unit + mocks |
| T18-T22 | `test_logger.py` | 8 méthodes événement, format JSONL, thread-safety (50 écritures concurrentes), encoding UTF-8 | Unit + threading stress |
| T23 | `test_a7_structure.py::test_agents_yaml_defines_three_roles` | YAML agents définit 3 rôles avec role/goal/backstory | Structurel |
| T24 | `test_a7_structure.py::test_tasks_yaml_defines_three_tasks_in_order` | YAML tasks définit 3 tâches, synthesis_task a `human_input=True`, chaque tâche a description+expected_output | Structurel |
| T25 | `test_a7_structure.py::test_agent_factories_importable` | Les 3 factories Python s'importent sans erreur | Structurel |
| T26 | `test_a7_structure.py::test_main_module_exposes_full_pipeline_entrypoints` | `src.main` expose `run_smoke_crew`, `run_market_intelligence_crew`, `main` | Structurel |
| T27 | `test_a7_structure.py::test_market_research_task_references_niche_placeholder` | La description de market_research_task contient `{niche}` pour interpolation | Structurel |
| T28 | `test_a7_structure.py::test_sentiment_task_has_market_context_dependency` | sentiment_analysis_task référence market_research_task dans context_tasks | Structurel |
| T29 | `test_a7_structure.py::test_set_logger_callable_on_each_tool` | Les 3 modules outil exposent `set_logger()` callable | Structurel |

Total : 30 tests automatisés (passants sans clé Gemini active ; 1 test smoke supplémentaire gated par marker `@pytest.mark.smoke`).

## 9.3 Couverture critique des outils

### 9.3.1 `web_search_tool` (5 tests)

Les tests T02-T06 vérifient le comportement de la couche DuckDuckGo via patching de la classe `DDGS`. Le rate limit (1 requête / 2 secondes) est validé par mesure du `time.sleep` interne ; le backoff exponentiel est validé en levant `RatelimitException` deux fois consécutivement et en confirmant l'attente cumulée 2+4+8 = 14 secondes. Les erreurs imprévues retournent un dict avec champ `error` au lieu de propager une exception.

### 9.3.2 `sentiment_tool` (7 tests)

Les tests T07-T13 couvrent l'ensemble du chemin d'inférence sans charger le vrai modèle de sentiment (RoBERTa). La fonction utilitaire `_make_stub_model(scores)` produit un MagicMock dont la sortie `.logits` est telle que `softmax(logits) == scores` à epsilon près, permettant de tester la logique de classification avec des distributions de probabilité contrôlées. La règle UNCERTAIN (confidence < 0.6) est validée par T08, le schéma de sortie complet par T12, et la somme softmax ≈ 1 par T13.

### 9.3.3 `review_loader_tool` (4 tests)

Les tests T14-T17 utilisent le sample CSV `data/sample/electronics_100.csv` (force-add documenté en D04). La résolution lazy de `DATA_PATH` est vérifiée en passant des variables d'environnement temporaires : T17 confirme qu'un changement post-import est respecté.

### 9.3.4 `AgentLogger` (5 tests)

Les tests T18-T22 valident les 8 méthodes événement et le format JSONL. Le test de stress (T22) lance 50 threads écrivant chacun 10 événements pour confirmer l'absence de race condition ; la sortie attendue est exactement 500 lignes JSONL valides, sans interleaving partiel.

## 9.4 Tests structurels A7 (7 tests)

Les tests T23-T29 ont été ajoutés lors de la phase A7 pour garantir que la couche d'orchestration est correctement câblée avant tout run live. Ils ne nécessitent ni Gemini ni GPU. Leur valeur est de détecter immédiatement toute dérive entre les configurations YAML et le code Python (par exemple, ajout d'un nouvel agent dans le YAML sans factory correspondante, ou suppression accidentelle d'une tâche). Ils sont rapides (~0.5 seconde au total) et s'exécutent à chaque commit.

## 9.5 Tests d'intégration end-to-end (manuel — niveau iii)

Les tests E2E exigent une clé Gemini active et seront exécutés par l'opérateur après rotation de la clé courante (R06). La procédure est la suivante :

| Étape | Commande | Résultat attendu |
|---|---|---|
| Validation clé | `python -m src.main --validate-key` | « Gemini key OK. » |
| Smoke test | `python -m src.main --smoke` | Sortie de greeter, ~2 secondes |
| Pipeline complet | `python -m src.main --niche "wireless earbuds" --category Electronics` | Rapport markdown dans `outputs/`, log dans `logs/` |
| Niche secondaire | `python -m src.main --niche "argan oil serums" --category All_Beauty` | Comme ci-dessus, niche différente |
| Test smoke pytest | `pytest tests/test_crew_smoke.py -v` | 1 test passant (29/29 total) |

L'objectif est d'au moins 3 niches testées avant la soutenance, dont 2 niches non vues pendant le développement, pour démontrer la robustesse à la diversité des requêtes.

## 9.6 Métriques de qualité

| Métrique | Valeur courante | Cible avant soutenance |
|---|---|---|
| Tests unitaires + structurels passants | 28 / 29 | 29 / 29 (après rotation clé) |
| Couverture des outils | 100 % des fonctions publiques | inchangé |
| Couverture du pipeline (E2E) | 0 / 5 niches | ≥ 3 / 5 niches |
| Temps total de la suite pytest | ~8 secondes | < 30 secondes |
| Faux positifs (tests flaky) | 0 | 0 |

## 9.7 Limitations connues

Trois limitations sont documentées :

1. **Pas de tests E2E automatisés** : le quota gratuit Gemini (20 requêtes/jour) ne permet pas d'inclure un E2E dans la suite pytest sans épuiser le quota. C'est une limitation pratique, pas conceptuelle ; un quota payant ou un mock LLM enregistré (cassette) lèverait la limitation.

2. **Pas de test de la classe NEUTRAL en condition réelle** : le test `test_neutral_low_confidence_uncertain` simule un cas NEUTRAL avec scores contrôlés ; il ne mesure pas la performance du vrai modèle sur de vrais avis 3 étoiles. Cette mesure est faite en évaluation hors pytest (`evaluate.py`), pas en test unitaire.

3. **Tests structurels n'attrapent pas les erreurs de prompt** : ils valident la présence des champs YAML mais pas leur efficacité (par exemple, un backstory de manager qui ne forcerait plus la délégation). La détection nécessite un E2E live.
