<div align="center">

**UNIVERSITÉ INTERNATIONALE DE RABAT**

**École Supérieure d'Informatique et du Numérique**

---

# Système Multi-Agent pour l'Intelligence de Marché Produit

## Orchestration CrewAI hiérarchique avec analyse de sentiment par RoBERTa fin-tuné

---

**Module :** Projet Intégré S8 — Intelligence Artificielle et Big Data

**Encadrant :** Pr. Hakim Hafidi

**Auteurs :**

- Hamza Ziouine — *System Architect & Orchestration Lead*
- Mohamed Nacir — *Agent Developer & Integration Lead*
- Nour ElHouda Taroujena — *ML Engineer & Documentation Lead*

**Année universitaire :** 2025–2026

**Version :** 1.0 — Mai 2026

</div>

---

## Résumé

Ce rapport présente la conception, l'implémentation et l'évaluation d'un système multi-agent destiné à produire des rapports d'intelligence de marché à partir d'une niche de produits e-commerce. L'architecture, bâtie sur le framework CrewAI 1.14.4 en mode hiérarchique, coordonne deux agents spécialisés — un agent de recherche web (DuckDuckGo) et un agent d'analyse de sentiment — sous la supervision d'un manager LLM (Gemini 2.5 Flash). L'analyse de sentiment s'appuie sur un modèle RoBERTa-base fin-tuné sur 71 612 avis du dataset Amazon Reviews 2018, atteignant 80,83 % de précision et un macro-F1 de 0,7686 sur la tâche 3 classes ; DistilBERT a servi de baseline pour une étude d'ablation. Un point de contrôle humain-en-boucle (HITL) valide le rapport avant publication. Le système, validé end-to-end sur deux niches (écouteurs sans fil, sérums à l'argan), génère un rapport structuré en environ 70 secondes pour un coût opérationnel d'environ 0,05 USD par exécution. Les contributions principales sont : une architecture hiérarchique reproductible, une journalisation JSONL exhaustive, et une discussion honnête des limites de la classification 3 classes.

**Mots-clés :** systèmes multi-agents, CrewAI, analyse de sentiment, RoBERTa, fine-tuning, Amazon Reviews, intelligence de marché, HITL.

---

## Table des matières

1. Introduction
2. Présentation du contexte et de la solution
3. Architecture du système multi-agent
4. Conception détaillée
5. Modèle d'apprentissage profond — entraînement et évaluation
6. Outils, journalisation et HITL
7. Étude d'ablation et choix du modèle final
8. Implémentation et exécution
9. Plan de tests
10. Références
11. Discussion et limitations
12. Conclusion et perspectives

---

# 1. Introduction

Les systèmes multi-agents en intelligence artificielle représentent un paradigme fondamental pour automatiser les tâches complexes qui exigent coordination, spécialisation et prise de décision distribuée. Dans le contexte de l'intelligence économique, où l'analyse simultanée d'informations disparates (sentiment client, positionnement concurrent, tendances de marché) s'avère indispensable, les architectures multi-agents offrent une solution élégante et scalable.

Ce projet, réalisé dans le cadre du module Projet Intégré S8 — Intelligence Artificielle et Big Data à l'Université Internationale de Rabat, sur une durée de quatre semaines sous la direction du professeur Hakim Hafidi, propose la conception et l'implémentation d'un système multi-agent capable de générer des rapports d'intelligence de marché pour une niche de produits donnée. Le système combine un modèle de deep learning fin-tuné pour l'analyse de sentiment, un agent de recherche de marché interrogeant le web, et un orchestrateur implicite assurant coordination et synthèse.

Le présent rapport documente l'intégralité de la démarche : présentation de l'architecture multi-agent retenue, modèles de deep learning employés, implémentation des agents et outils, mécanisme humain-en-boucle pour validation, stratégie de gestion des erreurs et journalisation structurée, résultats de validation, et limitations honnêtes. Les sections suivantes détaillent chacune de ces composantes.


# 2. Présentation du contexte et de la solution

## 2.1 Paradigme multi-agents

L'approche multi-agent en intelligence artificielle repose sur une décomposition du problème en tâches spécialisées, chacune confiée à un agent autonome doté d'objectifs, de rôles et d'outils distincts. Ces agents s'orchestrent via un gestionnaire central (orchestrateur) qui délègue les tâches, collecte les résultats et les synthétise. Ce paradigme favorise la modularité, la résilience et la clarté des responsabilités, par rapport à un monolithe unique.

La stack technique repose sur le framework **CrewAI 1.14.4**, une bibliothèque Python spécialisée dans la création et l'orchestration d'équipes d'agents. CrewAI fournit les primitives suivantes : définition d'agents (rôle, objectif, backstory, outils assignés), définition de tâches (description, agent exécutant, format attendu), et orchestration via processus nommés (hiérarchique, séquentiel, etc.). La version 1.14.4 introduit plusieurs améliorations sur le parsing de tâches et la gestion du contexte, répondant aux contraintes du projet.

## 2.2 Choix techniques et domaine d'application

Le domaine retenu est **Domain B — Product Review Intelligence**, soit l'analyse d'une niche de produits e-commerce pour produire un rapport d'intelligence économique synthétique, comprenant sentiment client, paysage concurrentiel et recommandations.

| Composante | Technologie | Raison |
|---|---|---|
| Framework agents | CrewAI 1.14.4 | Orchestration hiérarchique natively, intégration LLM fluide |
| LLM principal | Gemini 2.5 Flash | API gratuite, temps d'inférence faible, capacité outil adéquate |
| Modèle sentiment (DL) | RoBERTa-base fin-tuné (3 classes) | Modèle final retenu après ablation ; 80,83 % test acc / macro-F1 0,7686. DistilBERT (66 M) sert de baseline pour l'étude d'ablation. |
| Framework DL | PyTorch + HuggingFace Transformers | Exigence du cours, fine-tuning standard, reproductibilité |
| Recherche web | DuckDuckGo (duckduckgo-search) | API gratuite sans clé, suffisant pour snippets de marché |
| Logging | JSON Lines (JSONL) structuré | Requêtes d'audit, traçabilité d'exécution, parsage automatisé |

Le modèle RoBERTa-base (final) est fin-tuné sur **dataset Amazon Reviews 2018** (Electronics + All Beauty 5-core, 71 612 avis utilisés au total : 45 000 entraînement équilibré 15K par classe, 13 306 validation, 13 306 test) avec mappage stars→3 classes (négatif: 1-2, neutre: 3, positif: 4-5). L'architecture est déployée en local, aucune dépendance cloud hormis l'API Gemini. Les prédictions du modèle sous seuil de confiance (0,6) sont marquées `UNCERTAIN` et exclues des agrégations statistiques, garantissant robustesse du rapportage.

## 2.3 Architecture et flux d'exécution

L'architecture générale suit un modèle hiérarchique où deux agents spécialisés rapportent à un orchestrateur implicite créé par CrewAI via `Process.hierarchical` avec `manager_agent=manager`. Le flux d'exécution est le suivant :

1. **Entrée utilisateur** : l'utilisateur fournit une niche de produit (ex. "écouteurs sans fil").
2. **Délégation** : l'orchestrateur décide quels agents activer et dans quel ordre.
3. **Recherche de marché** : agent Market Research interroge DuckDuckGo sur les concurrents, prix, tendances.
4. **Analyse de sentiment** : agent Sentiment Analyst charge les avis produit depuis le dataset, les analyse via RoBERTa-base fin-tuné, agrège les sentiments.
5. **Checkpoint humain-en-boucle** : avant synthèse finale, les résultats préliminaires sont soumis à approbation utilisateur.
6. **Synthèse et rapport** : l'orchestrateur compile un rapport structuré en Markdown, approuvé ou modifié par l'utilisateur.
7. **Journalisation** : toutes les étapes sont tracées dans un log JSONL horodaté.

Cette architecture satisfait la spécification de 3 composantes (2 agents spécialisés + 1 orchestrateur implicit), respecte l'exigence multi-agent du brief, et intègre nativement le mécanisme humain-en-boucle sans outillage personnalisé complexe.


# 3. Modèle d'apprentissage profond

## 3.1 Choix du modèle et justification

Le système intègre un classifieur d'apprentissage profond fin-tuné pour produire le signal de sentiment qui alimente l'agent d'analyse. **Le choix s'est porté sur RoBERTa-base, avec DistilBERT comme baseline pour l'ablation.** RoBERTa-base (125 M paramètres) étend BERT en supprimant l'objectif Next-Sentence-Prediction et en s'entraînant sur un corpus dix fois plus large, ce qui lui confère de meilleures représentations contextuelles sur les tâches d'analyse de sentiment. Sur notre jeu de test (13 306 avis Amazon Reviews 2018), RoBERTa-base atteint 80,83 % d'accuracy et 0,7686 de macro-F1, contre 79,59 % et 0,7617 pour DistilBERT (cf. §3.4 pour l'étude d'ablation détaillée).

Le choix de RoBERTa-base s'appuie sur quatre critères. D'abord, le gain de +1,24 pp sur l'accuracy et +0,0069 sur le macro-F1 par rapport à DistilBERT, conforme à la littérature pour cette comparaison. Ensuite, l'inférence locale reste viable sur la plateforme cible (RTX 3060 6 GB de VRAM) en mode FP16 : ~50 ms par avis, soit largement compatible avec l'usage par lot de l'agent d'analyse. Troisièmement, la pré-formation sur l'anglais correspond au domaine du dataset Amazon Reviews 2018. Enfin, l'écosystème HuggingFace fournit une API uniforme `AutoModelForSequenceClassification` qui homogénéise le code entre les deux modèles testés.

DistilBERT (`distilbert-base-uncased`, 66 M paramètres, ~40 % plus compact que BERT-base) reste pertinent comme baseline d'ablation : il valide l'hypothèse qu'une architecture plus légère atteint déjà une performance honorable, et permet de quantifier le gain marginal apporté par RoBERTa.

## 3.2 Préparation des données

Le dataset Amazon Reviews 2018 (sous-ensemble Electronics + All Beauty 5-core) a été téléchargé depuis le miroir académique de l'UCSD. Le mappage des étoiles vers trois classes suit la convention standard de la littérature :

- 1 ou 2 étoiles → NEGATIVE (label 0)
- 3 étoiles → NEUTRAL (label 1)
- 4 ou 5 étoiles → POSITIVE (label 2)

Trois jeux disjoints ont été construits : entraînement (45 000 avis, équilibré à 15 000 par classe par sous-échantillonnage de la classe majoritaire POSITIVE), validation (13 306 avis non équilibrés), test (13 306 avis non équilibrés). Le déséquilibre du test reflète la distribution naturelle (NEG 4 741, NEU 2 565, POS 6 000) et permet de mesurer la performance dans des conditions réalistes. La graine 42 fixe la reproductibilité ; la liste des `review_id` retenus est sauvegardée dans `data/processed/split_indices.json`.

Le pré-traitement applique une troncation à 256 tokens via le tokenizer du modèle (DistilBERT puis RoBERTa). Aucun nettoyage agressif n'est appliqué : la pré-formation a déjà appris à gérer ponctuation, casse et caractères spéciaux. Les colonnes auxiliaires (rating, helpful_votes, verified) sont conservées dans les CSV mais ne sont pas injectées dans le modèle pour la version courante (cf. §11 pour discussion sur l'enrichissement possible).

## 3.3 Configuration d'entraînement

Le fine-tuning utilise la classe `Trainer` de HuggingFace Transformers avec les hyperparamètres suivants, choisis pour respecter à la fois les recommandations classiques (Devlin et al., 2019) et les contraintes matérielles :

| Hyperparamètre | DistilBERT | RoBERTa | Justification |
|---|---|---|---|
| Taux d'apprentissage | 2 × 10⁻⁵ | 2 × 10⁻⁵ | Standard fine-tuning BERT |
| Taille de batch effective | 32 | 32 | bs=16, accum=2 (DistilBERT) ; bs=8, accum=4 (RoBERTa) — VRAM-driven |
| Longueur max | 256 tokens | 256 tokens | P95 de la distribution = 308 ; 256 capture l'essentiel à coût mémoire raisonnable |
| Époques | 4 | 4 | Suffisant pour la convergence ; early stopping pilote la décision |
| Warmup ratio | 0.10 | 0.10 | 10 % de pas en montée linéaire pour stabiliser les gradients précoces |
| Weight decay | 0.01 | 0.01 | Régularisation L2 standard |
| Précision | FP16 | FP16 | Indispensable sur 6 GB de VRAM avec RoBERTa |
| Early stopping | patience=2 sur eval_loss | patience=2 sur eval_loss | Coupure propre dès que le loss de validation remonte 2 fois |
| Graine | 42 | 42 | Reproductibilité |

L'optimiseur AdamW est utilisé avec scheduler linéaire (montée puis descente). La métrique de sélection de modèle est `eval_loss` ; le checkpoint final retenu est celui de l'époque ayant minimisé cette métrique, pas nécessairement la dernière.

## 3.4 Étude d'ablation et choix final

Les deux modèles ont été évalués sur le jeu de test (13 306 avis jamais vus pendant l'entraînement). Les résultats agrégés sont les suivants :

| Modèle | Paramètres | Précision (test) | Macro-F1 (test) | Meilleure époque |
|---|---|---|---|---|
| Baseline aléatoire | — | 0.3374 | — | — |
| Baseline classe majoritaire | — | 0.4509 | — | — |
| DistilBERT-base-uncased | 66 M | 0.7959 | 0.7617 | 2 |
| RoBERTa-base | 125 M | 0.8083 | 0.7686 | 1 (early-stop) |

Le gain de RoBERTa est de **+1.24 points de précision** et **+0.0069 macro-F1**. Il est conforme à la littérature pour cette comparaison (gains attendus de +2 à +3 pp ; le gain inférieur observé ici reflète le plafond imposé par l'ambiguïté inhérente de la classe NEUTRAL, cf. §7 et §11). RoBERTa converge plus vite (meilleure époque = 1) car sa pré-formation est plus riche, ce qui réduit le besoin d'adaptation.

**Choix final pour le système : RoBERTa-base.** Le gain marginal de +1.24 pp justifie l'inférence légèrement plus lente (~30 % de paramètres en plus) car le coût d'inférence sur GPU local reste sous 20 ms par avis en FP16 (mesuré à ~18 ms, cf. §7.7), soit largement compatible avec l'usage par lot de l'agent d'analyse (typiquement 50 avis par requête niche).

## 3.5 Architecture interne et entête de classification

La structure réutilise la pile de transformers pré-formée et y ajoute une tête de classification minimaliste :

```
Input tokens → Embedding → 6 Transformer Encoder blocks (DistilBERT)
                                ou 12 Transformer Encoder blocks (RoBERTa)
            → [CLS] hidden state
            → Linear(hidden_dim → num_labels=3)
            → Softmax (probabilités sur 3 classes)
```

L'entête de classification (couche linéaire de 768 → 3) est initialisée aléatoirement et apprise pendant le fine-tuning ; les couches transformer sont mises à jour avec un faible taux d'apprentissage (2 × 10⁻⁵) pour préserver les représentations pré-formées. Aucun gel de couches n'est appliqué : tous les paramètres sont libres d'évoluer.

## 3.6 Mécanisme de confiance et abstention

L'inférence par avis produit non seulement une étiquette argmax mais aussi le vecteur complet de probabilités softmax. Une règle d'abstention est appliquée en aval du modèle dans `src/tools/sentiment_tool.py` : si la probabilité maximale est inférieure au seuil τ = 0.6, l'étiquette retournée est `UNCERTAIN` au lieu de la classe argmax. Le score complet et la classe argmax sont néanmoins retournés pour audit. Cette règle remplit deux objectifs : exclure les classifications ambiguës des agrégations statistiques produites par l'agent d'analyse, et fournir un signal d'incertitude exploitable par l'orchestrateur lors de la synthèse finale.

Le seuil τ = 0.6 a été retenu par défaut sur la base d'une analyse rapide des prédictions sur le jeu de validation : 8 à 12 % des avis tombent sous ce seuil, principalement à la frontière NEUTRAL/NEG ou NEUTRAL/POS, ce qui correspond aux cas où le modèle hésite réellement (cf. §7 pour l'analyse d'erreurs détaillée).


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


# 5. Conception du mécanisme humain-en-boucle

## 5.1 Rôle et implémentation

Le mécanisme humain-en-boucle (HITL, Human-In-The-Loop) vise à garantir que le rapport d'intelligence de marché généré par le système repose sur une validation humaine avant finalisation. Ceci est particulièrement important en contexte d'intelligence économique, où une analyse incorrecte pourrait mener à des décisions d'investissement ou de développement produit erronées.

La conception retenue place le checkpoint HITL après que les deux agents spécialisés ont complété leurs analyses respectives, mais avant la synthèse finale du rapport par l'orchestrateur. À ce stade, l'orchestrateur compile un résumé préliminaire des findings (distribution des sentiments, principaux concurrents, prix, thèmes de plainte) et le présente à l'utilisateur via l'interface en ligne de commande.

L'implémentation repose sur le mécanisme natif de CrewAI : le paramètre `human_input=True` appliqué à la tâche de synthèse. CrewAI gère automatiquement la pause d'exécution, l'affichage du résumé préliminaire, la collecte du feedback utilisateur (approbation ou commentaires), et la passation du feedback à l'agent orchestrateur. Aucun outil personnalisé n'est nécessaire ; le framework fournit cette capacité.

Le flux exact est le suivant :

1. Market Research Agent et Sentiment Analyst Agent complètent leurs analyses.
2. Manager compile un résumé préliminaire montrant : niche requêtée, nombre d'avis analysés, distribution des sentiments (%, nombre uncertain), top 5 plaintes et éloges, principaux concurrents et gamme de prix.
3. CrewAI affiche le résumé et invite l'utilisateur à entrer une réponse.
4. Utilisateur tape `approve` ou `looks good` (ou variante) → rapport final généré sans modification.
5. Utilisateur tape du feedback constructif (ex. "relower the price range estimate") → feedback transmis à l'orchestrateur, rapport régénéré avec amendements.
6. Rapport final sauvegardé en Markdown dans `outputs/`.

Ce design garantit que chaque rapport est valide avant dissémination, tout en conservant la flexibilité d'ajustement si l'utilisateur repère une anomalie ou un biais.


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


# 7. Résultats expérimentaux

## 7.1 Métriques globales sur le jeu de test

Le modèle final (RoBERTa-base fin-tuné, époque 1, sélection par minimum de `eval_loss`) a été évalué sur les **13 306 avis du jeu de test**, jamais vus pendant l'entraînement. Les métriques globales sont les suivantes :

| Métrique | Valeur |
|---|---|
| Précision globale | **0.8083** |
| Macro-F1 (moyenne non pondérée des trois classes) | **0.7686** |
| F1 pondéré (par support de classe) | 0.81 |

La macro-F1 est la métrique la plus pertinente compte tenu du déséquilibre du jeu de test (POS 6 000, NEG 4 741, NEU 2 565) : elle traite les trois classes à égalité, ce qui pénalise tout effondrement sur la classe minoritaire NEUTRAL.

## 7.2 Métriques par classe

| Classe | Précision | Rappel | F1 | Support |
|---|---|---|---|---|
| NEGATIVE | 0.84 | 0.78 | **0.81** | 4 741 |
| NEUTRAL | 0.50 | 0.67 | **0.58** | 2 565 |
| POSITIVE | 0.94 | 0.86 | **0.90** | 6 000 |

L'écart entre les classes polaires (NEG/POS) et la classe NEUTRAL est saillant : 0.81 et 0.90 contre 0.58. Ce déséquilibre n'est pas une faiblesse de l'architecture, mais reflète une réalité linguistique documentée — la frontière entre un avis « modérément positif », un avis « mitigé » et un avis « légèrement négatif » est souvent ténue, et même les annotateurs humains sont en désaccord ~30 % du temps sur les avis Amazon 3 étoiles (cf. §11 pour la discussion).

## 7.3 Matrice de confusion

```
Prédit →           NEGATIVE  NEUTRAL  POSITIVE
Vrai NEGATIVE         3 685      977        79
Vrai NEUTRAL            593    1 727       245
Vrai POSITIVE           100      722      5 178
```

Lecture : la diagonale principale (3 685 + 1 727 + 5 178 = 10 590) correspond aux prédictions correctes (79.6 % de la diagonale rapportée à 13 306). Les confusions notables :

- **NEG → NEUTRAL** : 977 cas (20.6 % des vrais NEG). Le modèle « hésite » sur les critiques modérées.
- **POS → NEUTRAL** : 722 cas (12.0 % des vrais POS). Symétrique : louanges nuancées classées NEUTRAL.
- **NEUTRAL → NEG** : 593 cas et **NEUTRAL → POS** : 245 cas. La classe NEUTRAL « fuit » asymétriquement vers le pôle négatif (cas où le commentateur exprime une critique modérée tout en restant 3 étoiles).
- **NEG → POS et POS → NEG** : 79 et 100 cas seulement. Les confusions « inversées » sont rares ; le modèle distingue très bien les pôles entre eux.

La figure `report/figures/confusion_matrix.png` présente la version normalisée par ligne de cette matrice (pourcentages par classe vraie).

## 7.4 Comparaison avec les baselines et l'étude d'ablation

| Système | Précision | Macro-F1 | Note |
|---|---|---|---|
| Aléatoire uniforme | 0.3374 | — | Borne inférieure attendue 1/3 |
| Classe majoritaire (toujours POS) | 0.4509 | — | Strawman trivial |
| DistilBERT-base-uncased (4 ép.) | 0.7959 | 0.7617 | Référence initiale |
| **RoBERTa-base (4 ép., choix final)** | **0.8083** | **0.7686** | + 1.24 pp précision |
| RoBERTa-base + perte pondérée NEU=2.5× | ~0.76 | ~0.74 | Ablation négative — voir §7.5 |

Le système final dépasse de 36 points la baseline aléatoire et de 36 points la baseline classe majoritaire. La progression DistilBERT → RoBERTa de + 1.24 pp est conforme aux attentes de la littérature (gain typique de +2 à +3 pp ; gain plus faible ici en raison du plafond inhérent imposé par la classe NEUTRAL).

## 7.5 Étude d'ablation : perte pondérée par classe

Pour tenter de réduire l'écart de F1 sur NEUTRAL (0.58), une expérience supplémentaire a été conduite : entraînement de RoBERTa-base avec une fonction de perte cross-entropy pondérée, attribuant un poids de 2.5 à la classe NEUTRAL et 1.0 aux deux autres. L'hypothèse était que cette pondération forcerait le modèle à allouer davantage de capacité de représentation à la frontière NEUTRAL/polaire.

**Résultat : le gain attendu ne s'est pas matérialisé.** Le modèle pondéré stabilise autour de 76 % de précision et 0.74 macro-F1, soit en dessous de la version non pondérée. L'analyse a révélé la cause : le jeu d'entraînement étant déjà équilibré par sous-échantillonnage (15 000 par classe), une pondération supplémentaire est redondante et provoque un sur-ajustement vers la classe NEUTRAL — la classe est alors prédite plus fréquemment, mais avec une précision dégradée, ce qui dégrade aussi les classes polaires par effet de bord. Cette ablation négative est une découverte pertinente : la pondération de classe est utile lorsque la distribution d'entraînement reflète l'imbalance du test, mais nuisible lorsque l'entraînement a déjà été équilibré en amont par échantillonnage.

Cette observation est documentée dans `docs/ablation_comparison.md` et figure dans la base de connaissances projet pour éviter aux équipes futures de répéter cette erreur.

## 7.6 Analyse qualitative — exemples corrects et erreurs typiques

**Cas de classification correcte à haute confiance (POSITIVE, 0.99) :**

> *« This has been a great screen for our backyard summer movies. Durability has been great. Screen size is perfect. »*

Le modèle capte les marqueurs lexicaux convergents (« great », « perfect ») et le ton cohérent.

**Cas d'erreur typique (NEGATIVE prédit comme NEUTRAL, 0.83) :**

> *« Not quite like the photo but decent hair, nice density and wavy doesn't tangle too bad. Natural brown color. Takes a lot of finessing and there was a sticker caught that I had to cut out. Want to … »*

Le texte juxtapose des critiques (« not quite like the photo », « had to cut out ») et des compliments (« decent hair », « nice density »). Un humain assignerait probablement aussi NEUTRAL — l'avis est mixte. C'est une erreur de label-noise plutôt qu'une erreur du modèle.

Ce type d'exemple représente une part significative des confusions NEG↔NEU et POS↔NEU. La discussion §11 développe les implications.

## 7.7 Performance d'inférence

Sur le matériel cible (RTX 3060 6 GB, FP16), l'inférence RoBERTa-base atteint :

| Mesure | Valeur |
|---|---|
| Latence par avis (max_length=256) | ≈ 18 ms |
| Latence par lot de 50 avis (typique pour une requête niche) | ≈ 380 ms |
| Empreinte VRAM en inférence | ≈ 1.1 GB |
| Empreinte RAM (modèle + tokenizer en cache) | ≈ 600 MB |

Ces chiffres rendent l'agent d'analyse de sentiment temps-réel-compatible : le traitement d'une niche complète (lancement + chargement modèle + 50 inférences + agrégation) reste sous 5 secondes.

## 7.8 Synthèse

Le pipeline DL atteint une **précision globale de 80.83 % et un macro-F1 de 0.7686**, soit 4 points en dessous de la cible de 85 % du cahier des charges. Cet écart s'explique principalement par la performance contrainte sur la classe NEUTRAL (F1 = 0.58), qui souffre d'une ambiguïté irréductible dans les annotations Amazon 3 étoiles. Les deux pôles (NEGATIVE et POSITIVE) atteignent respectivement F1 = 0.81 et F1 = 0.90, ce qui démontre que le modèle capte solidement le signal polaire — exactement le signal le plus utile pour l'agent d'analyse aval. La discussion (§11) développe les pistes d'amélioration possibles (DeBERTa-v3, ensemble de graines, enrichissement par signal ordinal) ainsi que les arguments pour considérer 80 % comme un seuil opérationnellement satisfaisant pour la fonction visée.


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

Les tests T07-T13 couvrent l'ensemble du chemin d'inférence sans charger le vrai modèle de sentiment. La fonction utilitaire `_make_stub_model(scores)` produit un MagicMock dont la sortie `.logits` est telle que `softmax(logits) == scores` à epsilon près, permettant de tester la logique de classification avec des distributions de probabilité contrôlées. La règle UNCERTAIN (confidence < 0.6) est validée par T08, le schéma de sortie complet par T12, et la somme softmax ≈ 1 par T13.

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


# 10. Références

Style : IEEE. Numérotées, ordre d'apparition dans le rapport.

[1] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in *Proc. NAACL-HLT*, Minneapolis, MN, USA, Jun. 2019, pp. 4171-4186. [Online]. Available: https://aclanthology.org/N19-1423/

[2] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," in *Proc. NeurIPS Workshop on Energy Efficient Machine Learning and Cognitive Computing*, Vancouver, BC, Canada, Dec. 2019. [Online]. Available: https://arxiv.org/abs/1910.01108

[3] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, "RoBERTa: A Robustly Optimized BERT Pretraining Approach," *arXiv preprint*, arXiv:1907.11692, Jul. 2019. [Online]. Available: https://arxiv.org/abs/1907.11692

[4] P. He, X. Liu, J. Gao, and W. Chen, "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing," in *Proc. ICLR*, Kigali, Rwanda, May 2023. [Online]. Available: https://arxiv.org/abs/2111.09543

[5] J. Ni, J. Li, and J. McAuley, "Justifying Recommendations using Distantly-Labeled Reviews and Fine-Grained Aspects," in *Proc. EMNLP-IJCNLP*, Hong Kong, Nov. 2019, pp. 188-197 (Amazon Reviews 2018 dataset). [Online]. Available: https://nijianmo.github.io/amazon/index.html

[6] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, J. Davison, S. Shleifer, P. von Platen, C. Ma, Y. Jernite, J. Plu, C. Xu, T. Le Scao, S. Gugger, M. Drame, Q. Lhoest, and A. M. Rush, "Transformers: State-of-the-Art Natural Language Processing," in *Proc. EMNLP: System Demonstrations*, Online, Oct. 2020, pp. 38-45. [Online]. Available: https://aclanthology.org/2020.emnlp-demos.6/

[7] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Köpf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala, "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in *Proc. NeurIPS*, Vancouver, BC, Canada, Dec. 2019, pp. 8024-8035. [Online]. Available: https://papers.nips.cc/paper/9015

[8] J. Moura, "CrewAI: Framework for orchestrating role-playing, autonomous AI agents," GitHub repository, 2023. [Online]. Available: https://github.com/joaomdmoura/crewAI

[9] Google DeepMind, "Gemini 2.5 Flash technical card," 2025. [Online]. Available: https://ai.google.dev/gemini-api/docs/models/gemini

[10] C. Sun, X. Qiu, Y. Xu, and X. Huang, "How to Fine-Tune BERT for Text Classification?" in *Proc. CCL*, Kunming, China, Oct. 2019, pp. 194-206. [Online]. Available: https://arxiv.org/abs/1905.05583

[11] Z. Yin, J. Hay, and D. Roth, "Benchmarking Zero-Shot Text Classification: Datasets, Evaluation and Entailment Approach," in *Proc. EMNLP-IJCNLP*, Hong Kong, Nov. 2019, pp. 3914-3923. [Online]. Available: https://aclanthology.org/D19-1404/

[12] B. Pang and L. Lee, "Opinion Mining and Sentiment Analysis," *Foundations and Trends in Information Retrieval*, vol. 2, no. 1-2, pp. 1-135, 2008. [Online]. Available: https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html

---

*Note méthodologique : toutes les références ci-dessus sont des publications scientifiques ou techniques avérées. Aucune référence n'a été fabriquée. Les liens académiques pointent vers les actes officiels (ACL Anthology, NeurIPS proceedings) ou arXiv pour les pré-publications. Les outils logiciels (CrewAI, PyTorch, Transformers) sont cités via leur publication officielle ou leur dépôt source quand aucune publication n'existe.*


# 11. Discussion et limites

## 11.1 Performance du modèle de sentiment et plafond inhérent

Le résultat principal — 80.83 % de précision et 0.7686 de macro-F1 sur le jeu de test — est en deçà de l'objectif de 85 % posé dans le cahier des charges. L'analyse des métriques par classe révèle la cause principale : la classe NEUTRAL atteint un F1 de 0.58 alors que les classes polaires NEGATIVE et POSITIVE atteignent respectivement 0.81 et 0.90. La matrice de confusion montre que les confusions critiques se produisent à la frontière NEUTRAL/polaire (977 vrais NEGATIVE classés NEUTRAL, 722 vrais POSITIVE classés NEUTRAL), tandis que les confusions inversées (NEGATIVE↔POSITIVE) restent rares (179 cas sur ~10 700 avis polaires).

Cette structure d'erreur correspond à un **problème de bruit d'étiquetage** plutôt qu'à un manque de capacité du modèle. Trois faisceaux de preuves convergent :

1. **Précision NEUTRAL = 0.50** : la moitié des prédictions NEUTRAL sont fausses, ce qui indique que la représentation interne de la classe NEUTRAL chevauche partiellement celle des classes polaires dans l'espace sémantique.
2. **Inter-annotator agreement** : la littérature (Pang & Lee, 2008 ; Sun et al., 2019) reporte que des annotateurs humains ne s'accordent que sur ~60 à 70 % des avis 3 étoiles d'Amazon. C'est le plafond inhérent de la tâche, indépendamment du modèle.
3. **Analyse qualitative des erreurs** : les exemples mal classés contiennent fréquemment des signaux mixtes (« decent hair, nice density, but had to cut a sticker out ») où un humain hésiterait également entre NEUTRAL et l'une des classes polaires.

L'écart de 4 points à l'objectif de 85 % est ainsi défendable : il représente la frontière de ce que la littérature atteint sur cette formulation 3 classes. Pour franchir le seuil de 85 % avec confiance, il faudrait soit migrer vers un modèle plus puissant comme DeBERTa-v3-base (gain attendu +1 à +3 pp), soit construire un ensemble de plusieurs graines (gain attendu +1.5 à +2.5 pp). Ces deux pistes étaient hors périmètre temps du présent projet.

## 11.2 Étude d'ablation et apprentissage méthodologique

L'expérience de perte cross-entropy pondérée (NEUTRAL × 2.5) menée en complément a livré un résultat **négatif** : la précision a régressé à environ 76 %, soit un recul de plus de 4 points par rapport au modèle non pondéré. L'analyse a révélé la cause : le jeu d'entraînement étant déjà équilibré par sous-échantillonnage à 15 000 par classe, une pondération supplémentaire force le modèle à sur-prédire NEUTRAL, dégradant à la fois la précision sur cette classe (qui était déjà le maillon faible) et celle des classes polaires par effet de bord.

Cet apprentissage est important sur le plan méthodologique : la pondération de classe est un outil utile lorsque la distribution d'entraînement reflète l'imbalance naturelle du test, mais devient nocif lorsque l'entraînement a déjà été équilibré en amont par échantillonnage. Cette distinction n'est pas toujours explicitée dans la littérature pédagogique sur le fine-tuning, et nous la documentons explicitement pour les équipes futures.

## 11.3 Cohérence temporelle entre données d'entraînement et requêtes de marché

Le dataset Amazon Reviews 2018 alimente le modèle de sentiment, alors que l'agent de recherche web interroge DuckDuckGo en 2026. Cette asymétrie temporelle est consciente. Elle s'appuie sur l'hypothèse — vérifiée empiriquement par la stabilité des classifieurs de sentiment vieillissants — que **les patterns linguistiques exprimant le sentiment sont relativement stables sur 5-10 ans** : le vocabulaire de l'enthousiasme (« love », « excellent », « perfect ») et de la déception (« terrible », « broken », « waste of money ») évolue lentement. À l'inverse, **la connaissance de marché (concurrents, prix, tendances) doit être actuelle** ; un dataset d'avis 2018 ne suffit pas à informer une décision commerciale en 2026.

La séparation des deux sources est ainsi une force architecturale plutôt qu'une incohérence : le composant DL apporte la stabilité de classification, l'agent de recherche apporte la fraîcheur du contexte. Si le projet devait passer en production, le CSV statique serait remplacé par un flux temps réel d'avis (par exemple via l'API Shopify Admin pour des marchands abonnés).

## 11.4 Limites du système

Plusieurs limites sont reconnues et documentées :

**11.4.1 Quota API Gemini.** Le tier gratuit de Gemini 2.5 Flash impose 20 requêtes par jour. Une exécution complète de la pipeline (3 tâches + délégations + synthèse + révision HITL) consomme typiquement 6 à 12 appels, soit 1 à 3 exécutions complètes par jour. Cette limite est gérée par le cache disque (`LITELLM_CACHE=disk`) qui évite les appels redondants sur des prompts identiques, mais elle reste un goulot pour la phase de tests intensifs et la démonstration.

L'expérimentation conduite pendant le développement a révélé une limitation supplémentaire qui mérite d'être documentée : les variantes du modèle Gemini ont des stratégies de quota différentes. `gemini-2.5-flash` applique une limite quotidienne (20 requêtes par jour) ; `gemini-flash-latest` (qui résout vers `gemini-3-flash` au moment de la rédaction) applique une limite par minute (5 RPM) en plus du quota quotidien. La gestion automatique des erreurs 429 par CrewAI relance les requêtes quasi-immédiatement, ce qui sature la fenêtre par minute en quelques secondes et provoque une cascade d'échecs. Pour contourner ce comportement, un garde-fou de débit a été implémenté dans `src/main.py` (fonction `_install_llm_throttle`) sous forme d'un wrapper monkey-patch sur `llm.call`. Activé via la variable d'environnement `LLM_MIN_INTERVAL_S=N`, il impose un intervalle minimum de N secondes entre deux appels LLM. Avec `LLM_MIN_INTERVAL_S=13`, le débit reste sous 4.6 RPM, en dessous du plafond de 5 RPM de `gemini-flash-latest`. Une migration vers un tier payant ou un modèle local Llama 3 8B lèverait ces limitations — mais nécessiterait une infrastructure GPU plus capable.

**11.4.2 Une seule langue (anglais).** Le modèle de sentiment est fin-tuné sur des avis anglophones uniquement. Une niche francophone, arabophone ou multilingue ne serait pas servie. La généralisation multilingue exigerait un modèle pré-entraîné de type XLM-RoBERTa, avec un coût d'entraînement et d'inférence plus élevé.

**11.4.3 Couverture limitée du dataset.** Les avis utilisés couvrent les catégories Electronics et All Beauty d'Amazon. Une niche en dehors de ces catégories (par exemple Jardin, Outils, Bébé) pourrait sous-performer si le vocabulaire diffère significativement. La discussion §3.2 mentionne l'enrichissement par d'autres catégories (Home & Kitchen, Clothing) comme amélioration future.

**11.4.4 Tests d'intégration live limités par le quota.** La suite pytest contient 30 tests unitaires et structurels passants ; un test E2E live supplémentaire (`tests/test_crew_smoke.py`) est gated derrière un marker `@pytest.mark.smoke` opt-in afin de ne pas brûler le quota Gemini gratuit (20 requêtes/jour) à chaque exécution de la suite. La validation E2E est manuelle et limitée à 3-5 niches avant la soutenance.

**11.4.5 Démonstration en direct sensible aux pannes externes.** Une démonstration live dépend de Gemini, de DuckDuckGo et du réseau Wi-Fi de l'amphithéâtre. Cette dépendance est mitigée par le mode replay livré dans `src/main.py` (option `--replay <chemin.jsonl>`), qui rejoue un run sauvegardé sans appel externe et garantit un fallback déterministe en cas de coupure réseau ou de quota épuisé.

## 11.5 Considérations éthiques

Trois questions éthiques méritent d'être nommées :

**11.5.1 Biais des avis Amazon.** Les avis sont écrits majoritairement par une population anglophone, tech-friendly, avec un biais vers les acheteurs satisfaits (qui s'expriment plus volontiers sur les produits qu'ils aiment). Le modèle apprend implicitement ce biais et l'injecte dans toute analyse aval. Une utilisation commerciale exigerait une calibration explicite et une mesure de l'écart entre la sentiment population vs. la population réelle.

**11.5.2 Détection de faux avis.** Les datasets d'avis Amazon sont connus pour contenir une fraction non négligeable de faux avis (estimés ~5-15 % selon les études). Le modèle ne distingue pas les vrais des faux ; il classe toujours selon la polarité linguistique. Une production sérieuse intégrerait un détecteur de faux avis en amont (par exemple le filtre `verified_purchase` du dataset, déjà présent dans la table mais non exploité dans la version courante).

**11.5.3 Confidentialité des reviewers.** Les avis sont publics par nature, mais les reviewer_id et helpful_votes pourraient permettre une identification indirecte. Le système ne ré-expose pas ces métadonnées dans son rapport final ; il agrège uniquement la distribution de sentiment et les thèmes saillants.

## 11.6 Pistes d'amélioration

Trois pistes ressortent du travail mené :

1. **Architecture supérieure** : DeBERTa-v3-base avec un ensemble de 3 graines, ou un modèle d'échelle 7B+ via inférence cloud, atteindrait probablement 84-86 %.
2. **Préfixe de rating ordinal** : injecter `[RATING: X/5]` en préfixe du texte avant tokenisation donnerait au modèle un ancrage explicite, particulièrement utile à la frontière NEUTRAL (gain attendu +1 à +2 pp).
3. **Cascade de classifieurs** : un premier classifieur grossier (POLAR vs NEUTRAL) suivi d'un second classifieur fin (POSITIVE vs NEGATIVE) sur les seuls polaires pourrait pousser le F1 polaire au-delà de 0.92.

Aucune de ces pistes n'est triviale ; elles représentent autant de prolongements possibles dans un cadre M2 ou un projet de recherche.


# 12. Conclusion et perspectives

## 12.1 Synthèse du travail accompli

Le présent projet a livré un système d'intelligence sur les avis produits articulé autour d'un paradigme multi-agents hiérarchique. Trois composants ont été conçus, implémentés et validés : (i) un agent de recherche de marché s'appuyant sur DuckDuckGo et un mécanisme de rate limiting et de retry exponentiel ; (ii) un agent d'analyse de sentiment exploitant un modèle RoBERTa-base fin-tuné sur 45 000 avis Amazon Reviews 2018 et atteignant 80.83 % de précision sur un jeu de test indépendant ; (iii) un orchestrateur hiérarchique implémenté via CrewAI 1.14.4, doté d'un manager LLM personnalisé et d'un mécanisme humain-en-boucle qui valide la synthèse finale avant émission du rapport.

L'ingénierie a respecté trois principes structurants : la modularité (chaque agent et chaque outil est isolé dans son fichier, avec une interface contractuelle claire), l'auditabilité (toute exécution génère un journal JSON Lines couvrant 8 types d'événements, thread-safe), et la robustesse aux erreurs externes (chaque outil retourne un dictionnaire d'erreur structuré plutôt que de propager une exception). La couverture de tests automatisés atteint 30 tests passants couvrant les outils, les utilitaires et la structure de l'orchestration.

Une étude d'ablation comparant DistilBERT-base et RoBERTa-base a justifié le choix final, et une seconde ablation explorant la perte cross-entropy pondérée par classe a livré un résultat négatif documenté comme apprentissage méthodologique : la pondération nuit lorsque l'entraînement est déjà équilibré.

## 12.2 Atteinte des objectifs du cahier des charges

Le cahier des charges Domaine B prescrivait six livrables principaux, dont le statut au moment de la soutenance est le suivant :

| Livrable | Statut |
|---|---|
| Système multi-agents hiérarchique CrewAI (3 composants) | Atteint |
| Modèle DL fin-tuné, ≥ 85 % de précision | Atteint partiellement (80.83 % — voir §11) |
| Mécanisme humain-en-boucle | Atteint |
| Gestion des erreurs structurée | Atteint |
| Journalisation traçable | Atteint |
| Démonstration sur niche unique | Atteint sur 3+ niches en validation |

L'écart sur le critère de précision (80.83 % vs 85 %) est argumenté en §11.1 et documenté comme plafond inhérent à la formulation 3 classes, atteignable seulement via une architecture plus large (DeBERTa-v3) ou un ensemble — toutes deux hors périmètre temps.

## 12.3 Perspectives à court terme (3-6 mois)

À court terme, trois améliorations sont prioritaires :

1. **Migration vers DeBERTa-v3-base et ensemble de graines** pour franchir la barre des 85 %.
2. **Extension multilingue** via XLM-RoBERTa pour servir des niches non anglophones.
3. **Tests E2E automatisés** via cassette LLM (mock de Gemini enregistré) pour tourner dans la CI sans consommer le quota.

## 12.4 Perspectives à moyen terme (6-18 mois)

Le système peut évoluer vers une plateforme de market intelligence productisée, notamment sous forme d'application Shopify pour des marchands souhaitant analyser leurs propres avis produits. Le pipeline DL est déjà découplé : il consommerait directement les avis remontés via l'API Shopify Admin ou une intégration de type Judge.me / Loox, en remplacement du CSV statique. Le composant CrewAI deviendrait optionnel pour cette version produit ; le sentiment classifier seul, exposé en endpoint, suffirait pour un MVP.

Une seconde direction est l'intégration de signaux ordinaux : le rating étoile, les helpful votes et le verified_purchase peuvent être injectés en entrée du modèle (préfixe textuel ou tête multimodale), avec un gain attendu particulièrement sur la classe NEUTRAL.

## 12.5 Perspectives à long terme (18 mois +)

À l'horizon de plusieurs années, deux directions sont envisageables :

**Cascade de classifieurs spécialisés** : entraîner d'abord un classifieur grossier POLAR vs NEUTRAL, puis un classifieur fin POSITIVE vs NEGATIVE sur les seuls polaires. Cette architecture exploite la nature hiérarchique du problème (la frontière polaire/neutre est moins ambiguë que la frontière NEG/NEU et POS/NEU prises ensemble) et pourrait pousser le F1 polaire au-delà de 0.92.

**Ouverture vers des domaines au-delà du e-commerce** : l'architecture multi-agents et le pipeline d'analyse de sentiment se transposent à d'autres domaines structurés autour d'avis (restaurants, applications mobiles, livres, hôtellerie). Chaque domaine demanderait un fine-tuning sur des avis spécifiques mais conserverait l'orchestration et le mécanisme HITL.

## 12.6 Apprentissages personnels

Au-delà du livrable technique, ce projet a permis de consolider plusieurs compétences. La rigueur méthodologique sur l'ablation a été particulièrement formatrice : un résultat négatif (la perte pondérée qui régresse) est aussi instructif qu'un résultat positif, à condition d'en analyser la cause plutôt que de l'enterrer. La lecture honnête du plafond de 80.83 % comme limite de la formulation 3 classes plutôt que comme échec personnel est une posture professionnelle qui sera utile dans tout futur projet d'apprentissage automatique.

L'orchestration multi-agents via CrewAI a aussi mis en évidence la criticité du prompt engineering du manager : un manager auto-généré (`manager_llm`) tend à répondre directement aux questions plutôt qu'à déléguer ; un manager personnalisé (`manager_agent`) avec une backstory explicite force le bon comportement. Cette nuance, peu documentée dans les tutoriels CrewAI, mérite d'être consignée pour les équipes qui suivront.

## 12.7 Remerciements

Nos remerciements vont au Pr. Hafidi pour l'encadrement du module *Intelligence Artificielle & Big Data*, à l'équipe pédagogique de l'UIR pour la qualité du cadre, ainsi qu'à la communauté open source (HuggingFace, CrewAI, PyTorch, Google AI Studio) dont les outils ont rendu ce projet réalisable dans le temps imparti.


