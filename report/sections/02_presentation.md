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

L'architecture générale suit un modèle hiérarchique où deux agents spécialisés rapportent à un orchestrateur implicite créé par CrewAI via `Process.hierarchical` avec `manager_llm='gemini-2.5-flash'`. Le flux d'exécution est le suivant :

1. **Entrée utilisateur** : l'utilisateur fournit une niche de produit (ex. "écouteurs sans fil").
2. **Délégation** : l'orchestrateur décide quels agents activer et dans quel ordre.
3. **Recherche de marché** : agent Market Research interroge DuckDuckGo sur les concurrents, prix, tendances.
4. **Analyse de sentiment** : agent Sentiment Analyst charge les avis produit depuis le dataset, les analyse via RoBERTa-base fin-tuné, agrège les sentiments.
5. **Checkpoint humain-en-boucle** : avant synthèse finale, les résultats préliminaires sont soumis à approbation utilisateur.
6. **Synthèse et rapport** : l'orchestrateur compile un rapport structuré en Markdown, approuvé ou modifié par l'utilisateur.
7. **Journalisation** : toutes les étapes sont tracées dans un log JSONL horodaté.

Cette architecture satisfait la spécification de 3 composantes (2 agents spécialisés + 1 orchestrateur implicit), respecte l'exigence multi-agent du brief, et intègre nativement le mécanisme humain-en-boucle sans outillage personnalisé complexe.
