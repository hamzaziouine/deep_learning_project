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
