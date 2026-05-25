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
