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

**11.4.4 Tests d'intégration live limités par le quota.** La suite pytest contient 30 tests unitaires et structurels passants ; un test E2E live supplémentaire (`tests/test_crew_smoke.py`) est gated derrière un marker `@pytest.mark.smoke` opt-in afin de ne pas brûler le quota Gemini gratuit (20 requêtes/jour) à chaque exécution de la suite. La validation E2E est manuelle et a couvert deux niches canoniques (écouteurs sans fil, sérums à l'argan) sauvegardées dans `demo/`.

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
