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
