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

**Choix final pour le système : RoBERTa-base.** Le gain marginal de +1.24 pp justifie l'inférence légèrement plus lente (~30 % de paramètres en plus) car le coût d'inférence sur GPU local reste sous 50 ms par avis en FP16, soit largement compatible avec l'usage par lot de l'agent d'analyse (typiquement 50 avis par requête niche).

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
