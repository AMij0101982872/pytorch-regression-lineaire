# Régression linéaire avec PyTorch

Travaux pratiques de Deep Learning (HESTIM 2) : premiers pas avec PyTorch, avec une régression linéaire simple sur des données synthétiques.

## Objectif

Entraîner un modèle `nn.Linear(1, 1)` à retrouver la relation **y = 3x + 2** à partir de données bruitées, puis évaluer la qualité de l'ajustement avec le score R².

## Contenu du projet

| Fichier | Description |
|---|---|
| `regresiion.ipynb` | Notebook principal : génération des données, entraînement, visualisation et évaluation |
| `lab1.ipynb` | Notebook de test de l'environnement |
| `main.py` | Script de démarrage généré par `uv` |
| `pyproject.toml` / `uv.lock` | Dépendances du projet (gérées avec `uv`) |

## Démarche

1. **Choix du périphérique** : GPU (`cuda`) s'il est disponible, sinon CPU.
2. **Génération des données** :
   - 50 valeurs de `x` régulièrement espacées entre -2 et 2 (`torch.linspace`), mises en forme `(50, 1)` avec `unsqueeze(1)`.
   - Cible `y = 3x + 2 + bruit`, avec un bruit gaussien d'écart-type 0,6.
3. **Visualisation** des données en nuage de points (matplotlib).
4. **Modèle** : une couche linéaire `nn.Linear(1, 1)`, soit `y_pred = w·x + b`.
5. **Entraînement** :
   - Fonction de perte : erreur quadratique moyenne (`nn.MSELoss`)
   - Optimiseur : `Adam` avec un taux d'apprentissage de 0,01
   - 100 époques : propagation avant, calcul de la perte, `zero_grad()`, `backward()`, `step()`
   - La perte de chaque époque est enregistrée pour tracer la courbe d'apprentissage.
6. **Visualisation des résultats** : courbe de la perte, puis droite prédite (en rouge) superposée aux données, calculée en mode `torch.inference_mode()`.
7. **Évaluation** : score R² avec `torcheval.metrics.R2Score`.

## Résultats

- La perte diminue au fil des époques : le modèle apprend.
- La droite prédite suit bien le nuage de points.
- **Score R² obtenu : 0,9724**, donc le modèle explique environ 97 % de la variance des données.

## Installation et exécution

Prérequis : Python ≥ 3.11 et [uv](https://docs.astral.sh/uv/).

```bash
# Installer les dépendances
uv sync

# Ouvrir le notebook dans VS Code ou Jupyter, avec le noyau de l'environnement .venv
```

### Dépendances

- `torch`: tenseurs, réseaux de neurones, optimisation
- `torcheval`: métriques d'évaluation (R²)
- `matplotlib`: visualisation
- `numpy`
- `torchviz`: visualisation du graphe de calcul
- `ipykernel`: exécution des notebooks
