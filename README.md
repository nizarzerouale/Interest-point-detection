# Détection de Points d'Intérêt avec OpenCV

Ce projet offre une plateforme interactive pour l'exploration et l'analyse de la détection de points d'intérêt dans les images, utilisant les célèbres détecteurs KLT (Kanade-Lukas-Tomasi) et Harris-Förstner. Il intègre une interface graphique qui permet aux utilisateurs de modifier en temps réel les hyperparamètres des détecteurs et d'observer directement leur impact sur les résultats de détection.

## Remerciements

Un merci spécial à **Nicolas ROUGON**, affilié à l'institut Polytechnique de Paris, Telecom SudParis et au département ARTEMIS, pour ses contributions significatives à ce travail. Son rôle dans le développement des notebooks de détection de points d'intérêt avec OpenCV, offrant un guide approfondi sur la détection différentielle de points d'intérêt basée sur le tenseur de structure, a été crucial.

## Caractéristiques Principales

- **Détecteurs de Points d'Intérêt**: Utilisez les détecteurs KLT et Harris-Förstner pour identifier les points d'intérêt dans les images.
- **Interface Utilisateur Dynamique**: Ajustez les hyperparamètres des détecteurs via des trackbars pour expérimenter et trouver la meilleure configuration pour votre image.
- **Sauvegarde Automatique des Résultats**: Les images traitées sont automatiquement sauvegardées en formats JPEG et PNG pour une analyse ultérieure.
- **Analyse de Robustesse**: Testez la répétabilité des détecteurs sous diverses conditions d'image, telles que le bruit, les variations d'éclairement, et plus encore.

## Commencer

Pour démarrer avec ce projet, clonez le dépôt et suivez les instructions ci-dessous pour configurer votre environnement.

### Prérequis

- Python 3.x
- OpenCV
- Numpy

### Installation

1. Clonez le dépôt GitHub :
git clone https://github.com/nizarzerouale/detection-points-d-int-r-ts
2. Installez les dépendances nécessaires :
pip install cv2 numpy sys


### Utilisation

Pour lancer l'application, exécutez le script principal à partir du notebook OpenCV_KeypointDetection


Utilisez les trackbars dans l'interface graphique pour ajuster les hyperparamètres des détecteurs et observer les résultats sur les images chargées.

## Contribution

Les contributions à ce projet sont les bienvenues. Pour proposer des améliorations ou des corrections, veuillez ouvrir une issue ou soumettre une pull request.
