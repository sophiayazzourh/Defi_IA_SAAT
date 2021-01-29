

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## Défi IA 2020-2021

- Lien du défi : https://www.kaggle.com/c/defi-ia-insa-toulouse/overview 
- Lien de la vidéo de présentation :  https://www.youtube.com/watch?v=374L1Px6mXY&feature=youtu.be 
- Lien des slides de présentation : https://www.docdroid.net/kVCLAO7/challenge-data-science-2020-pdf 

SAAT, équipe de l'INSA Toulouse composée de Vu Nam Anh LE, Aimée SIMCIC--MORI, Thanh Tin VO & Sophia YAZZOURH. 

Cette année, nous avons eu l’opportunité de participer au concours Defi-IA 2021 sur Kaggle, organisé par plusieurs écoles, notamment l’INSA Toulouse. L’objectif de ce défi est de créer un algorithme qui attribue la bonne catégorie des métiers à une descriptiond’un emploi. Cela revient donc à faire une classification multi-classe parmi 28 catégories d’emploi.

Les données ont été récupérées de CommonCrawl, qui a été utilisé pour entraîner le modèle GPT-3. Les données sont donc représentatives de ce qui peuvent être trouvés sur Internet en anglais parlé. Par conséquent, elles contiennent naturellement des biais de langage, de la discrimination. L’enjeu de ce concours est donc de développer un algorithme qui est à la fois précis, mais aussi juste sur les erreurs de classifications homme/femme.

Ici, on trouvera la solution proposée par l'équipé SAAT. Le développement sera découpée en deux parties : tout d'abord le pre-processing appliqué au données, puis l'algorithme de classification choisi. Dans la première partie, on nettoiera les données et on appliquera une vectorirization. Dans la seconde partie, on entraînera une modèle de régression logistique avant de prédire sur les données de test fournies par le défi et de les mettre au format de soumission attendu. 


<!-- GETTING STARTED -->
## Pour reproduire nos résultats

Vous trouverez içi les étapes principales qui permettent de reproduire les résultats finaux que nous avons présentés. 

### Pré-requis

- Linux or macOS
- Python 3
- Pour les utilisateurs de pip, vous pouvez utilisez la commande `pip install -r requirements.txt`.


### Mise en place

1. Cloner Defi_IA_SAAT
   ```sh
   git clone https://github.com/sophiayazzourh/Defi_IA_SAAT.git
   ```
Vous avez principalement besoin des dossiers data et scripts, ainsi que du script python Solution_Defi_IA_SAAT.py. 
  
2. Ouvrir le terminal 

Ouvrez votre terminal et placez vous dans le dossier qui contient :  data, scripts et Solution_Defi_IA_SAAT.py. 

Vous pourrez ainsi lancer le script grâce à la commande : 
   ```sh
   python Solution_Defi_IA_SAAT.py
   ```
Trois dossiers seront ainsi crées : 

- vectorization qui contient les modèles de vectorisation 
- model qui contient les modèles de régression logistique 
- résultat qui contient les prédictions au format attendu par le défi

3. Temps de calcul 

Avec un macOS équipé d'un processeur 3,1 GHz Intel Core i5 double voeur on a les temps de calcul suivant : 

- Pour la partie pre-processing (Cleaning + Vectorisation) : 
- Pour la partie entraînement de la régression logisitique : 

4. (Optionel) Le notebook 

Dans ce répertoire git, vous pourrez également retrouver la solution proposée par notre équipe au format notebook afin d'en faciliter la lecture. 

<!-- USAGE EXAMPLES -->
## Résultats obtenus 

Les résultats que nous avons obtenus, nous place à la 69 ème place du classement et nous avons obtenu un score de 0.73339. 




