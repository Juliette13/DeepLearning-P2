# DeepLearning : Classification d'images

Présentation Equipe : 
* Florian Caliz, 
* Aubin Porte, 
* Paul Loublier, 
* Juliette Verlaine.

__Contexte__ : Développer et mettre en place un modèle permettant de classifier des images.

## Instanciation du projet

### a. CNN - Home Made Model

### b. Inception V3 - Modèle pré entraîné

* Etape 1 : Télécharger les données sources [ici](https://www.kaggle.com/alxmamaev/flowers-recognition) et dezipper les données.
* Etape 2 : A la racine du folder dezippé flowers-recognition, placer le fichier _retrain.py_ et le fichier _label_image.py_
* Etape 3 : Lancer le fichier _retrain.py_ pour entraîner le modèle: 
```
python retrain.py --bottleneck_dir=bottlenecks --how_many_training_steps 500 --model_dir=inception --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt --image_dir flowers/
```
* Etape 4: Lancer le fichier _label_image.py_ pour tester le modèle - Remplacer le paramètre --image par l'image sur laquelle vous souhaitez tester le modèle: 
```
python label_image.py --graph=retrained_graph.pb --image=test_images/rose.jpg --labels=retrained_labels.txt  --output_layer=final_result --input_layer=Placeholder
```

## CNN - Home Made

## Inception V3

Inception v3 est un modèle de reconnaissance d'images couramment utilisé qui a démontré, sur l'ensemble de données ImageNet, une justesse supérieure à 78,1 %. Il s'agit de l'aboutissement de nombreuses idées développées par plusieurs chercheurs au fil des années. Il est basé sur l'article originel Rethinking the Inception Architecture for Computer Vision (Repenser l'architecture Inception pour la vision par ordinateur) de Szegedy, et. al.
Le modèle lui-même est constitué de composants de base symétriques et asymétriques incluant convolutions, pooling moyen, pooling maximal, concaténations, abandons et couches entièrement connectées. La normalisation par lots (batchnorm) est amplement utilisée dans le modèle et appliqué aux entrées d'activation. La perte est calculée via Softmax.

![Architecture Inception V3](img/architecture_inceptionV3.png)

### a. Implémentation

Le modèle Inception v3, pré entraîné sur les données ImageNet est constitué de 7 couches, voir ci dessus le schéma. A ces 7 couches nous avons ajouté une couche fully connected permettant de ré entraîner le modèles sur nos classes : daysi, dendelion, rose, sunflower et tulip.

### b. Résultats

En comparaison des résultats du CNN Home Made, le modèle Inception v3 renvoie de meilleurs résultats sur l'entraînement, mais pas aussi significativement que nous le pensions : 

![Résultats 1](img/train_result.png)
![Résultats 1](img/train_result.png)

