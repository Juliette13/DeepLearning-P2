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
