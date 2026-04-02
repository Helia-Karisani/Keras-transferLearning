
# Transfer Learning with Keras

This project demonstrates how to apply **transfer learning** using Keras by reusing a pre-trained convolutional neural network and adapting it to a new classification task.

---

##  Overview

The notebook builds a model by taking an existing trained network and modifying it instead of training from scratch. This allows the model to benefit from previously learned visual features such as edges, textures, and shapes.

---

##  Core Idea

A pre-trained model already contains useful feature detectors. Instead of relearning everything:

- early layers are **kept fixed (frozen)**  
- new layers are added for the new task  
- only the final layers are trained  

This makes training faster and more efficient.

---

##  Workflow

The notebook follows these main steps:

1. Load a pre-trained model (without its top classification layer)  
2. Freeze the base layers so their weights do not change  
3. Add new layers for the target classification problem  
4. Compile the model with appropriate loss and optimizer  
5. Train the model on the new dataset  

---

##  Freezing Layers

Freezing means preventing updates to weights:

```python
for layer in base_model.layers:
    layer.trainable = False
````

This ensures that previously learned features are preserved during training.

---

##  Adding New Layers

New layers are added on top of the base model to adapt it:

```python
x = base_model.output
x = Dense(...)(x)
output = Dense(...)(x)
```

These layers learn task-specific patterns.

---

##  Training

Only the newly added layers are trained initially. This allows the model to adjust to the new dataset without destroying useful pre-trained features.

Optionally, some deeper layers can later be unfrozen for fine-tuning.

---

##  Why This Approach Works

* early CNN layers learn **general features** (edges, textures)
* later layers learn **task-specific features**
* transfer learning keeps general knowledge and updates only what is needed

---

##  File

```
Keras-transferLearning.ipynb
sample_data
```

---

##  Summary

This project shows how to efficiently reuse pre-trained models in Keras by freezing layers and adding new ones, enabling faster training and better performance on new tasks.

```

```
