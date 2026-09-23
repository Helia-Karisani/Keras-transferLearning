# Transfer Learning with Keras

This project applies **transfer learning** in Keras by reusing a pre-trained convolutional neural network and adapting it to a new classification task.

---

## Overview

The notebook builds a model by taking an existing trained network and modifying it instead of training from scratch. This lets the model reuse previously learned visual features such as edges, textures, and shapes.

---

## Core Idea

A pre-trained model already contains useful feature detectors. Instead of relearning everything:

- early layers are **kept fixed (frozen)**
- new layers are added for the new task
- only the final layers are trained

This makes training faster.

---

## Workflow

1. Load a pre-trained model (without its top classification layer)
2. Freeze the base layers so their weights do not change
3. Add new layers for the target classification problem
4. Compile the model with a loss and optimizer
5. Train the model on the new dataset

---

## Freezing Layers

```python
for layer in base_model.layers:
    layer.trainable = False
```

This keeps previously learned features unchanged during training.

---

## Adding New Layers

```python
x = base_model.output
x = Dense(...)(x)
output = Dense(...)(x)
```

These layers learn task-specific patterns.

---

## Training

Only the newly added layers are trained at first. Later, some deeper layers can be unfrozen for fine-tuning.

Early CNN layers learn general features (edges, textures), while later layers learn task-specific ones, so it makes sense to keep the early layers and retrain the top.

---

## Files

```
Keras-transferLearning.ipynb
sample_data/
three png outputs
```
