# Self-Pruning Neural Network (CIFAR-10)

## What is this?

This project explores a simple idea:

What if a neural network could decide by itself which connections it doesn’t need?

Instead of training a large model and pruning it later, this network learns to shrink itself while training.

---

## The Idea

Every weight in the network has a small “gate” attached to it.

* If the gate is open (≈ 1) → the weight is used
* If the gate is closed (≈ 0) → the weight is ignored

So during training, the model is not just learning what to predict, it is also learning what to remove.

---

## How it works (in simple terms)

Each weight becomes:

Effective Weight = Weight × Sigmoid(Gate)

And the loss becomes:

Total Loss = Classification Loss + λ × Sparsity Loss

* Classification Loss helps the model make correct predictions
* Sparsity Loss pushes unnecessary connections toward zero
* λ (lambda) controls how aggressive the pruning is

---

## What I observed

When training with different λ values:

* Small λ → higher accuracy, less pruning
* Medium λ → balanced behavior
* Large λ → more pruning, lower accuracy

This shows a clear trade-off between performance and efficiency.

---

## Outputs

The project generates:

* Gate distribution plots (to observe pruning behavior)
* Training curves (accuracy vs epochs)
* JSON summary of results

---

## Tech Used

* PyTorch
* NumPy
* Matplotlib

---

## How to run

```bash
pip install -r requirements.txt
python self_pruning_net.py
```

---

## Why this matters

Modern models are large and resource-heavy. In real-world systems like mobile or edge devices, efficiency is important.

This approach shows that a model can learn normally while also reducing its own complexity during training.

---

## Future ideas

* Extend this approach to CNNs for better performance
* Convert soft pruning into hard pruning
* Deploy as an API or web application

---

## Final thought

This project explores whether a neural network can learn not only what matters, but also what does not.

The results show that it can.
