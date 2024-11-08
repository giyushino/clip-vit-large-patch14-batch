# clip-vit-large-patch14-batch
Allowing weaker devices to run inferences custom datasets through clip-vit-large-patch14 in batches to save memory and computational power.

Used transformers library to import clip-vit-large-patch14 and cifar10 dataset. First ran into problems trying to see how model performs on 30+ input images, wrote small function to split number of desired images into batches of 10 (seems to be faster than 8 for some reason). Google Colab and personal computer either ran out of RAM or VRAM. With changes and optimizations, former can easily run 5000+ images and latter can run 400. Wrote several functions to compute accuracy of the model currently and find the worst performing group. 

Wrote function to fine tune model using Pytorch. Utilized cross entropy loss by taking the softmax of the logits for each image and label.
