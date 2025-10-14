ViT is essentially the AIAYN for vision. While AIAYN postulated that in NLP tasks, you have some token embeddings, and well AIAYN, lets scrap the RNNs, here we say we do not need the CNNs, we have patch embeddings, and AIAYN.

So it's like this

You have Vanilla ANN's, and they treat every pixel equally, so you need a ridiculous amounts of data for them to generalize, way more than feasible.

Now with CNNs, you say hey, let me exploit the fact that I am dealing with images, and let me sort of in a way attend to the pixels close by, and now the curse of dimensionality is bettered. So I add in the inductive bias.

Now with ViT, the model actually learns where to pay attention to, instead of just assuming it should be the nearby pixels.

Initially we needed huge data, but DeIT made it practical. I digress, that paper is a different discussion.

there is a matrix E, we take the patch, unroll it, multiply it with E, get the position embedding, and thats the input to the transformer. Now we have a standard transformer. We have another input, the cls token, like the bert embedding, so it is not associated with any patch, but is a learned input,so you input it to the transformer encoder, then the output foes to an MLP head, and you output the classificaiton scores.

BERT introduced the CLS token, that converts into an embedding after going through the layers, embed like a general representation of the whole sentence, and you can use it for classification tasks.
It is even more important here, the ideal mode of pre training is a classification task.

Think jigsaw puzzles, order of patches is very important. We see positional embeddings close to each other are more similar, so there is a sense of locality.
