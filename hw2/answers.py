r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 1, 0.01, 0
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp(opt_name):
    wstd, lr, reg, = 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different hyperparameters for each optimizer.
    # ====== YOUR CODE: ======
    if opt_name == 'vanilla':
        wstd, lr, reg = 1, 1e-5, 0 # Seems to be the only ones that matter to test
    if opt_name == 'momentum':
        wstd, lr, reg = 1, 1e-6, 0
    if opt_name == 'rmsprop':
        wstd, lr, reg = 1, 1e-6, 0
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd, lr = 0.1, 3e-4
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1. We would expect the training process of models with dropout to generalize better with the training data, 
resulting in a more generalized model with a lower expected overfitting to the training data. 
We would anticipate a slower convergence rate on the training set but better results in terms of test accuracy and loss.

For our graphs, we can clearly observe that despite the introduction of dropout making the training process slower, 
we achieve better results in terms of test loss and accuracy, as expected.

However, on the training set, we encounter some unexpected outcomes. 
The dropout model shows an overall improvement in train accuracy compared to the non-dropout model. 
We notice that in general, we experience some unstable training patterns across all models, 
which can be attributed to a high learning rate.

It's possible that the introduction of dropout effectively "dampens" the learning rate, resulting in a more stable training process.
It's worth noting that if we were to select a better learning rate, specifically a smaller one, 
we might obtain better results in terms of train accuracy for the non-dropout model. 
This improvement would be due to preventing overfitting, which may not carry over to the test segment.

2 . A very high dropout rate effectively reduces the expressivity of the model,
impairing its ability to accurately capture the complexity of the training data distribution.
We would anticipate observing a significant decrease in the convergence rate during training 
and a decline in test accuracy when using a high dropout rate.

We can clearly observe this effect in our results. 
The model with a high dropout rate exhibits difficulties in converging on the training segment, 
leading to a degenerate learning curve and inferior performance in terms of test loss and accuracy compared to its low dropout counterpart.
"""

part2_q2 = r"""
**Your answer:**

It's possible that the test accuracy and test loss will both  increase for a few epochs, 
because besides measuring the number of mistakes the cross entropy loss also measures the magnitude of the mistake 
but in contrast the accuracy only measures the number of mistakes.
that's why there can be a few epochs where the number of mistakes is decreasing but the magnitude of the mistakes is
increasing and therefore the loss is increasing as well. 
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""

**Your answer:**
We ran the experiments with 1 hidden layer of 128 in classifiers, pooling every L/2 layers, 
and early stopping with patience of 10 epochs.

For K=32:

Train Accuracy:
The depth 4 train loss decreased faster than the depth 2 net,
and although we did early stopping,from the graph we can guess that it will also achieve higher train accuracy, due to added expressivity.

Test Accuracy:
The depth 4 net achieved slightly better test accuracy than the depth 2 net,
we think this is due to the added expressivity. 
But it also had a much larger generalization gap, adding regularization might help with that.


The depth 8 and 16 nets did not manage to train at all, we think this is due to vanishing gradients in the higher depths.
To resolve this we can use skip connections to help the gradients flow better,
or use batch normalization to help with the vanishing gradients.

For K=64:

The results are very similar to the results of K=32.
In both depth 2 and depth 4 nets, the train and test accuracy for K=64 is slightly higher than for K=32.(about 1%)

Regarding the train accuracy, this gain is much smaller than what we got for increasing the depth.
(again with a grain of salt due to the early stopping, but this seems to be the trend)
This can possibly be explained by the fact that the increasing depth adds more non-linear operations,
while increasing the channels only allows us to learn more feature, and perhaps the 32 channels were about enough for this task.

The depth 8 and 16 nets still did not manage to train at all.


"""

part3_q2 = r"""

**Your answer:**
Depths 2,4:
Like in 1.1 we got slightly better train and test accuracy for depth 4, across all Ks.

The difference between K's is about the same for both depths,
with K=32 achieving the worst test accuracy, and all other about the same.

Depth 8 was still to deep and did not manage to train at all.

"""

part3_q3 = r"""
**Your answer:**
L=1,2 (depth 3,6):
Again the deeper network L=2 acheived slightly better test accuracy than L=1,
with the best being 78.18%.

We can see that in both depths, the test accuracy is higher than the best performing net in experiment 1.2 (L=4,K=256).
So we conclude that in this case the increasing channel number is good for generalization.
We think that this is because increasing the channel number allows heirarchical learning of features:
The lower layers can focus on low-level features like edges and textures, while higher layers can learn more high-level concepts like shapes and objects.

Again the deeper nets (L=3,4) did not manage to train at all.


"""


part3_q4 = r"""
**Your answer:**
Modifications:
1. We added batch normalization after each convolutional layer (to help with vanishing gradients)
2. We added skip connections each time the channel number changed-every L layers (to help with the gradient flow)
4. We set pooling to be every L layers.
4. We changed the classifier layers to be of dimension 512,512 (128 before).
5. We added dropout (p=0.2) after each convolutional layer and (p=0.5) after the first classifier layer
    (to help generalization).

Unchanged: optimizer, learning rate, weight decay, batch size, initialization.

The test accuracies for K=[64,128,256,512] are: L1=79.58, L2=85.82, L3=85.16, L4=85.68 
 (compared to 78.18% before modification achieved by L=2,K=[64,128,256]).

Oberservations:
1. The batch norm and skip connections allowed us to train the deeper nets (maximal depth of 16).

2. Dropout seems to shrink the generalization gap in all depths (relative to their counterparts in 1.2). 

3. L=2,3,4 test accuracies are very similar, despite the depth increase.
Maybe this is because L=2 (depth 8) is sufficient, 
and the skip connections for L=3,4 allowed learning a similar function to L=2.
(this is just an hypothesis, we did not check the weights to see if this is true).
 
4. Train loss is higher for deeper nets,
we don't know why this is the case, as they should have more expressivity.
Seems like an optimization problem, maybe with decreasing learning rate or lower dropout this will change?


"""
