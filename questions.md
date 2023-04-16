Notes from building make-more part3

- Bad initial loss: logits at initialization should be small values(equal) => use no bias + small weights
But can we set our inittal weights to 0 ? 

Fixing tanh activations =>

DEAD NEURON => IF THE GRAD OF THAT NEURON IS ZERO FOR ALL THE SAMPLES (CAN HAPPEN FOR ANY ACTIVATION). hOW IT CAN HAPPEN => A. CAN HAPPEN DURING INITIALIZATION B. Optimization => if high train loss/grad update some of the neurons might never get activated for any train example and they will remain same for the rest of the training. not a problem for (leakyrelu)

Much better optimizers,normalization layers and residual connections. with these we can go by initializing with divide by sqrt(fanin).no need to be super exact.

batch-norm = solves the dead relu prob directly. motif: (conv/linear(no need bias here) + batchnorm + activation)
residual network/blocks = 


1. Why its better to use cross-entropy than implement our own loss ?

ans: 
 - pytorch can internally fuse the operations thus more efficient. Also we are creating more intermediate memory(nodes) that are unnecssary

- numerically stable (as softmax is same even if we add arbitrary number to each of the values). Thus pytorch internally subtracts the max-value.

2. Why mini-batch gd is a good idea ?

- doing more steps of approx gradient is better than doing accurate lower # steps.

3. What's the initial loss that you would expect before training and is it same ?

4. What happens if your logits are large unequal numbers ? Try making weights smaller ? Then what happens when you initialize all the weights to zero then ?

5. what's the loss landscape now and explain ?(avoiding easy gains by just reducing the numbers)

6. Activation functions: (tanh). Study it's problems.
- what happens when it's exactly 0(gradient just passes through)
- when it's 1 (gradient vanishes)

7. What's a dead-neuron ?

Suppose at initialization you manage to avoid dead-neurons, can you still have the risk of having dead-neurons during training.

code
1. get the loss by fixing confidently wrong(softmax)

2. Fix tanh activations

Train for fixed epochs and report the loss.

3. What are other techniques that let's us get away with not so exact initialization ? (kaiming-he)

- Residual layers
- normalization (batch-norm)
- better optimizers


4. pre-activations = we don't want them to be way too small(then they are basically inactive) or large(as tanh becomes inactive)


#small details matter
5. How many trainable weigts are introduced by batch-normalization ? Why scale and shift in batch-norm.

What's unnatural about batch-norm ?
the hidden state is function of not just that example but function of whatever else happens to be in that batch. Is this good or bad ?

- It turns out that this jittering for each sample introduces noise and it acts like data augmentation => regularization => better. But also no one likes that the examples in the batch are coupled mathematically. This introduces all sorts of bugs.

- Layer, instance normalization.

- division by `epsilon`

- is adding bias doing anything or can we just remove it ? (as we are subtracting the mean it's not required)


# calculate means and std of distributions

#before batch norm: for large neural networks => specific gain values are used based on the activation used (for instance tanh uses 5/3) in the initialization
#batch norm avoids the need for doing it. Gain should'nt be too large or too small.


# keep track of grad to data ratio.

# should all layers have same learning rate => am i training not fast enough ? check the ratio of update to data. around 1e-3 is good.
#signifacantly more robust to initialization and gain but still need to fiddle with the learning rate.

#I believe you can calculate the gain by doing Gain = 1/sqrt(E[ f(Z)^2 ]) where Z is a standard Gaussian (so that Gain*f(Z) will have unit variance when Z is a standard Gaussian). 
#If you do this for tanh you ~=1.592 which I guess is close to 5/3?
