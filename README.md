# Snake
Snake Activation For Pytorch
## Happy Snake Year 2025!

# what's Snake?
snake is an easy activation function.

- SnakeA is: y = tanh(x) + relu(x)
- SnakeB is: y = tanh(x) + silu(x)
- SnakeC is: y = erf(x) + gelu(x)

# their graphs are as follows:
![image](https://github.com/user-attachments/assets/d5844cac-ce02-4520-b538-9780bd2f83c9)

SnakeA <br/><br/><br/><br/><br/>
![image](https://github.com/user-attachments/assets/92d8aea9-107d-4f00-9793-c2efde3128ae)

SnakeB <br/><br/><br/><br/><br/>

![image](https://github.com/user-attachments/assets/51a8596a-3c5d-4a0e-8b19-a45b4d4286ab)

SnakeC <br/><br/><br/><br/><br/>
## reason
I've notice that fashion activations like SiLU, GeLU, ReLU as well as Mish are the kind of self-gated activations. They have one in common which is they are closely zero while the input
 is negative. According to the paper **Searching for Activation Functions** (which Introduced swish activation), they found that most input values of the swish activation are in negative part, which in my opinion 
 shows that the well-trained net is "eager to learn something nagetive". Other than that, if we follows the units like "conv-bn-act" or "linear-act", the output of the activation will be the input of the next layer's 
 linear weight, not bias. and the input of value will be a closely zero if the network is "eager to learn something negative", then grad will be hard to flow through this part.<br/>
 So, Snake activation is an activation more like ELU. More grad flow pass the layer and linear in its next layer can get more information.
<br/><br/><br/><br/><br/><br/>
picture from Geogerbra
