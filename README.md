## Snake
Snake Activation For Pytorch
### Happy Snake Year 2025!

### what's Snake?
snake is an easy activation function.

- SnakeA is: y = tanh(x) + relu(x)
- SnakeB is: y = tanh(x) + silu(x)
- SnakeC is: y = erf(x) + gelu(x)

### their graphs are as follows:
![image](https://github.com/user-attachments/assets/d5844cac-ce02-4520-b538-9780bd2f83c9)

SnakeA <br/><br/><br/><br/><br/>
![image](https://github.com/user-attachments/assets/92d8aea9-107d-4f00-9793-c2efde3128ae)

SnakeB <br/><br/><br/><br/><br/>

![image](https://github.com/user-attachments/assets/51a8596a-3c5d-4a0e-8b19-a45b4d4286ab)

SnakeC <br/><br/><br/><br/><br/>

![image](https://github.com/user-attachments/assets/015c170a-e75f-46c6-a5ca-3e407807cf16)
DytSnakeB <br/><br/><br/><br/><br/>

### reason
I've notice that fashion activations like SiLU, GeLU, ReLU as well as Mish are the kind of self-gated activations. They have one in common which is they are closely zero while the input
 is negative. According to the paper **Searching for Activation Functions** (which Introduced swish activation), they found that most input values of the swish activation are in negative part, which in my opinion 
 shows that the well-trained net is "eager to learn something nagetive". Other than that, if we follows the units like "conv-bn-act" or "linear-act", the output of the activation will be the input of the next layer's 
 linear weight, not bias. and the input of value will be a closely zero if the network is "eager to learn something negative", then grad will be hard to flow through this part.<br/>
 So, Snake activation is an activation more like ELU. More grad flow pass the layer and linear in its next layer can get more information.
<br/><br/><br/><br/><br/><br/>
picture from Geogerbra

## SFReLU
full name Soft maxout Funnel Rectified Linear Unit, which is a soft version of FReLU.
origin FReLU are in: https://arxiv.org/pdf/2007.11824.pdf <br>
code in: https://github.com/megvii-model/FunnelAct <br>
### what's SFReLU?
SFReLU is an easy paramed activation funcion.
```
f(x0, x1) --> x:
    temp0 = x0 - x1
    temp0 = silu(temp0)
    x = temp0 + x1
    return x

sfrelu(x) --> x:
    y = dwconv(x) # shape same as x
    x = f(x, y)
    return x
```
### their graphs are as follows:
![image](https://github.com/user-attachments/assets/47993677-09ff-453b-b5fb-fb00961c8e83)
### reason
the origin maxout can be described as the function below.
```
f(x0, x1) --> x:
    temp0 = x0 - x1
    temp0 = relu(temp0)
    x = temp0 + x1
    return x
```
I simply replace relu with silu, which is considered a "soft" version of relu.
