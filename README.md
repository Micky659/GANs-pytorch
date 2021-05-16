# GANs-pytorch
Implementation of different generative adversarial networks using pytorch.
GAN consist of two neural network that compete each other to get better in a particular task while simultaneously trying to defeat each other. The two networks namely generator that is tasked with creating a data(imagery, text, sound) out to pure noise and a discriminator that discrimnate between real data and fake data forged by generator, both the netwroks learn from each other and get better until eventually generator makes data that it is impossible for discrimnator to differentiate real data from fake data anymore and outputs 1/2 for all inputs.

Requirements- 

		python==3.8.3
		pip==20.0.2
		matplotlib==3.1.3
		pytorch==1.5.0
		torchvision==0.6.0
		numpy==1.18.4
		opencv-python==4.2.0.32
		GitPython==3.1.2
		tensorboard==2.2.2
		imageio==2.9.0
		jupyter==1.0.0

## 1) Classic GAN
Classic gan is the implementation of vanilla gan developed originally by Ian Goodfellow in his paper Generative Adversarial Nets proposed in 2014. It uses MNIST dataset and developes real looking handwritten digits from noise and feedback by discrimnator. 

Original paper- https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf

![Classic gan output](https://user-images.githubusercontent.com/45831878/118401005-9a423a00-b681-11eb-8bc4-4a70edf3bd9e.gif)

### Usage

To train- 

		train_classic_gan.py --num_epochs --batch_size --enable_tensorboard 
		
<p>&nbsp;</p>

**This implementation is inspired from [pytorch-GANs](https://github.com/gordicaleksa/pytorch-GANs.git) created by Aleksa Gordic**

<p>&nbsp;</p>

I will be uploading implementation of other gans in future like Cycle-GAN, DCGAN and StyleGAN trying to follow the code style of classic GAN.

<p>&nbsp;</p>

***Be my guest to build over my code and use it wisely***
