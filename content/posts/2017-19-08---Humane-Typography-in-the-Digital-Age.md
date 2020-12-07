---
title: GAN & Conditional GAN
date: "2020-12-05T22:40:32.169Z"
template: "post"
draft: false
slug: "humane-typography-in-the-digital-age"
category: "BASIC"
tags:
  - "AI"
  - "GAN"
  - "Conditional GAN"
description: "An Essay on Typography by Eric Gill takes the reader back to the year 1930. The year when a conflict between two worlds came to its term. The machines of the industrial world finally took over the handicrafts."
socialImage: "/media/42-line-bible.jpg"
---


<br>

여러분은 아래 그림에서 위조지폐를 찾을 수 있나요?

![https://www.lifentalk.com/97](/media/fake_money.jpg)
<5만원권 지폐(좌), 5만원권 위조지폐(우)>

위조지폐범들은 어떻게 이런 정교한 위조지폐를 만들었을까요? 아마 처음에는 이렇게까지 정교하지 않았을 겁니다. 만든 위조지폐가 경찰에게 잡히고, 걸린 부분을 보완하고, 다시 검거되고, 다시 더 나은 위조지폐를 만들고, 이렇게 범인과 경찰이 쫓고 쫓기는 와중에 위조지폐는 계속해서 발전했을 것입니다. 이것을 인공지능에 적용하면 어떨까요?

GAN은 위와 같은 적대적 관계 인공지능 모델로 구현한 것입니다. 위조지폐 생성범에 대응되는 Generative model, 이를 검거하는 경찰에 대응되는 Dicriminator model이 적대적 관계를 이루며 인공지능 모델의 능력을 향상시킵니다. 

이번 포스팅에서는 GAN (Generative Adversarial Nets)과 GAN을 변형한 Conditional GAN을 살펴보겠습니다.



## Generative Adversial Nets

우선 Conditional Adversarial Nets의 발판이 된 GAN을 먼저 소개합니다. GAN은 Generative Adversarial Nets로써, 생성 모델 G와 판별 모델 D의 적대적 관계로 성능을 높이는 모델입니다.

![GvsD.png](/media/GvsD.png)

이러한 적대적 구조는 Minimax Algorithm이라고 불리는데, 범인 vs 경찰과 같이 적대적 관계를 이루는 Game에서 주로 쓰이는 알고리즘입니다. 바둑, 체스와 같은 1:1 게임을 예로 들어 볼까요? 내 차례에서 나는 상대방이 '최대'로 얻을 수 있는 점수를 '최소'화하는 전략을 쓸 것입니다. 이것이 바로 Minimax-Minimize Maximum-전략인 것이죠. 동일하게 내가 '최소'로 얻을 수 있는 점수를 '최대'화하려고도 하기 때문에, Minimax Algorithm은 Maximin 알고리즘으로 불리기도 합니다.

GAN은 다음과 같은 목적을 가진 Minimax Game입니다.
<br>
G(=Generator, 범인): To maximize the probability of D making a mistake.
<br>
D(=Discriminator, 경찰): To maximize the accuracy of discriminating the sample from G
<br><br>
그리고 이는 다음과 같은 하나의 수식으로 표현할 수 있습니다.

![equ1.png](/media/equ1.png)

이 아름다운 수식은 매우 직관적이기 때문에 쉽게 이해할 수 있습니다. GAN은 Generator(범인, 이하 G)가 실수하게끔, 즉 범인의 정답률(실제 지폐와의 일치율)을 최소화하고(=minG), Discriminator(경찰, 이하 D)가 성공하게끔, 즉 범인의 검거율을 최대화(maxD)하는 것이 목적인 가치 함수(V(D,G))인 것입니다.

그런데 Value function에서 log (1-D(G(z))) 부분을 G에 대해 minimize하는 대신 log(D(G(z)))를 maximize 하도록 G를 학습시키기도 합니다.
학습 초기에는 G가 아주 이상한 이미지를 생성하기 때문에 D가 너무 쉽게 G가 만들어낸 이미지를 가짜라고 판별하게 됩니다. 따라서 log(1-D(G(z))) 값이 쉽게 saturate하여 gradient를 너무 작아 학습이 더디게 진행됩니다.
하지만 이 부분을 G = arg max G log(D(G(z)))로 바꾸게 되면 초기에 D가 G로 나온 이미지를 잘 구별한다고 해도 위와 같은 문제가 생기지 않기 때문에 원래 문제와 같은 fixed point를 얻게 되면서도 stronger gradient를 줄 수 있는 해결방법이 되는 것입니다.
[문단 출처](http://jaejunyoo.blogspot.com/2017/01/generative-adversarial-nets-1.html)



## GAN의 구조
GAN의 원리를 알았으니 이제 GAN이어떻게 구성되었는지 살펴보겠습니다.

우선 Generator는 thetag를 parameter로 갖는 미분 가능한 함수로써, multilayer perceptron으로 구성됩니다. G는 확률 분포 Pg를 생성합니다. Pg가 위조지폐범이 만드는 위조지폐가 되는 것이죠. 하지만 이 범인은 생각보다 멍청해서, 처음에는 정말 고약한 위조지폐를 만들어 냅니다. G는 random noise 변수인 pz(z)를 통해서 초기 생성값을 만들어냅니다.

Discriminator 또한 multilayer perceptron으로 구성됩니다. D(x)는 Discriminator가 판별한, x가 '진짜'일 확률을 나타냅니다. 경찰은 Training data(진짜 데이터)와 G가 만들어낸 Generation data(가짜 데이터)를 구분할 확률을 최대화하는 방향으로 학습을 반복합니다. 


G와 D는 이와 같이 데이터 생성 분포를 반복적으로 만들면서, 서로의 성능을 개선시킵니다. 이 과정은 다음과 같은 그래프로 확인할 수 있습니다.

![graph1.png](/media/graph1.png)

파란색은 D의 판별 결과(=D(x), 1에 가까울 수록 진짜, 0에 가까울수록 가짜로 판별한 것)의 분포, 까만색은 실제 '진짜' 데이터의 데이터 분포, 녹색은 G가 생성한 데이터 분포입니다.

그래프를 한 번 살펴볼까요?
'진짜'데이터는 가우시안 분포를 따르고 있는 평범한 분포입니다. random noise input으로 생성된 '가짜'데이터는, 진짜 데이터와는 다른 분포 양상을 보이고 있습니다. (a), (b), (c), (d)의 과정을 거치면서 G가 생성한 데이터들의 분포(녹색)는 진짜 데이터들의 분포(검정)을 따라갑니다. 그에 맞추어 판별 결과의 분포는 1/2에 수렴합니다. 무한한 경쟁을 반복했을 때 G가 만들어내는 확률분포는 진짜 데이터의 확률분포가 일치하게 되고, D는 진짜와 가짜를 구분할 수 없게 됨으로써 항상 1/2의 확률로 진짜라고 판별하게 되는 것입니다. 판별이 무의미해지는 것이죠.

## GAN의 시사점
GAN은 기존 모델과는 차별화된 획기적 모델입니다. 수많은 차원에서 방대한 데이터를 class label에 mapping해야 했던 기존 모델과는 달리, 복잡한 계산과 Marcov Chain 없이 back-propagation만으로 학습이 가능한 GAN은 새로운 패러다임을 제시했다해도 과언이 아닙니다. 또한 GAN의 output은 input이 어떠한 class일 것이라는 단순 probability를 나타냈던 기존의 수많은 모델들과는 달리 데이터의 형태, 즉 분포와 분산을 만들어냈습니다. 이를 통해 Optimization Algorithm 등으로 상용이 가능하며 input 데이터가 어떠한 class인지 '판별'에 그쳤던 기존의 인공지능 모델과는 달리 '창조'를 해낼 수 있게 된 것입니다.


## Conditional Generative Nets


Conditional Generative Adversirial Nets은 GAN 모델의 conditional version입니다. 기존 모델에 조건을 부여함으로써 생성되는 이미지의 boundary를 한정해주어 random하게 생성되는 이미지를 제어할 수 없다는 기존 모델의 한계를 개선하였습니다. 

![conditionalgan.jpg](/media/conditionalgan.jpg)

*Figure 1: Conditional Adversarial Nets*

위 그림은 Conditional GAN을 도식화 한 것입니다. 기존 GAN 모델에 extra information인 y를 추가해 줌으로써 conditional model을 구축한 것입니다. 이 때 y는 class label이 될 수도 있고, 다른 양식일 수도 있는 모든 종류의 보조 정보 입니다. 본 논문에서는 MNIST의 class label이 y가 되어 예를 보여줍니다. Extra information y는 discriminator와 generator 모두에게 additional input layer로 추가됩니다.

기존 GAN의 Generator에서는 prior input noise pz(z)만이 input이 되었던 것과는 달리, Conditional GAN에서는 extra information인 y가 joint hidden representation으로써 결합되어 함께 입력으로 들어갑니다. Discriminator 역시, y가 x와 함께 input으로 들어갑니다. 이는 GAN과 동일하게 MLP(Multi Layer Perceptron)으로 삽입됩니다.

이는 다음과 같은 하나의 수식으로 정리될 수 있습니다.

![eq2.jpg](/media/eq2.jpg)

GAN과 동일하게 two-player minimax 게임이며, 위의 가치함수는 이 게임의 objective function입니다.


## Experimental Results

본 논문의 저자는 Conditional GAN의 Experiments로 Unimodal과 Multimodal 두 가지 케이스로 나누어 실험을 진행했습니다.

1. Unimodal의 경우
저자는 MNIST를 실험 대상 dataset으로 삼았습니다. 

![fig2.jpg](/media/fig2.jpg)

> MNIST의 경우 target value가 0~9까지 이므로 학습시킬 때 encoding matrix를 넣어서 학습시킵니다. batch를 30, z의 dimension이 100, x의 dimention이 784(32x32)이므로 Generator의 input dimension 은 30x(100+10)이고, Discriminator의 input dimension은 30x(784+10)입니다. — [출처](https://t-lab.tistory.com/29)
>


Typography is not about typefaces. It’s not about what looks best, it’s about what feels right. What communicates the message best. Typography, in its essence, is about the message. “Typographical design should perform optically what the speaker creates through voice and gesture of his thoughts.”, as El Lissitzky, a famous Russian typographer, put it.

## Loss of humanity through transitions

Each transition took away a part of humanity from written language. Handwritten books being the most humane form and the digital typefaces being the least. Overuse of Helvetica is a good example. Messages are being told in a typeface just because it’s a safe option. It’s always there. Everyone knows it but yet, nobody knows it. Stop someone on the street and ask him what Helvetica is? Ask a designer the same question. Ask him where it came from, when, why and who designed it. Most of them will fail to answer these questions. Most of them used it in their precious projects but they still don’t spot it in the street.

<figure>
	<blockquote>
		<p>Knowledge of the quality of a typeface is of the greatest importance for the functional, aesthetic and psychological effect.</p>
		<footer>
			<cite>— Josef Mueller-Brockmann</cite>
		</footer>
	</blockquote>
</figure>

Typefaces don’t look handmade these days. And that’s all right. They don’t have to. Industrialism took that away from them and we’re fine with it. We’ve traded that part of humanity for a process that produces more books that are easier to read. That can’t be bad. And it isn’t.

> Humane typography will often be comparatively rough and even uncouth; but while a certain uncouthness does not seriously matter in humane works, uncouthness has no excuse whatever in the productions of the machine.
>
> — Eric Gill

We’ve come close to “perfection” in the last five centuries. The letters are crisp and without rough edges. We print our compositions with high–precision printers on a high quality, machine made paper.

![type-through-time.jpg](/media/type-through-time.jpg)

*Type through 5 centuries.*

We lost a part of ourselves because of this chase after perfection. We forgot about the craftsmanship along the way. And the worst part is that we don’t care. The transition to the digital age made that clear. We choose typefaces like clueless zombies. There’s no meaning in our work. Type sizes, leading, margins… It’s all just a few clicks or lines of code. The message isn’t important anymore. There’s no more “why” behind the “what”.

## References


- [The first transition](#the-first-transition)
- [The digital age](#the-digital-age)
- [Loss of humanity through transitions](#loss-of-humanity-through-transitions)
- [Chasing perfection](#chasing-perfection)


Human beings aren’t perfect. Perfection is something that will always elude us. There will always be a small part of humanity in everything we do. No matter how small that part, we should make sure that it transcends the limits of the medium. We have to think about the message first. What typeface should we use and why? Does the typeface match the message and what we want to communicate with it? What will be the leading and why? Will there be more typefaces in our design? On what ground will they be combined? What makes our design unique and why? This is the part of humanity that is left in typography. It might be the last part. Are we really going to give it up?

*Originally published by [Ihyun Song](http://matejlatin.co.uk/) on [Medium](https://medium.com/design-notes/humane-typography-in-the-digital-age-9bd5c16199bd?ref=webdesignernews.com#.lygo82z0x).*