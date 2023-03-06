
Start with the convolution with input size $n$, output size $o$: $$o = \left \lfloor \frac{n-f+2p}{s} \right \rfloor + 1,$$ say $$n - f + 2p = ms + q, \; 0 \leq q < s,$$then $o = m + \lfloor \frac {q}{s} \rfloor + 1 = m+1$.  

Reversing, the transpose conv operation with input size $o$ has output size 
$n_{out} = (o-1)s + f -2p = ms + f -2p = n-q,$ so need output padding $q$. 

## Fixing padding
Say $$ f= 2k + l, \; l \in \{0,1\}, k\geq0,$$and set $$p=\left \lfloor \frac{f}{2} \right \rfloor = \left \lfloor \frac{2k+l}{2} \right \rfloor = k ,$$
then $$\begin{align} -f +2p &= -f +2k\\
&= -f +(f-l) = -l, \end{align}$$
so we have set $p$ in a way that reduces $f$ to $f \; (mod \, 2) = l$.  

**Consider a resnet skip connection with $f=1$, then the kernel size of the main convolutions should also be odd in order to end up with the same image size. ** 

Using this $p$ the convolutions have output size $$\begin{align} \left \lfloor \frac{n-f + 2p}{s} \right \rfloor + 1 &= \left \lfloor \frac{n -l}{s} \right \rfloor +1 \end{align}$$
- If $f$ is odd, $s=2$ and $n$ is even, get output $(n/2 -1) +1=n/2$.   
- If $f$ is odd, $s=2$  and $n$ is odd, get output $\frac{n-1}{2} + 1 = \frac{n+1}{2}$. 
- If $f$ is odd and $s=1$, then  get output $n-1+1=n$ -> **samepad**. 

Inverting, get $$s(o-1) + f -2p = so -s + l.$$
- If $f$ is odd and $s=2$, get $2o-1$.  
- If $f$ is odd, $s=2$ and $n$ is even, get $2(n/2) -1 = n-1$, so need outpad $1$.  
- If $f$ is odd, $s=2$  and $n$ is odd, get $2((n+1)/2)-1 = n+1 -1 = n$.
