- self-attention is quadratic in sequence length
- On Modern GPU, compte speed has outpaced memory speed
- Memory in GPU is HBM or SRAM
- GPU SRAM is 20 MB only but very high speed, 19TB/sec
- io aweare, uses tiling to reduce #memory read/writes between HBM and SRAM
- fewer HMP access than standard attention
- block spare flash attention: approx attention algo that is faster than any existing approx attention method

- Kernel fusion: if there are multiple operation applied to the same input, the input can be loaded once from HBM, instead of multiple times for each operation.

- Q,K,V in HBM,

to be continued not feeling it right now

do read https://modal.com/blog/reverse-engineer-flash-attention-4
