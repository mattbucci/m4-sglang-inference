# single-buffer mode: True (encode and gpu columns are separable only when True)
| case | launches | encode us/launch | gpu us/launch | weights MB/iter | weight GB/s |
|---|---|---|---|---|---|
| add chain | 1 | 0.54 | 2.51 | 0.00 | None |
| add fan | 1 | 2.46 | 1.46 | 0.00 | None |
| multiply-add fan (2 launches) | 2 | 0.84 | 0.19 | 0.00 | None |
| rms_norm chain | 1 | 0.69 | 2.65 | 0.00 | None |
| add + rms_norm chain (2 launches) | 2 | 1.01 | 2.63 | 0.00 | None |
| sigmoid * x + y chain (3 launches) | 3 | 0.84 | 1.69 | 0.00 | None |
| softmax 256 chain | 1 | 0.50 | 2.70 | 0.00 | None |
| partition 256 chain (the argpartition sort kernel) | 1 | 1.11 | 7.08 | 0.00 | None |
| argpartition 256 fan | 1 | 3.28 | -0.24 | 0.00 | None |
| router tail chain: softmax, argpartition, gather, sum, divide, concatenate (6 launches) | 6 | 3.63 | 3.96 | 0.00 | None |
| qmm 2048x2048 chain, distinct weights | 1 | 1.88 | 13.24 | 2.25 | 178.1 |
| qmm 2048x2048 fan, distinct weights | 1 | 2.01 | 9.87 | 2.25 | 239.1 |
| routed experts chain: 3 gather_qmm, 3 activation, sum, scale (8 launches) | 8 | 2.85 | 9.57 | 13.50 | 184.9 |
| token tail per iteration: greedy argmax over 248,320 | 1 | 10.33 | 59.48 | 0.00 | None |
| token tail per iteration: sampled, full argsort over 248,320 (the serving chain) | 1 | 86.89 | 300.05 | 0.00 | None |
| token tail per iteration: sampled, top-20 from the 20 best blocks of 512 (same candidates) | 1 | 94.41 | 148.48 | 0.00 | None |
