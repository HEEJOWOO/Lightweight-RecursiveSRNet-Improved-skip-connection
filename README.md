[RDCAB-RecursvieSRNet](https://github.com/HEEJOWOO/RDCAB-RecursiveSRNet-2021.02.IPIU-) 

[RDCAB-RecursvieSRNet→YouTube](https://www.youtube.com/watch?v=BW7Z-MUu7m4) 

[RDCAB-RecursvieSRNet→IPIU](http://www.ipiu.or.kr/2021/index.php)

[RDCAB-RecursvieSRNet-Split-Version](https://github.com/HEEJOWOO/RDCAB-RecursivSRNet-Split-Version-) 

[RDCAB-RecursvieSRNet-FD-IR](https://github.com/HEEJOWOO/RDCAB-RecursiveSRNet-FD-IR) 

# Lightweight RecursiveSRNet Improved Skip Connection

## ProPosed Network
![RDCAB_RecursiveSRNet_upgrade](https://user-images.githubusercontent.com/61686244/120751086-5fc21380-c542-11eb-88c3-34d63d74f0d9.png)

## ProPosed Blcok
![RDCAB_upgrade](https://user-images.githubusercontent.com/61686244/120751102-694b7b80-c542-11eb-870e-a5f286adafca.png)


## Experiments
* At this time, learning was conducted only at x4 magnification, and it will be studied at x2 and x3 magnifications in the future.

* Check train.py for detailed network configuration.

* Ubuntu 18.04, RTX 3090 24G
* Train : DIV2K
* Test : Set5, Set14, BSD100, Urban100

* The DIV2K, Set5 dataset converted to HDF5 can be downloaded from the links below.
* Download Igor Pro to check h5 files.



|Dataset|Scale|Type|Link|
|-------|-----|----|----|
|Div2K|x2|Train|[Down](https://www.dropbox.com/s/41sn4eie37hp6rh/DIV2K_x2.h5?dl=0)|
|Div2K|x3|Train|[Down](https://www.dropbox.com/s/4piy2lvhrjb2e54/DIV2K_x3.h5?dl=0)|
|Div2K|x4|Train|[Down](https://www.dropbox.com/s/ie4a6t7f9n5lgco/DIV2K_x4.h5?dl=0)|
|Set5|x2|Eval|[Down](https://www.dropbox.com/s/b7v5vis8duh9vwd/Set5_x2.h5?dl=0)|
|Set5|x3|Eval|[Down](https://www.dropbox.com/s/768b07ncpdfmgs6/Set5_x3.h5?dl=0)|
|Set5|x4|Eval|[Down](https://www.dropbox.com/s/rtu89xyatbb71qv/Set5_x4.h5?dl=0)|



|x4|Set5/ProcessTime|Set14/ProcessTime|BSD100/ProcessTime|Urban100/ProcessTime|
|--|----------------|-----------------|------------------|--------------------|
|RDN|32.47 / 0.018|28.81 / 0.023|27.72 / 0.017|26.61 / 0.040|
|RDCAB-RecursiveSRNet|32.29 / 0.012|28.64 / 0.016|27.62 / 0.013|26.16 / 0.021|
|Split Vesrion|32.24 / 0.015|28.65 / 0.018|27.62 / 0.018|26.08 / 0.026|
|FD & IRB|32.28 / 0.007|28.66 / 0.008|27.64 / 0.006|26.19 / 0.010|
|Improved Residual|32.45 / 0.015|28.79 / 0.014|27.70 / 0.014|26.42 / 0.019|

|x4|RDN|RDCAB-RecursvieSRNet|Split Version|FD & IRB|Improved Residual|
|-|---|--------------------|-------------|--------|-----------------|
|Parameters|22M|2.1M|1.2M|1.63M|1.63M|

|x4|RDN|RDCAB-RecursvieSRNet|Split Version|FD & IRB|Improved Residual|
|-|---|--------------------|-------------|--------|-----------------|
|Multi-Adds|1,309G|750G|567G|293G|293G|

* The proposed network has an average processing speed of 1.5x faster across all test sets than RDN. It has parameters and computations such as Split-Version and has twice the slow throughput, but produces 0.15 dB higher performance on average on all test sets.

## Reference
[RDN](https://arxiv.org/abs/1802.08797)

[DRRN](https://openaccess.thecvf.com/content_cvpr_2017/papers/Tai_Image_Super-Resolution_via_CVPR_2017_paper.pdf)

[RCAN](https://arxiv.org/abs/1807.02758)

[IMDN](https://arxiv.org/abs/1909.11856)

[AWSRN](https://arxiv.org/abs/1904.02358)

[RFDN](https://arxiv.org/abs/2009.11551)

[LESRCNN](https://arxiv.org/abs/2007.04344)

[IdleSR](https://link.springer.com/chapter/10.1007/978-3-030-67070-2_8)
