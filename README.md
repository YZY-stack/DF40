# DF40: Toward Next-Generation Deepfake Detection

Welcome to our work *DF40*, for next-generation deepfake detection. 

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-brightgreen.svg)](https://creativecommons.org/licenses/by-nc/4.0/) ![Release .10](https://img.shields.io/badge/Release-1.0-brightgreen) ![PyTorch](https://img.shields.io/badge/PyTorch-1.11-brightgreen) ![Python](https://img.shields.io/badge/Python-3.7.2-brightgreen)


**DF40 Dataset Highlight:**. The key features of our proposed **DF40 dataset** are as follows:

> ✅ **Forgery Diversity**: *DF40* comprises **40** distinct deepfake techniques (both representive and SOTA methods are included), facilialting the detection of nowadays' SOTA deepfakes and AIGCs. We provide **10** face-swapping methods, **13** face-reenactment methods, **12** entire face synthesis methods, and **5** face editing.
> 
> ✅ **Forgery Realism**: *DF40* includes realistic deepfake data created by highly popular generation software and methods, *e.g.,* HeyGen, MidJourney, DeepFaceLab, to simulate real-world deepfakes. We even include the just-released DiT, SiT, PixArt-$\alpha$, etc.
> 
> ✅ **Forgery Scale**: *DF40* offers **million-level** deepfake data scale for both images and videos.

The figure below provides a brief introduction to our DF40 dataset.

<div align="center"> 
</div>
<div style="text-align:center;">
  <img src="df40_figs/df40_intro.jpg" style="max-width:60%;">
</div>


---


The following two figures display the statistical description and illustrate the details of our DF40 dataset. **Please check our paper for details.**

<div align="center"> 
</div>
<div style="text-align:center;">
  <img src="df40_figs/table1.jpg" style="max-width:60%;">
</div>

<div align="center"> 
</div>
<div style="text-align:center;">
  <img src="df40_figs/table2.jpg" style="max-width:60%;">
</div>



---

The figure below shows the pipeline of four generation methods used in our DF40 dataset.


<div align="center"> 
</div>
<div style="text-align:center;">
  <img src="df40_figs/df40_pipeline.jpg" style="max-width:60%;">
</div>




## 💥 DF40 Dataset
| Type                    | ID-Number | Method        | Download Link | Visual Examples |
|-------------------------|-----------|---------------|---------------|-----------------|
| Face-swapping (FS)      | 1         | FSGAN         |Coming Soon               |  [![fsgan-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/fsgan.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/fsgan.gif)               |
|                         | 2         | FaceSwap      |Coming Soon               |  [![faceswap-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/faceswap.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/faceswap.gif)               |
|                         | 3         | SimSwap       |Coming Soon               |  [![simswap-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/simswap.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/simswap.gif)               |
|                         | 4         | InSwapper     |Coming Soon               |  [![inswap-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/inswap.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/inswap.gif)               |
|                         | 5         | BlendFace     |Coming Soon               |   [![blendface-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/blendface.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/blendface.gif)              |
|                         | 6         | UniFace       |Coming Soon               |   [![uniface-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/uniface.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/uniface.gif)              |
|                         | 7         | MobileSwap    |Coming Soon               |   [![mobileswap-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/mobileswap.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/mobileswap.gif)              |
|                         | 8         | e4s           |Coming Soon               |   [![e4s-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/e4s.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/e4s.gif)              |
|                         | 9         | FaceDancer    |Coming Soon               |    [![facedancer-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/facedancer.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/facedancer.gif)             |
|                         | 10        | DeepFaceLab   |Coming Soon               |    [![deepfacelab-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/deepfacelab.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/deepfacelab.gif)            |
| Face-reenactment (FR)   | 11        | FOMM          |Coming Soon               |    [![fomm-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/fomm.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/fomm.gif)             |
|                         | 12        | FS_vid2vid    |Coming Soon               |    [![face_vid2vid-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/face_vid2vid.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/face_vid2vid.gif)             |
|                         | 13        | Wav2Lip       |Coming Soon               |    [![wav2lip-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/wav2lip.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/wav2lip.gif)             |
|                         | 14        | MRAA          |Coming Soon               |    [![mraa-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/mraa.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/mraa.gif)             |
|                         | 15        | OneShot       |Coming Soon               |    [![oneshot-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/oneshot.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/oneshot.gif)             |
|                         | 16        | PIRender      |Coming Soon               |    [![pirender-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/pirender.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/pirender.gif)             |
|                         | 17        | TPSM         |Coming Soon               |    [![tpsm-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/tpsm.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/tpsm.gif)             |
|                         | 18        | LIA           |Coming Soon               |    [![lia-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/lia.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/lia.gif)             |
|                         | 19        | DaGAN         |Coming Soon               |    [![dagan-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/dagan.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/dagan.gif)             |
|                         | 20        | SadTalker     |Coming Soon               |    [![sadtalker-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/sadtalker.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/sadtalker.gif)             |
|                         | 21        | MCNet         |Coming Soon               |    [![mcnet-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/mcnet.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/mcnet.gif)             |
|                         | 22        | HyperReenact  |Coming Soon               |     [![hyperreenact-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/hyperreenact.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/hyperreenact.gif)            |
|                         | 23        | HeyGen        |Coming Soon               |     [![heygen-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/heygen.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/heygen.gif)            |
| Entire Face Synthesis (EFS) | 24    | VQGAN         |Coming Soon               |     [![vqgan-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/vqgan.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/vqgan.png)            |
|                         | 25        | StyleGAN2     |Coming Soon               |     [![stylegan2-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/stylegan2.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/stylegan2.png)            |
|                         | 26        | StyleGAN3     |Coming Soon               |     [![stylegan3-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/stylegan3.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/stylegan3.png)            |
|                         | 27        | StyleGAN-XL   |Coming Soon               |     [![styleganxl-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/styleganxl.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/styleganxl.png)            |
|                         | 28        | SD-2.1        |Coming Soon               |      [![sd2.1-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/sd2.1.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/sd2.1.png)           |
|                         | 29        | DDPM          |Coming Soon               |      [![ddpm-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/ddim.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/ddim.png)           |
|                         | 30        | RDDM          |Coming Soon               |      [![rddm-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/rddm.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/rddm.png)           |
|                         | 31        | PixArt-$\alpha$ |Coming Soon             |        [![pixart-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/pixart.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/pixart.png)         |
|                         | 32        | DiT-XL/2      |Coming Soon               |       [![dit-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/dit.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/dit.png)           |
|                         | 33        | SiT-XL/2      |Coming Soon               |       [![sit-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/sit.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/sit.png)          |
|                         | 34        | MidJounery6   |Coming Soon               |      [![mj-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/mj.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/mj.png)           |
|                         | 35        | WhichisReal   |Coming Soon               |      [![vqgan-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/whichisreal.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/whichisreal.png)           |
| Face Edit (FE)          | 36        | CollabDiff    |Coming Soon               |       [![collabdiff-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/collabdiff.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/collabdiff.png)          |
|                         | 37        | e4e           |Coming Soon               |       [![e4e-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/e4e.jpg)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/e4e.jpg)          |
|                         | 38        | StarGAN       |Coming Soon               |       [![stargan-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/stargan.jpg)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/stargan.jpg)          |
|                         | 39        | StarGANv2     |Coming Soon               |                 |
|                         | 40        | StyleCLIP     |Coming Soon               |       [![styleclip-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/styleclip.jpg)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/styleclip.jpg)          |


## 👀 Visual examples
<a href="#top">[Back to top]</a>

1. Example samples created by **FS (face-swapping)** methods:

<div align="center"> 
</div>
<div style="text-align:center;">
  <img src="./df40_figs/fs_visual.png" style="max-width:60%;">
</div>


2. Example samples created by **FR (face-reenactment)** methods:

<div align="center"> 
</div>
<div style="text-align:center;">
  <img src="./df40_figs/fr_visual.png" style="max-width:60%;">
</div>

3. Example samples created by **EFS (entire face synthesis)** methods:

<div align="center"> 
</div>
<div style="text-align:center;">
  <img src="./df40_figs/efs_visual.png" style="max-width:60%;">
</div>

4. Example samples created by **FE (face editing)** methods:

<div align="center"> 
</div>
<div style="text-align:center;">
  <img src="./df40_figs/unknown_visual.png" style="max-width:60%;">
</div>

