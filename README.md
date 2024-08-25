# DF40: Toward Next-Generation Deepfake Detection

Welcome to our work *DF40*, for next-generation deepfake detection. 

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-brightgreen.svg)](https://creativecommons.org/licenses/by-nc/4.0/) ![Release .10](https://img.shields.io/badge/Release-1.0-brightgreen) ![PyTorch](https://img.shields.io/badge/PyTorch-1.11-brightgreen) ![Python](https://img.shields.io/badge/Python-3.7.2-brightgreen)


**DF40 Dataset Highlight:** The key features of our proposed **DF40 dataset** are as follows:

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




## 💥 DF Dataset
| Type                    | ID-Number | Method        | Download Link | Visual Examples |
|-------------------------|-----------|---------------|---------------|-----------------|
| Face-swapping (FS)      | 1         | FSGAN         |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EUYXBP-FLfpGqoXNvemgCzIBb5mOp4MLzpMaXkVOnB0wzg?e=DtZmzv)               |  [![fsgan-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/fsgan.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/fsgan.gif)               |
|                         | 2         | FaceSwap      |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EfawT-a5jEFGlOq9h6gKHb8BZjCRlUL175-RCkC4xwjYxw?e=SAHQkh)               |  [![faceswap-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/faceswap.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/faceswap.gif)               |
|                         | 3         | SimSwap       |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EcHhmhM4w2FBjDX1DiQpvk0BSk50dgcAT7TiH5-rPmIIDA?e=Kk9pt9)               |  [![simswap-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/simswap.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/simswap.gif)               |
|                         | 4         | InSwapper     |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EWp5dSXVyiFIjqUAO-pqBwABEDMNi1VOlrAXtiCLaMqoqQ?e=7CQaIB)               |  [![inswap-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/inswap.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/inswap.gif)               |
|                         | 5         | BlendFace     |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/Efhz5ni0hRNMss16Ia-cOSkBhboyQFGxE1xGvmFDc61xXw?e=y1gXSF)               |   [![blendface-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/blendface.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/blendface.gif)              |
|                         | 6         | UniFace       |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EQLpC2UcH7pGpg1kBkweBbAB-AGa0ys_B5GqeIQGQ3SVpw?e=64eRr9)               |   [![uniface-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/uniface.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/uniface.gif)              |
|                         | 7         | MobileSwap    |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EUZ2RNJuIANBp_-xL_qNCa4BdS8T1sPTlHy3TwVVzURgug?e=2Ci41t)               |   [![mobileswap-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/mobileswap.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/mobileswap.gif)              |
|                         | 8         | e4s           |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EcPeAiqVLuVIugvovJEev1EBJamTG6GxKNipsKoi-OYKQw?e=NPagz5)               |   [![e4s-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/e4s.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/e4s.gif)              |
|                         | 9         | FaceDancer    |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/ERlrQNqPCEpPnKKyLu7KEYMBateR-IRzdAyhy1nV0NhCaQ?e=R7qznh)               |    [![facedancer-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/facedancer.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/facedancer.gif)             |
|                         | 10        | DeepFaceLab   |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/ERLDxXqXjjlPgbpQyZ15mIoBsHqSNvkvpuNfRAbmrnPjbg?e=2PTLBS)               |    [![deepfacelab-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/deepfacelab.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/deepfacelab.gif)            |
| Face-reenactment (FR)   | 11        | FOMM          |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EUws9ZP8DfxInXFBDHO-yZ4BTLdxwvByXam3WtMuvJ-Alg?e=orReX0)               |    [![fomm-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/fomm.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/fomm.gif)             |
|                         | 12        | FS_vid2vid    |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EbPmQ8Et1NdBnApAAKGdl-4BiGhBklD6pQQ3KdYSMt6jNA?e=fRPIbM)               |    [![face_vid2vid-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/face_vid2vid.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/face_vid2vid.gif)             |
|                         | 13        | Wav2Lip       |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EWWh6zIJShZOlFPUAjT7ogABUvJzPnryeiXTlR6ID0j6sQ?e=sOLrgx)               |    [![wav2lip-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/wav2lip.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/wav2lip.gif)             |
|                         | 14        | MRAA          |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/ETz0X1hB5CBEm0ADJ3dn-F8Bk5foynvQ4jmHnavZejvhdQ?e=cbqZa0)               |    [![mraa-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/mraa.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/mraa.gif)             |
|                         | 15        | OneShot       |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EZJHNvdyjUFBppW-OHQUg_IBxZvThiGNsplpw1B6XKUCsw?e=izixge)               |    [![oneshot-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/oneshot.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/oneshot.gif)             |
|                         | 16        | PIRender      |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/ESSUbXL36MBLpJXRsLYFOrIBF8p71mNeHu6j7BddXj0X_A?e=2H7CYC)               |    [![pirender-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/pirender.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/pirender.gif)             |
|                         | 17        | TPSM         |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EdEl6WJr35xPhoa-KwH2sH0Bg5lPR0EmGqAi5ZmHwenzZA?e=uaEK5y)               |    [![tpsm-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/tpsm.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/tpsm.gif)             |
|                         | 18        | LIA           |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EdQp7ty5MOpOjViCBAQxvrEBC_mNS57QvsEsnrXqqOv0mw?e=leBGZm)               |    [![lia-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/lia.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/lia.gif)             |
|                         | 19        | DaGAN         |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EUxxlMvrS4tMjfRDLxGfhUABzhrQ4BgRoQOK30VuR6T49w?e=fnFeXl)               |    [![dagan-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/dagan.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/dagan.gif)             |
|                         | 20        | SadTalker     |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EabgEcMgF8pJkCfxUAamBikB-hbguKrTuuBmBRfA859tHA?e=GnSTyg)               |    [![sadtalker-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/sadtalker.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/sadtalker.gif)             |
|                         | 21        | MCNet         |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/ER18O8NJwu9Fo0w6r0Xz48sBrrBLFt6bIVR_iJyc4T4QHg?e=rg0r0h)               |    [![mcnet-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/mcnet.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/mcnet.gif)             |
|                         | 22        | HyperReenact  |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EcbYB4rIAb5Hs9_HtdSITW0B4SRONFu2wjIT4yvWn2JgUA?e=wIKFVz)               |     [![hyperreenact-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/hyperreenact.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/hyperreenact.gif)            |
|                         | 23        | HeyGen        |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EfUFQs-MZRZOq3fpfbPgmasBXsKQAwwMGxjP8E50OdwqCQ?e=0URXYd)               |     [![heygen-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/heygen.gif)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/heygen.gif)            |
| Entire Face Synthesis (EFS) | 24    | VQGAN         |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EaX03B4zRzBFqtKXaQDAR3gBh18HYKv5q6k0SQvTMYz5OQ?e=H4OqG3)               |     [![vqgan-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/vqgan.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/vqgan.png)            |
|                         | 25        | StyleGAN2     |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EfcmYbPWeC1FkMs2mnpSpkQBCMtODYwuyLMoauHo5KvNSQ?e=UZsjPC)               |     [![stylegan2-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/stylegan2.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/stylegan2.png)            |
|                         | 26        | StyleGAN3     |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/ERvrO00gex5Eu-ZW_D_hyKEBjhhXa76aqkTTDN8Q34Izpw?e=R5XZ9N)               |     [![stylegan3-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/stylegan3.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/stylegan3.png)            |
|                         | 27        | StyleGAN-XL   |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/ESM4dEPKYspLjMxzWWlO6owBSGwpNWo4XYUuaCzAdbWPPQ?e=g4T7PS)               |     [![styleganxl-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/styleganxl.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/styleganxl.png)            |
|                         | 28        | SD-2.1        |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/ESY6AbSIx_hHj-2aLduJrdwBxQwe3U5ANeG0sGFk0PnuOQ?e=Vjwm0n)               |      [![sd2.1-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/sd2.1.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/sd2.1.png)           |
|                         | 29        | DDPM          |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EVfM0nmd0-tGgEZXoxM_TVEB-fGjJ1X8VeI_FqfDZGdzDA?e=u9R4nG)               |      [![ddpm-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/ddim.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/ddim.png)           |
|                         | 30        | RDDM          |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EQLtPwX7-IZKhY20J4h1S4sBc0bpyRn0MdYBcbBTHiupWg?e=cCbfCR)               |      [![rddm-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/rddm.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/rddm.png)           |
|                         | 31        | PixArt-$\alpha$ |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/ERnzCPmKchdEqlwnyK5zvxQB4sVFQo6wE_h2zwXLVgSb5A?e=mARXUu)             |        [![pixart-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/pixart.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/pixart.png)         |
|                         | 32        | DiT-XL/2      |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/Eb6NICxFafhFq54pRNlM74sB32e9Tq48hvUF53A5tu1I5Q?e=Nefcld)               |       [![dit-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/dit.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/dit.png)           |
|                         | 33        | SiT-XL/2      |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EW-3ZFwOh9tPtTvBmpJoHJoBZnLoHv4QS6Bq8CYZPZ5P0w?e=aRBzlv)               |       [![sit-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/sit.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/sit.png)          |
|                         | 34        | MidJounery6   |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/Ea7gykdNkYxIjdihRkOX-OkB-wfrXRvcyHrL8xO_FrT1Iw?e=kXyI3J)               |      [![mj-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/mj.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/mj.png)           |
|                         | 35        | WhichisReal   |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EfE0VjMMmhVGhaB3NOKkIewBFelxCqJZjbKWEcSGmbYVEg?e=k6cCAZ)               |      [![vqgan-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/whichisreal.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/whichisreal.png)           |
| Face Edit (FE)          | 36        | CollabDiff    |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EXYtnmeNlDlHgiMrLoesrZMBfQaDmX-HaN2-o-DVnZBe7Q?e=y5KC3p)               |       [![collabdiff-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/collabdiff.png)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/collabdiff.png)          |
|                         | 37        | e4e           |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EerqA8zZXfpCiuZINnh3PksB8KvKdhuNINJ5mDGraYrkTw?e=wZRPnV)               |       [![e4e-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/e4e.jpg)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/e4e.jpg)          |
|                         | 38        | StarGAN       |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/Ea6LPDfiPvROm2QrD9yDvUUBHUVQLOlZW1UjufLblzBUBw?e=eAzmJ8)               |       [![stargan-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/stargan.jpg)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/stargan.jpg)          |
|                         | 39        | StarGANv2     |               |                 |
|                         | 40        | StyleCLIP     |[One-Drive](https://cuhko365-my.sharepoint.com/:u:/g/personal/222041040_link_cuhk_edu_cn/EbKlB_5yBthCrngJl74LSG4BiX6Jwf9ciElzJlD-H-_Vzw?e=eEPuuN)               |       [![styleclip-Example](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/styleclip.jpg)](https://github.com/YZY-stack/temp_40_pangu/releases/download/visual_example/styleclip.jpg)          |



## ⏳ Quick Start

### 1. Installation
Please run the following script to install the required libraries:

```
sh install.sh
```

### 2. Preprocessing

<a href="#top">[Back to top]</a>

To start preprocessing your dataset, please follow these steps:

1. Download the [shape_predictor_81_face_landmarks.dat](https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/shape_predictor_81_face_landmarks.dat) file. Then, copy the downloaded shape_predictor_81_face_landmarks.dat file into the `./preprocessing/dlib_tools folder`. This file is necessary for Dlib's face detection functionality.

2. Open the [`./preprocessing/config.yaml`](./preprocessing/config.yaml) and locate the line `default: DATASET_YOU_SPECIFY`. Replace `DATASET_YOU_SPECIFY` with the name of the dataset you want to preprocess, such as `FaceForensics++`.

7. Specify the `dataset_root_path` in the config.yaml file. Search for the line that mentions dataset_root_path. By default, it looks like this: ``dataset_root_path: ./datasets``.
Replace `./datasets` with the actual path to the folder where your dataset is arranged. 

Once you have completed these steps, you can proceed with running the following line to do the preprocessing:

```
cd preprocessing

python preprocess.py
```


### 3. Rearrangement
To simplify the handling of different datasets, we propose a unified and convenient way to load them. The function eliminates the need to write separate input/output (I/O) code for each dataset, reducing duplication of effort and easing data management.

After the preprocessing above, you will obtain the processed data (*i.e., frames, landmarks, and masks*) for each dataset you specify. Similarly, you need to set the parameters in `./preprocessing/config.yaml` for each dataset. After that, run the following line:
```
cd preprocessing

python rearrange.py
```
After running the above line, you will obtain the JSON files for each dataset in the `./preprocessing/dataset_json` folder. The rearranged structure organizes the data in a hierarchical manner, grouping videos based on their labels and data splits (*i.e.,* train, test, validation). Each video is represented as a dictionary entry containing relevant metadata, including file paths, labels, compression levels (if applicable), *etc*. 


### 4. Training and Evaluation

<a href="#top">[Back to top]</a>

**(a). Protocol-1: Same Data Domain, Differenet Forgery Types**

First, you can run the following lines to train a model (*e.g.,* I3D model):
```
python3 -m torch.distributed.launch --nproc_per_node=8 training/train.py \
--detector_path ./training/config/detector/i3d.yaml \
--train_dataset FSAll_ff \
--test_dataset FSAll_ff \
--ddp
```

Note, we here perform both training and testing on FSAll_ff (using all FS data of FF domain as the *evaluation set*) to select the best checkpoint. Once finished training, you can use the best checkpoint to evaluate other testing datasets.

```
python3 training/test.py \
--detector_path ./training/config/detector/i3d.yaml \
--test_dataset "FSAll_ff" "FRAll_ff" "EFSAll_ff" \
--weights_path ./training/weights/i3d.pth
```
Then, you can obtain all evaluation results reported in Tab. 3 of the manuscript.



**(b). Protocol-2: Same Forgery Types, Differenet Data Domain**
Similarly, you can run the following lines for Protocol-2.

```
python3 training/test.py \
--detector_path ./training/config/detector/i3d.yaml \
--test_dataset "FSAll_cdf" "FRAll_cdf" "EFSAll_cdf" \
--weights_path ./training/weights/i3d.pth
```
Then, you can obtain all evaluation results reported in Tab. 4 of the manuscript.




## 👀 More visual examples
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

