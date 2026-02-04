# 🧠 Multimodal Brain Tumor Segmentation

Automatic segmentation of gliomas in pre-operative MRI scans using a **U-Net** based semantic segmentation pipeline.

---

## 📂 Dataset Overview

The data is collected from the **Multimodal Brain Tumor Segmentation Challenge 2018 (BraTS)**.

* [Prepared Dataset](https://drive.google.com/drive/folders/1RSjZ6ASBMSPgUtFQAzvpBx1aW5VXPtAM?usp=sharing)
* [Trained Model Weights](https://drive.google.com/file/d/11WJbOZ9KdNMwNAGX8ZYyyjD1b3nMH4uG/view?usp=sharing)

### Imaging Data Description

1. **Modalities (NIfTI `.nii.gz` format)**:

   * Native (T1)
   * Post-contrast T1-weighted (T1Gd)
   * T2-weighted (T2)
   * T2 FLAIR

2. **Segmentation Labels (BraTS reference)**:

   * GD-enhancing tumor (ET — label 4)
   * Peritumoral edema (ED — label 2)
   * Necrotic and non-enhancing tumor core (NCR/NET — label 1)
   * Background/Remaining Region (label 0)

3. Pre-processing: co-registered to the same anatomical template, interpolated to 1mm³, skull-stripped.

4. Data organization:

   * **LGG** (Lower Grade Glioma)
   * **HGG** (High Grade Glioma)
     Each patient folder contains 4 modalities and segmentation labels.

5. Conversion: `.nii.gz` → 3D NumPy arrays using **SimpleITK**.

6. Data is already **skull-stripped**.

---

## 🎯 Task

Segment gliomas in pre-operative MRI scans using provided **training data** and produce segmentation labels for evaluation.

---

## 🛠️ Methodology

### Data Pre-processing

* Combine each patient’s MRI volume into NumPy arrays `(N, S, H, W, X)`:

  * `N` = number of patients
  * `S` = number of slices per 3D volume
  * `H, W` = slice dimensions
  * `X` = number of modalities

* Memory optimization (Google Colab Free GPU):

  1. HGG (210 patients) divided into 3 sets of 70 patients each: `data11.npy`, `data12.npy`, `data13.npy`
     LGG (75 patients) → `data2.npy`
     Corresponding ground truths: `gt11.npy`, `gt12.npy`, `gt13.npy`, `gt2.npy`
  2. Only mid slices with tumor information (30–120) were selected → reshaped for training:

     * HGG: `(5600, 240, 240, 4)`
     * LGG: `(6750, 240, 240, 4)`
       Ground truth one-hot encoded similarly.
  3. Center-cropped → final size `(192,192,4)`.
  4. Split randomly: **Train:Val:Test = 60%:20%:20%**.

---

### Proposed Model: **U-Net**

A deep convolutional network for semantic segmentation with skip connections for precise tumor boundary extraction.

![](/method.JPG)

---

### Dice Coefficient & Loss Function

* Sørensen-Dice coefficient measures overlap between predicted and ground truth masks.

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/a80a97215e1afc0b222e604af1b2099dc9363d3b)

* Modified Dice for model training:

```python
def dice_coef(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection) / (K.sum(K.square(y_true),axis=-1) + K.sum(K.square(y_pred),axis=-1) + epsilon)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
```

---

## 📈 Results Obtained

### HGG Result Samples

![](https://github.com/as791/Brain-Tumor-Segmentation-BRaTS-18/blob/master/Result%20Samples/HGG-1.png)
![](https://github.com/as791/Brain-Tumor-Segmentation-BRaTS-18/blob/master/Result%20Samples/HGG-2.png)
![](https://github.com/as791/Brain-Tumor-Segmentation-BRaTS-18/blob/master/Result%20Samples/HGG-3.png)

### LGG Result Samples

![](https://github.com/as791/Brain-Tumor-Segmentation-BRaTS-18/blob/master/Result%20Samples/LGG-1.png)
![](https://github.com/as791/Brain-Tumor-Segmentation-BRaTS-18/blob/master/Result%20Samples/LGG-2.png)

---

## 📝 Evaluated Results

| Test Data | Dice Coefficient |
| --------- | :--------------: |
| HGG Set-1 |      0.9795      |
| HGG Set-2 |      0.9855      |
| HGG Set-3 |      0.9793      |
| LGG       |      0.9950      |

---

## 📚 References

1. **BraTS 2017/2018 Dataset**

   * [Official BraTS 2017 Registration & Dataset](https://www.med.upenn.edu/sbia/brats2017/registration.html)
