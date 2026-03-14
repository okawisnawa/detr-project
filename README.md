# DETR-Based Precision Agriculture: Cocoa Disease Detection

Mobile Deep Learning System for Early Detection of Cocoa Diseases Using DETR (DEtection TRansformer)

---

## Authors

**I Putu Oka Wisnawa, S.Kom., M.T.**
**Ni Luh Putu Listya Dewi**

---

# 1. Research Overview / Gambaran Riset

## English

This research focuses on the development of a **mobile-based precision agriculture system** for early detection of cocoa fruit diseases using deep learning.

The system utilizes the **DETR (DEtection TRansformer)** architecture to detect and classify two major cocoa diseases:

* **Anthracnose**
* **Black Pod Rot**

The trained detection model is designed to be deployed into a **Flutter-based mobile application**, enabling real-time inference directly on mobile devices. The goal of this research is to support **precision agriculture practices** by providing early disease detection tools for farmers and agricultural researchers.

---

## Bahasa Indonesia

Penelitian ini berfokus pada pengembangan **sistem pertanian presisi berbasis mobile** untuk deteksi dini penyakit buah kakao menggunakan deep learning.

Sistem ini menggunakan arsitektur **DETR (DEtection TRansformer)** untuk mendeteksi dan mengklasifikasikan dua penyakit utama pada buah kakao:

* **Antraknosa (Anthracnose)**
* **Busuk Buah (Black Pod Rot)**

Model deteksi yang telah dilatih dirancang untuk diimplementasikan ke dalam **aplikasi mobile berbasis Flutter**, sehingga memungkinkan proses inferensi secara **real-time pada perangkat seluler**. Tujuan penelitian ini adalah mendukung **pertanian presisi** melalui teknologi deteksi penyakit tanaman secara dini.

---

# 2. Repository Structure / Struktur Repositori

```
detr-project/
├── .devcontainer/
│   ├── devcontainer.json
│   └── Dockerfile
├── configs/
│   ├── dataset/
│   │   ├── coco_detection.yaml
│   │   └── brats_medical.yaml
│   ├── model/
│   │   └── detr_resnet50.yaml
│   └── experiment/
├── data/
│   ├── raw/
│   └── processed/
├── deployment/
├── scripts/
│   ├── train_dist.sh
│   └── setup_env.sh
├── src/
│   ├── datasets/
│   ├── engine/
│   ├── models/
│   │   ├── backbone/
│   │   ├── transformer/
│   │   ├── criterion.py
│   │   ├── matcher.py
│   │   └── position_encoding.py
│   └── utils/
├── tests/
├── main.py
├── environment.yml
├── .gitignore
└── README.md
```

---

# 3. Model Architecture / Arsitektur Model

## English

This project implements the **DETR (DEtection TRansformer)** architecture which combines:

* **CNN Backbone** (ResNet-50) for feature extraction
* **Transformer Encoder-Decoder** for global context modeling
* **Hungarian Matching Loss** for optimal prediction-target pairing

Main components include:

**Backbone**

ResNet-50 extracts spatial feature maps from the input images.

**Transformer Encoder**

Processes image features as a sequence and models global dependencies.

**Transformer Decoder**

Uses object queries to predict bounding boxes and object classes.

**Prediction Heads**

* Bounding Box Regression
* Object Class Prediction

---

## Bahasa Indonesia

Proyek ini mengimplementasikan arsitektur **DETR (DEtection TRansformer)** yang menggabungkan:

* **CNN Backbone** (ResNet-50) untuk ekstraksi fitur
* **Transformer Encoder-Decoder** untuk pemodelan konteks global
* **Hungarian Matching Loss** untuk mencocokkan prediksi dengan target secara optimal

Komponen utama:

**Backbone**

ResNet-50 mengekstraksi fitur spasial dari citra masukan.

**Transformer Encoder**

Memproses fitur citra sebagai urutan dan memodelkan dependensi global.

**Transformer Decoder**

Menggunakan object queries untuk memprediksi bounding box dan kelas objek.

---

# 4. Dataset Structure / Struktur Dataset

The dataset used in this project follows the **COCO Detection Format**.

Dataset directories:

```
data/
├── raw/
├── processed/
```

### raw/

Contains the **original collected dataset** from field observations or image acquisition.

Example:

```
data/raw/
    cocoa_images/
    annotations_raw/
```

### processed/

Contains the **preprocessed dataset ready for training**, typically converted to COCO format.

Example:

```
data/processed/
    images/
    annotations/
        instances_train.json
        instances_val.json
```

---

# 5. Using .gitkeep for Dataset Directories

## English

The dataset directories contain a `.gitkeep` file to preserve the directory structure in the repository while keeping the repository lightweight.

Git does not track empty directories. Therefore `.gitkeep` files are used as placeholders.

Instructions:

1. Navigate to the dataset directory:

```
cd data/raw
cd data/processed
```

2. Place your dataset files inside the corresponding folders.

3. Do **not remove the directory structure**, as it is required by the training pipeline.

4. Large datasets should **not be committed directly to Git**.

Recommended alternatives:

* Git LFS
* Google Drive
* Institutional data repository

---

## Bahasa Indonesia

Direktori dataset berisi file `.gitkeep` untuk mempertahankan struktur folder dalam repositori tanpa harus menyimpan dataset besar di dalam Git.

Git tidak melacak direktori kosong, sehingga `.gitkeep` digunakan sebagai file penanda.

Instruksi penggunaan:

1. Masuk ke direktori dataset:

```
cd data/raw
cd data/processed
```

2. Letakkan dataset Anda pada folder yang sesuai.

3. Jangan menghapus struktur folder karena diperlukan oleh pipeline training.

4. Dataset berukuran besar **tidak disarankan disimpan langsung di Git**.

Alternatif penyimpanan yang direkomendasikan:

* Git LFS
* Google Drive
* Repository data institusi

---

# 6. Environment Setup / Pengaturan Environment

## English

This project is highly optimized for **micromamba** to ensure fast, lightweight, and isolated environment management, particularly when working within a **distrobox container**.

## Bahasa Indonesia

Proyek ini dioptimalkan untuk penggunaan **micromamba** guna memastikan pengelolaan environment yang **cepat, ringan, dan terisolasi**, terutama saat bekerja di dalam **container distrobox**.

Using micromamba significantly reduces dependency resolution time compared to traditional Conda environments and ensures reproducible builds for machine learning experiments.

Penggunaan micromamba secara signifikan mengurangi waktu resolusi dependensi dibandingkan lingkungan Conda tradisional serta membantu memastikan eksperimen machine learning dapat direproduksi dengan konsisten.

---

# Quick Setup (Recommended) / Setup Cepat (Direkomendasikan)

## English

Run the provided automation script to set up the environment and verify your **NVIDIA hardware compatibility**.

## Bahasa Indonesia

Jalankan skrip otomatisasi yang disediakan untuk menyiapkan environment serta memverifikasi kompatibilitas **perangkat keras NVIDIA** Anda.

```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

The script will automatically:

* Create the micromamba environment
* Install required dependencies
* Verify CUDA availability
* Validate GPU compatibility

Skrip ini akan secara otomatis:

* Membuat environment micromamba
* Menginstal seluruh dependensi yang diperlukan
* Memverifikasi ketersediaan CUDA
* Memastikan kompatibilitas GPU

---

# Manual Installation (Using Micromamba)

# Instalasi Manual (Menggunakan Micromamba)

**English**

If you prefer to configure the environment manually, run the following commands:

**Bahasa Indonesia**

Jika Anda ingin melakukan konfigurasi environment secara manual, jalankan perintah berikut:

```bash
# Create the environment from the YAML configuration file
micromamba create -f environment.yml -y

# Activate the environment
micromamba activate detr-env
```

After activation, verify that the environment has been successfully created:

Setelah aktivasi, pastikan environment telah berhasil dibuat:

```bash
python --version
```

---

# Environment Reproducibility / Reproduksibilitas Environment

This repository provides the full environment specification in:

```
environment.yml
```

This file ensures that all dependencies required for training and deployment can be reproduced consistently across different systems.

File ini memastikan bahwa seluruh dependensi yang diperlukan untuk proses pelatihan dan deployment dapat direproduksi secara konsisten pada berbagai sistem.

---

# 7. Training Pipeline / Pipeline Pelatihan Model

To train the DETR model:

```
bash scripts/train_dist.sh
```

Or run directly:

```
python main.py \
  --config configs/model/detr_resnet50.yaml \
  --dataset configs/dataset/coco_detection.yaml
```

Training includes:

* Data loading
* Model training
* Validation
* Evaluation metrics computation

---

# 8. Evaluation Metrics / Metode Evaluasi

Model performance is evaluated using **COCO Detection Metrics**.

Primary metrics:

* **mAP (Mean Average Precision)**
* **AP50**
* **AP75**
* **Precision**
* **Recall**

These metrics evaluate the ability of the model to correctly detect diseased cocoa fruits.

---

# 9. Deployment Pipeline / Deployment Model

After training, the model can be exported for mobile deployment.

```
deployment/
```

Supported formats include:

* ONNX
* TensorRT
* Mobile inference engines

The optimized model will be integrated into a **Flutter mobile application** for real-time disease detection.

---

# 10. Research Applications / Aplikasi Penelitian

This research contributes to:

* Precision agriculture systems
* Smart farming technology
* Early detection of plant diseases
* AI-assisted crop monitoring

---

# 11. Citation

If you use this work in your research, please cite it as follows:

```
@researchproject{wisnawa2026cocoa,
  author = {Wisnawa, I Putu Oka and Dewi, Ni Luh Putu Listya},
  title = {DETR-Based Object Detection for Early Identification of Cocoa Fruit Diseases in Precision Agriculture},
  year = {2026},
  institution = {Politeknik Negeri Bali},
  keywords = {DETR, Precision Agriculture, Cocoa Disease Detection, Computer Vision}
}
```

---

# 12. License

This repository is intended for **academic research and precision agriculture development**.

Please contact the authors for collaboration or dataset usage permissions.
