# Dokumentasi Teknis Sistem Analisis Pencitraan Otak dan Traktografi 3D

Dokumentasi ini dibuat oleh **PT Ananda Technology Solution** sebagai panduan teknis komprehensif untuk implementasi, eksekusi, dan interpretasi klinis/teknis dari sistem *3D Brain Tractography* berbasis Python native.

# Gambar Output
![Gambar_SS_Output]()

# Video presentasi belajar Dipy Brain Analysis Neurosains Otak
[Klik untuk nonton)(https://youtu.be/8ji7o3dB8Sk)

---

## Daftar Isi
1. [Pendahuluan dan Landasan Neurosains](#1-pendahuluan-dan-landasan-neurosains)
2. [Arsitektur Sistem dan Pipeline Data](#2-arsitektur-sistem-dan-pipeline-data)
3. [Dependensi dan Prasyarat Sistem](#3-dependensi-dan-prasyarat-sistem)
4. [Langkah Instalasi dan Menjalankan Program](#4-langkah-instalasi-dan-menjalankan-program)
5. [Analisis Struktur Kode Komputasi](#5-analisis-struktur-kode-komputasi)
6. [Penjelasan Komprehensif Hasil Running Program](#6-penjelasan-komprehensif-hasil-running-program)
7. [Panduan Navigasi Interaktif Visualisasi 3D](#7-panduan-navigasi-interaktif-visualisasi-3d)
8. [Troubleshooting dan Optimasi Performa](#8-troubleshooting-dan-optimasi-performa)

---

## 1. Pendahuluan dan Landasan Neurosains

Sistem ini dirancang untuk memetakan arsitektur struktural substansia alba (*white matter*) pada otak manusia menggunakan data *Diffusion Magnetic Resonance Imaging* (dMRI). Di dalam otak, akson atau serat saraf terbungkus oleh selubung mielin yang bertindak sebagai isolator listrik alami. Struktur ini membatasi pergerakan molekul air sehingga air berdiffusi secara tidak acak (*anisotropic diffusion*). Molekul air akan bergerak lebih bebas searah dengan jalurnya akson dibandingkan menembus dinding mielin.

Melalui pendekatan matematika *Diffusion Tensor Imaging* (DTI), sistem mengukur tensor difusi per voxel untuk mengekstrak nilai *Fractional Anisotropy* (FA) dan *eigenvectors*. Nilai FA berkisar antara 0 (difusi isotropik sempurna seperti cairan serebrospinal) hingga 1 (difusi anisotropik linier sempurna seperti traktus saraf yang sangat rapat). Pemetaan inilah yang digambarkan oleh program sebagai "kabel saraf" atau *streamlines*.

---

## 2. Arsitektur Sistem dan Pipeline Data

Alur pemrosesan data (*data pipeline*) dalam program ini mengikuti standar pemrosesan neurosains modern yang dibagi menjadi empat fase utama:

```
+------------------+     +-----------------------+     +----------------------+     +-----------------------+
|  1. Data Ingest  | --> | 2. Tensor Computation | --> | 3. Local Tractography| --> | 4. Hardware Render    |
| (NIfTI + G-Table)|     |  (Masking & FA Fit)   |     | (Seeding & Tracking) |     |  (Fury OpenGL Window) |
+------------------+     +-----------------------+     +----------------------+     +-----------------------+
```

1. **Data Ingest**: Memuat file citra resonansi magnetik berformat NIfTI (`.nii.gz`) beserta tabel gradien b-values dan b-vectors.
2. **Tensor Computation**: Membangun topeng latar belakang (*background mask*) untuk mereduksi *noise*, menghitung nilai tensor difusi, dan mengekstrak metrik FA.
3. **Local Tractography**: Menentukan *Region of Interest* (ROI) sebagai koordinat awal (*seeds*), mengekstrak puncak orientasi difusi (*peaks*), dan menelusuri jalur serat langkah-demi-langkah menggunakan kriteria batas ambang (*stopping criterion*).
4. **Hardware Render**: Mengonversi koordinat *streamlines* menjadi aktor grafis 3D dan melakukan *rendering* interaktif memanfaatkan kartu grafis (GPU).

---

## 3. Dependensi dan Prasyarat Sistem

Program ini berjalan di atas Python native tanpa memerlukan *wrapper* perangkat lunak eksternal seperti FSL atau MRtrix3. Berikut prasyarat yang harus dipenuhi:

### Perangkat Lunak & Library
* **Python**: Versi `3.10` atau `3.11` (Direkomendasikan).
* **NumPy**: Untuk operasi matriks array multidimensi berkecepatan tinggi.
* **Nibabel**: Untuk membaca struktur metadata koordinat spasial (`affine matrix`) dari file medis.
* **DIPY (Diffusion Imaging in Python)**: Library inti untuk komputasi algoritma neurosains dan traktografi.
* **Fury**: Library *rendering* 3D berbasis VTK (Visualization Toolkit) yang dioptimalkan untuk visualisasi biosains.

### Perangkat Keras (Hardware)
* Kartu grafis (GPU) dengan dukungan driver **OpenGL 3.3** atau versi di atasnya.
* RAM minimal 8 GB (16 GB direkomendasikan untuk dataset beresolusi tinggi).

---

## 4. Langkah Instalasi dan Menjalankan Program

### Langkah 1: Isolasi Lingkungan (Opsional namun Direkomendasikan)
Buat virtual environment baru agar tidak terjadi konflik dengan package lain:
```bash
python -m venv env_brain
source env_brain/bin/activate  # Untuk Linux/macOS
# atau
env_brain/Scripts/activate     # Untuk Windows (Command Prompt)
```

### Langkah 2: Instalasi Dependencies
Instal seluruh ekosistem library neurosains yang dibutuhkan menggunakan `pip`:
```bash
pip install numpy nibabel dipy fury
```

### Langkah 3: Eksekusi Program
Jalankan file utama program menggunakan interpreter Python:
```bash
python brain_analysis.py
```
*Catatan: Pada eksekusi pertama, sistem akan mendownload dataset referensi "Stanford HARDI" secara otomatis sebesar ~87 MB ke dalam direktori lokal komputer Anda.*

---

## 5. Analisis Struktur Kode Komputasi

Berikut penjelasan fungsional dari blok kode utama yang menyusun program `brain_analysis.py`:

* **`download_and_load_data()`**: Fungsi ini bertanggung jawab atas ketersediaan data. Memuat file gambar `.nii.gz` serta koordinat gradien medan magnet `(gtab)` yang merekam arah difusi molekul air.
* **`TensorModel(gtab)`**: Membuat model matematika tensor difusi. Model ini memetakan elipsoid difusi pada setiap voxel di dalam ruang 3D.
* **`ThresholdStoppingCriterion(fa, 0.2)`**: Ini adalah fungsi kontrol batas. Algoritma pelacakan kabel saraf akan otomatis berhenti melacak jika jalur menyentuh voxel dengan nilai FA di bawah 0.2. Hal ini mencegah kabel saraf keluar dari area substansia alba menuju substansia grsea atau cairan otak.
* **`utils.seeds_from_mask(seed_mask, density=1)`**: Menentukan titik awal pelacakan. Di sini kita memotong koordinat voxel spesifik `[40:60, 40:60, 35:45]` yang merupakan representasi area *Corpus Callosum* (jembatan saraf utama penghubung hemisfer kiri dan kanan).
* **`LocalTracking(...)`**: Mesin komputasi utama yang mengintegrasikan arah puncak difusi (*peaks*), batas ambang (*stopping criterion*), titik awal (*seeds*), dan matriks transformasi koordinat (*affine*) untuk menghasilkan ribuan garis traktografi (`streamlines`).

---

## 6. Penjelasan Komprehensif Hasil Running Program

Ketika program berhasil dijalankan, sebuah jendela grafis native berukuran 1200x900 piksel akan muncul di layar. Jendela ini menampilkan visualisasi **3D Brain Tractography** interaktif. Berikut interpretasi detail dari output visual tersebut:

### 1. Representasi Garis (Streamlines)
Setiap garis lengkung yang terlihat di layar merepresentasikan **berkas akson (fasciculus)** atau sekumpulan kabel saraf nyata yang menghubungkan berbagai area korteks otak. Garis-garis ini tidak digambar secara acak, melainkan hasil rekonstruksi dari pergerakan molekul air di dalam jaringan otak pasien.

### 2. Standar Pewarnaan Arah Difusi (DTI Color-Coding)
Warna pada setiap segmen garis ditentukan secara otomatis menggunakan standar neuroimaging internasional berbasis vektor arah (X, Y, Z) yang dipetakan ke ruang warna RGB (Red, Green, Blue):

| Warna | Dimensi Arah Vektor | Deskripsi Jalur Saraf | Contoh Anatomi Otak |
| :--- | :--- | :--- | :--- |
| **Merah (Red)** | Sumbu X (Kiri ke Kanan) | **Komisural**: Menghubщением hemisfer otak kiri dan hemisfer kanan. | *Corpus Callosum* |
| **Hijau (Green)** | Sumbu Y (Depan ke Belakang) | **Asosiasi**: Menghubungkan area anterior (depan) dan posterior (belakang) pada hemisfer yang sama. | *Superior Longitudinal Fasciculus* |
| **Biru (Blue)** | Sumbu Z (Atas ke Bawah) | **Proyeksi**: Menghubungkan pusat korteks bagian atas dengan batang otak atau tulang belakang bagian bawah. | *Corticospinal Tract* / Jalur Motorik |

### 3. Kepadatan dan Lokalisasi Serat
Karena kita menerapkan pembatasan koordinat awal (*seed mask*) pada area `[40:60, 40:60, 35:45]`, visualisasi akan berpusat pada jembatan saraf tengah otak. Anda akan melihat kumpulan serat dominan berwarna merah yang melengkung ke arah samping kanan dan kiri secara simetris, merepresentasikan struktur *Corpus Callosum*.

---

## 7. Panduan Navigasi Interaktif Visualisasi 3D

Jendela visualisasi yang dihasilkan bersifat interaktif penuh menggunakan *mouse* dan *keyboard*. Anda dapat melakukan analisis spasial dengan kontrol berikut:

* **Rotasi Kamera**: Klik kiri tahan pada area kosong, lalu geser *mouse* ke segala arah untuk memutar objek otak dalam ruang 3D.
* **Pergeseran Objek (Pan)**: Klik kanan tahan (atau tekan tombol roda tengah *mouse*), lalu geser *mouse* untuk menggeser posisi kamera ke atas, bawah, kiri, atau kanan.
* **Zoom In / Zoom Out**: Putar roda gulir (*scroll wheel*) ke depan untuk memperbesar gambar saraf guna melihat detail struktur voxel, atau putar ke belakang untuk memperkecil tampilan seluruh anatomi.
* **Reset Tampilan**: Tekan tombol **`R`** pada *keyboard* Anda untuk mengembalikan posisi kamera dan tingkat pembesaran ke kondisi default semula.
* **Keluar dari Aplikasi**: Tekan tombol **`Q`** pada *keyboard* atau klik ikon tanda silang `(X)` pada sudut jendela untuk menutup program dengan aman.

---

## 8. Troubleshooting dan Optimasi Performa

* **Masalah Jendela Hitam / Freeze**: Pastikan driver GPU (NVIDIA/AMD/Intel) Anda sudah terinstal dengan benar dan mendukung akselerasi perangkat keras OpenGL. Jika menggunakan SSH, pastikan *X11 forwarding* aktif atau jalankan langsung secara lokal.
* **Konsumsi Memori Tinggi**: Jika program terasa lambat, Anda dapat menurunkan nilai `density` pada fungsi `seeds_from_mask` menjadi kurang dari 1, atau memperkecil rentang slice koordinat matriks pada `seed_mask`.
