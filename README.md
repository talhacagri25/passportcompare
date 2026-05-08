# Passport Fraud Detection

Single-file passport fraud detection: scans a directory of passport images,
finds faces, embeds them with AdaFace IR-101 (WebFace12M), and outputs ranked
candidate pairs where the same person appears under different passport identities.

## Setup

```bash
# 1) Clone the repo (SCRFD detector ships with it, ~17 MB)
git clone https://github.com/talhacagri25/passportcompare.git
cd passportcompare

# 2) Download AdaFace IR-101 weights (~250 MB, one-time)
mkdir -p checkpoints
wget -O checkpoints/adaface_ir101_webface12m.pt \
  https://github.com/talhacagri25/passportcompare/releases/download/v1.0/adaface_ir101_webface12m.pt

# 3) Install Python deps
pip install -r requirements.txt
```

After step 2, no further network access is required at runtime. If the .pt file
is missing the script exits with a clear error showing the wget command above.

## Run

Defaults:
- Input:  `/TeftisDataScience/BerkayDeneme/Pasaport_Resim_Kontrol/Indirilen_Pasaport`
- Output: `/TeftisDataScience/BerkayDeneme/TalhaOutput`

```bash
python passport_fraud.py
```

Override:
```bash
python passport_fraud.py --input /path/to/images --output /path/to/out --threshold 0.20
```

**Original input files are never modified.** TIF/JPG/PNG inputs are read only;
the output dir gets fresh JPG copies of the suspect-pair images for review.

## Output

```
TalhaOutput/
├── candidates.xlsx          # Excel: rank, cosine, file names, paths (sortable)
├── candidates.csv           # CSV mirror
├── candidates.html          # Open in a browser for side-by-side review
├── summary.txt              # Run statistics + cosine bucket counts
├── fraud_candidates/        # JPG copies of every suspect pair, named by suspicion
│   ├── 0001_0.756_pid_a.jpg
│   ├── 0001_0.756_pid_b.jpg
│   ├── 0002_0.612_pid_c.jpg
│   └── ...
├── embeddings.npy           # (N, 512) cached for re-runs
├── passport_ids.txt
├── failed_detection.txt
└── index.faiss
```

The `fraud_candidates/` filenames begin with `<rank>_<cosine>_` so a file
browser sorts the most suspicious pairs first by default.

`candidates.xlsx` columns:
`rank | cosine | passport_a | passport_b | review_image_a | review_image_b | original_a | original_b`

## Threshold

`--threshold 0.20` is the default starting point. The right operating point
depends on your real passport quality distribution. Calibrate by:

1. Collect ~200-500 labelled pairs (known same-person and known different-person).
2. Run the pipeline, then sweep thresholds in [0.10, 0.45].
3. Pick the threshold where precision/recall trade-off matches your tolerance
   for false positives.

Typical reference points (from independent benchmarks, NOT a guarantee for your data):
- XQLFW (real low-q LFW): 0.10–0.15 → ~93% TPR @ 10% FPR
- Synthetic dev passports: 0.30 → ~100% recall

## Resume

Re-running on the same input directory uses cached `embeddings.npy` if the
passport set is unchanged. Force re-extraction with `--no-resume`.

## Hardware

- CUDA recommended (~16 ms/image on RTX 4070 Super, faster on A100)
- CPU fallback works but is roughly 5-10× slower
