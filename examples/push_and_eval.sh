#!/bin/bash
# Script untuk Upload hasil training manual ke HuggingFace via Docker dan menjalankan Eval

# --- KONFIGURASI ---
# GANTI DENGAN DATA BAPAK (Bisa dicek di .env Bapak)
TASK_ID="83d2264b-9aff-475a-856a-b1705635ce66"
EXPECTED_REPO_NAME="animagine-optuna-winner"
HUGGINGFACE_TOKEN="MASUKKAN_TOKEN_HF_WRITE_BAPAK"
HUGGINGFACE_USERNAME="Gege24"

# Path lokal hasil training barusan
OUTPUTS_DIR="$(pwd)/checkpoints"
LOCAL_FOLDER="/app/checkpoints/$TASK_ID/$EXPECTED_REPO_NAME"

echo "🚀 Memulai proses upload 'Docker-Style'..."

docker run --rm \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --env HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" \
  --env HUGGINGFACE_USERNAME="$HUGGINGFACE_USERNAME" \
  --env TASK_ID="$TASK_ID" \
  --env EXPECTED_REPO_NAME="$EXPECTED_REPO_NAME" \
  --env LOCAL_FOLDER="$LOCAL_FOLDER" \
  --env HF_REPO_SUBFOLDER="checkpoints" \
  --name hf-uploader-eval \
  hf-uploader

echo ""
echo "✅ Jika upload di atas sukses, silakan jalankan perintah Eval ini:"
echo "----------------------------------------------------------------"
echo "python3 -m utils.run_evaluation --task_id \"$TASK_ID\" --models \"$HUGGINGFACE_USERNAME/$EXPECTED_REPO_NAME\""
echo "----------------------------------------------------------------"
