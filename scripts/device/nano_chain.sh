#!/bin/bash
# Nano, after run 5 was stopped (2026-11-01: Mistral moves to the USB stick, 317 MB/s O_DIRECT reads with
# max_sectors_kb 1024 vs the SD card's 70):
#   1 llmrt_kq10 on the 4 resident bundles not yet run with it (Phi-3 coded x2, SmolLM <=1% coded + twin), so the
#     Nano GPU columns come from one binary (kq_out10/, energy_kq10/)
#   2 GPU Mistral coded, both points, streamed from the USB stick at budget 2560, our lookup drafter
#     (stream_usb10/, energy_stream_usb10/)
#   3 llama.cpp CPU Mistral (2 comparators, our 2 GGUFs) paged from the USB stick, greedy then prompt lookup
#     (energy_kq/llama_mistral_usb/)
# Stops if the stick's max_sectors_kb is not 1024 (not persistent across reboots). DMESG_BASE=12: known traces.
set -u
cd "$(dirname "$0")/../.."
N=jetson-nano; export DMESG_BASE=12
usb_ok() { [ "$(ssh $N cat /sys/block/sda/queue/max_sectors_kb)" = 1024 ] && ssh $N 'mountpoint -q /mnt/usb' \
  || { echo "$(date -Is) USB stick not at max_sectors_kb 1024 / not mounted: stop"; exit 1; }; }
BIN=llmrt_kq10 OUT=10 DB=1 LLMRT_ARGS="--drafter ngramcache" scripts/device/nano_kq_driver.sh \
  phi3_kq_IQ4_XS_ref phi3_kq_IQ4_XS_g1_lam0_c1.25 smollm_kq_Q5_K_M_g1_lam0 smollm_kq_Q5_K_M_g1_lam0_pk
echo NANO_USB_1_DONE
usb_ok
NOZC=0 BUDGET_MB=2560 BIN=llmrt_kq10 LLMRT_ARGS="--drafter lookup" scripts/device/nano_mistral_driver.sh usb10 \
  /mnt/usb/rt mistral_kq_IQ4_XS_g1_lam0_c1.5_ilc mistral_kq_IQ4_XS_g1_lam0_ilc
echo NANO_USB_2_DONE
usb_ok
GD=/mnt/usb/gguf D=energy_kq/llama_mistral_usb scripts/device/nano_llama_mistral.sh 0 7
echo NANO_USB_DONE
