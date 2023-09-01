#!/bin/bash
# Update files list with: find . -iname "*.zst"
zst_files='./Tensor/A_TALYS_14_z_1.npz.zst
./Tensor/A_TALYS_1_z_1.npz.zst
./Tensor/A_TALYS_4_z_1.npz.zst
./Tensor/A_TALYS_28_z_1.npz.zst
./Tensor/A_TALYS_56_z_1.npz.zst
./Catalog/light_sfr_cleaned_corrected_cloned_LVMHL.dat.zst
./Catalog/sfrd_local.dat.zst
./Catalog/smd_local.dat.zst'

for f in $(echo $zst_files)
do
  cmd="zstd -d -q --rm  $f"
  echo $cmd
  eval $cmd
done
