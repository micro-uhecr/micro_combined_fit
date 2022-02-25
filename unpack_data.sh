#!/bin/bash
# Update files list with: find . -iname "*.zst"
zst_files='./Tensor/A_14_z_1.npz.zst
./Tensor/A_1_z_1.npz.zst
./Tensor/A_4_z_1.npz.zst
./Tensor/A_28_z_1.npz.zst
./Tensor/A_56_z_1.npz.zst
./Tensor_creation/Intermediate_MergedInput/SimProp_A_12_zmax_2.5.dat.zst
./Tensor_creation/Output_Tensor/SimPropA_12_z_2.5.npz.zst
./Tensor_creation/Input_Simulations/SimProp_12_0.10_0.20_1.txt.zst
./Tensor_creation/Input_Simulations/SimProp_12_0.20_0.30_1.txt.zst
./Tensor_creation/Input_Simulations/SimProp_12_0.50_2.50_1.txt.zst
./Tensor_creation/Input_Simulations/SimProp_12_0.30_0.50_1.txt.zst
./Tensor_creation/Input_Simulations/SimProp_12_0.00_0.01_1.txt.zst
./Tensor_creation/Input_Simulations/SimProp_12_0.05_0.10_1.txt.zst
./Tensor_creation/Input_Simulations/SimProp_12_0.01_0.05_1.txt.zst
./Catalog/light_sm_sfr_baryon_od_Resc_Rscreen_merged.dat.zst
./Catalog/light_sfr_cleaned_corrected_cloned_LVMHL.dat.zst
./Catalog/sfrd_local.dat.zst'

for f in $(echo $zst_files)
do
  cmd="unzstd -q --rm  $f"
  echo $cmd
  eval $cmd
done
