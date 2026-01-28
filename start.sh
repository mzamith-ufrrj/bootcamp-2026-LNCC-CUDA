#!/usr/bin/bash
#Script para submeter os jogs

TARGET='/scratch/treinamento/marcelo.zamith2/2026-LNCC-bootcamp/GOL'
echo $TARGET
mkdir -p $TARGET

rsync -av --progress * $TARGET/
sbatch GOL.srm

