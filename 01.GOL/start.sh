#!/usr/bin/bash

# file start.sh
# Justificativa técnica para o uso da partição $SCRATCH no SDumont.
# A submissão de jobs deve ser feita obrigatoriamente a partir do $SCRATCH pelos seguintes motivos:
# 1. DESEMPENHO (I/O Paralelo): O $SCRATCH utiliza o sistema de arquivos Lustre, projetado 
# para alta vazão de dados. O $HOME utiliza NFS, que não suporta acessos simultâneos 
# de múltiplos nós de computação, gerando gargalos severos.
# 2. ARQUITETURA DE REDE: Os nós de computação (que possuem as GPUs) são conectados ao 
# $SCRATCH via rede InfiniBand de baixíssima latência. O acesso ao $HOME é feito por 
# uma rede administrativa mais lenta.
# 3. ESCALABILIDADE: Rodar jobs no $HOME pode travar o servidor de arquivos para todos 
# os usuários. O $SCRATCH distribui os dados em vários discos (OSTs), permitindo 
# leituras e escritas paralelas eficientes.
# O $SCRATCH é temporário e NÃO possui backup. Após o término do job, mova os 
# resultados importantes de volta para o $HOME.
 

TARGET='/scratch/treinamento/marcelo.zamith2/2026-LNCC-bootcamp/GOL'
echo $TARGET
mkdir -p $TARGET

rsync -av --progress# $TARGET/
sbatch GOL.srm

