#! /bin/bash

# If you need to specify package directories
bedtools=$(which bedtools)
bwa=$(which bwa)

# Command line
hg38=$1

$bedtools getfasta -s -name -fi $hg38 -bed annotation/ensemblannot/L1HSPA2_ensemblfinal.400.bed > annotation/ensemblannot/L1HSPA2_ensemblfinal.400.fa
$bwa index annotation/ensemblannot/L1HSPA2_ensemblfinal.400.fa
