# k-pops
A kernel-based gene prioritization tool

K-PoPS is based on and follows the general framework of [PoPS](https://github.com/FinucaneLab/pops/tree/master)
. It is designed for gene prioritization based on functional genomic data, and additionally allows for interpretation of predictions. 

K-PoPS has 3 steps, 

In Step 0, use the `src/munge_feature_files.py` from PoPS to process the functional genomic data in a tabular format: 

```
python munge_feature_directory.py \
 --gene_annot_path example/data/utils/gene_annot_jun10.txt \
 --feature_dir example/data/features_raw/ \
 --save_prefix example/data/features_munged/pops_features \
 --max_cols 500
```

K-PoPS doesn't directly use these genetic features, but instead construct a kernel from these genetic features. We have prepared a script to create kernel: 

```
python prepare_kernel.py --prefix /data/to/pops_features
```

In Step 1, follow PoPS to run MAGMA: 

```
./magma \
 --bfile {PATH_TO_REFERENCE_PANEL_PLINK} \
 --gene-annot {PATH_TO_MAGMA_ANNOT}.genes.annot \
 --pval {PATH_TO_SUMSTATS}.sumstats ncol=N \
 --gene-model snp-wise=mean \
 --out {OUTPUT_PREFIX}
```




In Step 2, run K-PoPS:

The main difference is that we don't have the feature selection step. If you wish to see the top contributor genes, use `--use_explain_mode`, and use `--explain_n_gene` and `--explain_n_contributor` to specify the top genes and top contributors to output. It creates a network in `xx.netwk` with `gene, contributor, score` as the header

```
python k-pops.py \
    --gene_annot_path "$gene_annot_file"   \
    --kernel_mat_prefix "$kernel_prefix"    \
    --use_magma_covariates    \
    --project_out_covariates_remove_hla    \
    --training_remove_hla  \
    --magma_prefix "$magma_prefix"    \
    --device cuda    \
    --use_explain_mode    \
    --explain_n_gene 5 \
    --explain_n_contributor 5 \
    --explain_remove_hla \
    --out_prefix "${output_trait_dir}/kernel"
```











