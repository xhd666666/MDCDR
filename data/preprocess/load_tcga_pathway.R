if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

BiocManager::install("clusterProfiler")

# Load the GSVA package
library(GSVA)
library(GSEABase)
library(clusterProfiler)


# Rows are sample names, columns are gene names
train_data <- read.csv('EXP/tcga_expression_filtered.csv', check.names = F, row.names = 1) 
colnames(train_data) <- gsub('.', '-', colnames(train_data), fixed = T)
train_data <- t(train_data)
train_data <- as.matrix(train_data)

# Calculate activity scores of each gene set in each sample
ref_gmt <- read.gmt("c2.cp.v2022.1.Hs.symbols.gmt")
gene_list <- split(as.matrix(ref_gmt)[, 2], ref_gmt[, 1])

# Calculate activity scores of each gene set in each sample
gsvapar <- gsvaParam(as.matrix(train_data), gene_list, minSize = 5)
gsva_es <- gsva(gsvapar)
gsva_res <- as.data.frame(gsva_es)

# Save results to CSV file
write.csv(gsva_res, "tcga_pathway.csv", row.names = TRUE)



