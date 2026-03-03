# MDCDR

## Environment Setup

Please install the environment using anaconda3;

conda create -n project python=3.9 

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 
pip install torch-geometric==2.3.1 
pip install numpy==1.25.2 pandas==2.1.0 gseapy==1.0.6 
pip install scikit-learn==1.3.0 scipy==1.11.2 rdkit==2023.3.3 
pip install openpyxl pubchempy subword_nmt

## Data Preparation

processed and initial data files are available at: 
https://drive.google.com/file/d/14i13ogH6yIARR8RkAI_CzYXXDI278utb/view?usp=drive_link 
or 
https://pan.baidu.com/s/1KdemH7HhDHbeXqlGenCUWg?pwd=ehyc

`1-deal_mut.py` Download and process cell line genomic mutation data.

`2-deal_cnv.py` Download and process cell line copy number variation data.

`3-deal_exp.py` Download and process cell line gene expression data.

`4-deal_GDSC_cls.py` Process the GDSC classification dataset.

`4-deal_GDSC2.py` Process the GDSC2 regression dataset.

`4-deal_TCGA.py` Process the TCGA classification dataset.

`5-filter_GDSC_cls.py` Filter cell lines and drugs for the GDSC classification dataset.

`5-filter_GDSC2.py` Filter cell lines and drugs for the GDSC2 regression dataset.

`5-filter_TCGA.py` Filter cell lines and drugs for the TCGA classification dataset.

`6-drug_response.py` Organize the response values in the dataset and filter out missing values.

`7-deal_ppi.py` Download PPI data.

`load_gdsc_cls_pathway.R` Calculate pathway activity scores using gene expression data of cell lines from the GDSC classification dataset.

`load_gdsc_regr_pathway.R` Calculate pathway activity scores using gene expression data of cell lines from the GDSC2 regression dataset.

`load_tcga_pathway.R` Calculate pathway activity scores using gene expression data of cell lines from the TCGA classification dataset.

`extract_gin.py` Obtain pre-trained features using GIN.

## Usage Examples

`main_GDSC_cls.py` 5-fold cross-validation on the GDSC classification dataset.

`main_GDSC_regr.py` 5-fold cross-validation on the GDSC regression dataset.

`main_GDSC_regr_leave_cell.py` Perform leave-cell-out experiments on the GDSC regression dataset with 5 random splits.

`main_GDSC_regr_leave_drug.py` Perform leave-drug-out experiments on the GDSC regression dataset with 5 random splits.

`main_TCGA_cls.py` 5-fold cross-validation on the TCGA classification dataset.

`main_TCGA_cls_leave_cell.py` Perform leave-cell-line-out experiments on the TCGA classification dataset with 5 random splits.

`main_TCGA_cls_leave_drug.py` Perform leave-drug-out experiments on the GDSC regression dataset with 5 random splits.

