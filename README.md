# CMFPNet: A Cross-Modal Multidimensional Frequency Perception Network for Extracting Offshore Aquaculture Areas from MSI and SAR Images
## Abstract：
The accurate extraction and monitoring of offshore aquaculture areas are crucial for the marine economy, environmental management, and sustainable development. Existing methods relying on unimodal remote sensing images are limited by natural conditions and sensor characteristics. To address this issue, we integrated multispectral imaging (MSI) and synthetic aperture radar imaging (SAR) to overcome the limitations of single-modal images. We propose a cross-modal multidimensional frequency perception network (CMFPNet) to enhance classification and extraction accuracy. CMFPNet includes a local–global perception block (LGPB) for combining local and global semantic information and a multidimensional adaptive frequency filtering attention block (MAFFAB) that dynamically filters frequency-domain information that is beneficial for aquaculture area recognition. We constructed six typical offshore aquaculture datasets and compared CMFPNet with other models. The quantitative results showed that CMFPNet outperformed the existing methods in terms of classifying and extracting floating raft aquaculture (FRA) and cage aquaculture (CA), achieving mean intersection over union (mIoU), mean F1 score (mF1), and mean Kappa coefficient (mKappa) values of 87.66%, 93.41%, and 92.59%, respectively. Moreover, CMFPNet has low model complexity and successfully achieves a good balance between performance and the number of required parameters. Qualitative results indicate significant reductions in missed detections, false detections, and adhesion phenomena. Overall, CMFPNet demonstrates great potential for accurately extracting large-scale offshore aquaculture areas, providing effective data support for marine planning and environmental protection.

### The CMFPNet structure.
<div align=center>
<img src=https://github.com/user-attachments/assets/6b6482bc-f70e-4449-9b2e-930db226f01b width=70% />
</div>

### The structure of the LGPB.
<div align=center>
<img src=https://github.com/user-attachments/assets/0c03dffc-b976-4e97-bd2b-e299aa0c1ba6 width=70% />
</div>

### The structure of the MAFFAB.
<div align=center>
<img src=https://github.com/user-attachments/assets/4d37f874-9518-4222-9b51-e6e5a9b7fb32 width=70% />
</div>

## Manuscript link：
https://www.mdpi.com/2072-4292/16/15/2825


## If it is useful for your research or application, please cite it in the following format (BibTeX):
@Article{

rs16152825,

AUTHOR = {Yu, Haomiao and Wang, Fangxiong and Hou, Yingzi and Wang, Junfu and Zhu, Jianfeng and Cui, Zhenqi},

TITLE = {CMFPNet: A Cross-Modal Multidimensional Frequency Perception Network for Extracting Offshore Aquaculture Areas from MSI and SAR Images},

JOURNAL = {Remote Sensing},

VOLUME = {16},

YEAR = {2024},

NUMBER = {15},

ARTICLE-NUMBER = {2825},

URL = {https://www.mdpi.com/2072-4292/16/15/2825},

ISSN = {2072-4292},

DOI = {10.3390/rs16152825}

}

