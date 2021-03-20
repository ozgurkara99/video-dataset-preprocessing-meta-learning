# Unofficial implementation for pre-processing and dataloader of Something-something-100 dataset
Something-something-v2 video dataset is splitted into 3 meta-sets including overall 100 classes according to CMU [1](https://openaccess.thecvf.com/content_ECCV_2018/html/Linchao_Zhu_Compound_Memory_Networks_ECCV_2018_paper.html), namely: meta-training, meta-validation, meta-test. The code also provides a dataloader in order to create episodes considering given n-way k-shot learning task. Videos are converted to the frames under sparse-sampling protocol described in TSN [2](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cao_Few-Shot_Video_Classification_via_Temporal_Alignment_CVPR_2020_paper.pdf)

# References
[1] https://openaccess.thecvf.com/content_ECCV_2018/html/Linchao_Zhu_Compound_Memory_Networks_ECCV_2018_paper.html
[2] https://openaccess.thecvf.com/content_CVPR_2020/papers/Cao_Few-Shot_Video_Classification_via_Temporal_Alignment_CVPR_2020_paper.pdf

