# Unofficial implementation for pre-processing and dataloader of Something-something-100 dataset
Something-something-v2 video dataset is splitted into 3 meta-sets, namely, meta-training, meta-validation, meta-test. Overall, dataset includes 100 classes that are divided according to CMU [1]. The list files can be found here: https://github.com/ffmpbgrnn/CMN  
The code also provides a dataloader in order to create episodes considering given n-way k-shot learning task. Videos are converted to the frames under sparse-sampling protocol described in TSN [2]  
Something-something-v2 dataset can be found here: https://20bn.com/datasets/something-something

# Usage
```
python main.py [options]
```

```
Options:
  --target          Target folder to which meta-sets copied
  --val             Validation set list, can be found in: https://github.com/ffmpbgrnn/CMN
  --test            Test set list, can be found in: https://github.com/ffmpbgrnn/CMN
  --train           Train set list, can be found in: https://github.com/ffmpbgrnn/CMN 
  --src             Source directory where all videos are available. Can be downloaded from: https://20bn.com/datasets/something-something
  --k               k-shot 
  --n               n-way
  --T               Videos are splitted into T number of segments under sparse-sampling protocol
```

# References
[1] https://openaccess.thecvf.com/content_ECCV_2018/html/Linchao_Zhu_Compound_Memory_Networks_ECCV_2018_paper.html  
[2] https://openaccess.thecvf.com/content_CVPR_2020/papers/Cao_Few-Shot_Video_Classification_via_Temporal_Alignment_CVPR_2020_paper.pdf

