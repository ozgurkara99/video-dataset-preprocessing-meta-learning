# Unofficial implementation for pre-processing and dataloader of Something-something-100 dataset
Something-something-v2 video dataset is splitted into 3 meta-sets, namely, meta-training, meta-validation, meta-test. Overall, dataset includes 100 classes that are divided according to CMU [1]. The list files can be found here: https://github.com/ffmpbgrnn/CMN  
The code also provides a dataloader in order to create episodes considering given n-way k-shot learning task. Videos are converted to the frames under sparse-sampling protocol described in TSN [2]  
Something-something-v2 dataset can be found here: https://20bn.com/datasets/something-something  
Torch dataset implementation can be found in torch_loader.py file where torch.utils.data.Dataset is utilized to create a dataset and it contains pre-determined number of episodes.
# Usage
```
pip install -r requirements.txt
```

```
python main.py [options]
```

```
  Options:     Default Values:                                   Description:
  --target     dataset                                           Target folder to which meta-sets copied
  --val        smsm-100/val.list                                 Validation set list, can be found in: https://github.com/ffmpbgrnn/CMN
  --test       smsm-100/test.list                                Test set list, can be found in: https://github.com/ffmpbgrnn/CMN
  --train      smsm-100/train.list                               Train set list, can be found in: https://github.com/ffmpbgrnn/CMN 
  --src        something v2/20bn-something-something-v2/         Source directory where all videos are available. 
  --k          1                                                 k-shot 
  --n          5                                                 n-way
  --T          8                                                 Videos are splitted into T number of segments under sparse-sampling protocol
  --ep_num     1000                                              Episode number for training
```
 
# References
[1] Zhu, L., & Yang, Y. (2018). Compound Memory Networks for Few-shot Video Classification. In Proceedings of the European Conference on Computer Vision (ECCV).  
[2] Wang, L., Xiong, Y., Wang, Z., Qiao, Y., Lin, D., Tang, X., & Gool, V. (2016). Temporal Segment Networks: Towards Good Practices for Deep Action Recognition. ECCV.

