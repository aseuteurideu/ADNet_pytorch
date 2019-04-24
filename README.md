# ADNet-pytorch
Implementation of ADNet (https://sites.google.com/view/cvpr2017-adnet) in PyTorch 0.4.1.

References:
1. ADNet Matlab code (https://github.com/hellbell/ADNet). From my test, published weight has distance precision ~76%
2. ADNet IEEE transaction on Neural Networks and Learning Systems 2018 (https://ieeexplore.ieee.org/abstract/document/8306309)
3. ADNet CVPR 2017 (http://openaccess.thecvf.com/content_cvpr_2017/papers/Yun_Action-Decision_Networks_for_CVPR_2017_paper.pdf)

This implementation still cannot reproduce same performance with paper. 
Current performance distance (20px) precision with 1x test trial: 

|SL     |SL+RL  |SL, nomd |SL+RL, nomd |
|-------|-------|---------|------------|
|75.3%  |67.9%  |54.9%    |54.1%       |

SL: Supervised Learning. RL: Reinforcement Learning, nomd: without multi-domain training

Inputs are welcome (especially RL part). Also currently cannot use multiple GPU (set CUDA_VISIBLE_DEVICES environment variable to select one GPU, if there are multiple GPUs in the system)

TODO:
* RL training debugging
* multiple GPU
* ALOV dataset training (to follow paper)

Requirements:
* Python 3.6
* PyTorch 0.4.1
* Cuda 9.0
* vggm.pth weight (https://data.lip6.fr/cadene/pretrainedmodels/vggm-786f2434.pth)
* (optional for visualization) tensorboardx (https://github.com/lanpa/tensorboardX) and tensorflow

## Structure
```
├── datasets 
        (wrapper for datasets)
        ├── data
                ├── otb   
                        ├── Basketball
                                ├── img
                                        ├── ....jpg
                                ├── groundtruth_rect.txt
                        ├── Biker
                                ├── img
                                        ├── ....jpg
                                ├── groundtruth_rect.txt
                        ├── (continue till last class)
                                
                ├── vot13
                        ├── bicycle
                                ├── ....jpg
                                ├── camera_motion.label
                                ├── groundtruth.txt
                                ├── (and other files)
                        ├── bolt    
                                ├── (similar structure with bicycle)
                        ├── (continue till last class)
                ├── vot14
                        ├── (similar structure with vot13)
                ├── vot15
                        ├── (similar structure with vot13)
├── mains
        (main.py for various datasets and tasks)
        ├── result_on_test_images
        ├── weights
        ├── (various main python files)
├── models
        (network descriptions)
├── options
        (template parser for command line inputs)
├── trainers
        (define how model's forward / backward and logs)
        ├── weights
                (the saved trained weights)
├── utils
        (toolbox for drawing and scheduling)
        ├── videolist
                ├── vot13-otb.txt
                ├── vot14-otb.txt
                ├── vot15-otb.txt
        ├── (various python files)
├── scripts
        (scripts for replicating experiement results)
├── work
        (default folder to store logs/models)
├── vggm.pth
        (the vggm weights for the base network)
```

## Usage examples
*  ADNet - train with SL & RL
    ```bash
    python mains/ADNet.py --visualize True
    ```
    
-------------------------------------------

*  ADNet_test
    ```bash
    python ADNet_test.py --save_result_images results_on_test_images --display_images False
    ```

*  ADNet_ratiosamples_0.7
    ```bash
    python ADNet_test.py --save_result_images results_on_test_images --display_images False --pos_samples_ratio 0.7
    ```

-------------------------------------------
*  Examples on creating plot
    ```bash
    python create_plots.py --bboxes_folder results_on_test_images/ADNet_RL_-0.5 --show_plot False --save_plot_folder results_on_test_images/ADNet_RL_-0.5
    ```
    