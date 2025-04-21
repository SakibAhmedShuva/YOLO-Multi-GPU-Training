# YOLO Multi-GPU Training

A repository for training YOLOv11 / YOLOv8 models with multi-GPU support.

## Overview

This repository contains code for training YOLO models using multiple GPUs. It includes scripts for data preparation, model training, and inference. The implementation is designed to work with the Ultralytics YOLO framework.

## My Journey to Multi-GPU YOLO Training

For months, I struggled with getting YOLO to utilize multiple GPUs effectively. Despite having access to dual T4 GPUs, I was frustratingly limited to using only a single GPU for training, which significantly slowed down my experimentation and model development.

After countless hours of testing different configurations, library versions, and approaches, I finally discovered the exact combination that enables true multi-GPU training with YOLO. `torchvision==0.19` was the specific version needed to make everything work together.

This repository shares my solution so others don't have to go through the same lengthy troubleshooting process I endured.

## Requirements

- Python 3.8+
- PyTorch 2.4.1+
- Ultralytics
- torchvision 0.19+
- CUDA-compatible GPU(s)

## Training

### Multi-GPU Training

To train using multiple GPUs:

```bash
yolo segment train data="./data/data.yaml" model="./runs/segment/train/weights/last.pt" epochs=1000 device=0,1 batch=22 workers=22 seed=101 patience=300 resume=True
```

## Parameters

- `data`: Path to the data configuration YAML file
- `model`: Path to the model weights (use pre-trained or checkpoint weights)
- `epochs`: Number of training epochs
- `device`: Devices to use for training (e.g., '0,1' for multiple GPUs)
- `batch`: Batch size
- `workers`: Number of worker threads
- `seed`: Random seed for reproducibility
- `patience`: Early stopping patience
- `resume`: Whether to resume training from the last checkpoint

## Notes

- Disable Weights & Biases logging with `os.environ['WANDB_MODE'] = 'disabled'`
- For optimal performance, adjust batch size and worker count based on your hardware
- Use the resume feature to continue training from previous checkpoints

## License

MIT

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO implementation
