# On-the-fly Data Augmentation for Time Series Forecasting

This repository contains the implementation and experimental setup for the paper "On-the-fly Data Augmentation for Forecasting with Deep Learning" (Cerqueira et al., 2024).

## Overview

This research explores the application of on-the-fly data 
augmentation techniques for improving deep learning-based 
time series forecasting. The implementation leverages 
data augmentation methods from the [metaforecast](https://github.com/vcerqueira/metaforecast) package, applying them dynamically during model training.

## Getting Started

### Prerequisites

- Python 3.9+
- PyTorch
- metaforecast
- Install dependencies listed in `requirements.txt` using:

```bash
pip install -r requirements.txt
```

## Running Experiments

To reproduce the experiments from the paper:

1. Execute the main experimental pipeline:
```bash
python scripts/experiments/run.py
```

2. Analyze the results:
```bash
python scripts/experiments/analyse.py
```

## Data Augmentation Methods

The experiments use the following augmentation techniques:
- Jittering
- Seasonal Moving Blocks Bootstrap
- TSmixup
- DTW Barycentric Averaging
- Time-warping
- Magnitude-warping
- Scaling

## Citation

If you use this code in your research, please cite:

```bibtex
@article{cerqueira2024fly,
  title={On-the-fly Data Augmentation for Forecasting with Deep Learning},
  author={Cerqueira, V{\'i}tor and Santos, Miguel and Baghoussi, Yassine and Soares, Carlos},
  journal={arXiv preprint arXiv:2404.16918},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or feedback about this implementation, please open an issue in this repository.
