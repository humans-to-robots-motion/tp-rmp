# tp-rmp
This repos implements the method for "Learning Task-parametrized Riemannian Motion Policies" from demonstrations.

## Installation

```bash
git clone https://github.com/humans-to-robots-motion/tp-rmp
cd tp-rmp
pip install -r requirements.txt
```

## Usage

To see the reproduction of picking skill under dynamic task situations, e.g. pick moving object, please run:

```bash
python scripts/test_tprmp_moving.py
```

For additionally avoiding obstables, please run:

```bash
python scripts/test_tprmp_with_rmpflow_moving.py
```