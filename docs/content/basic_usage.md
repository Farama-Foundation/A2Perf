---
layout: "contents"
title: Basic Usage
firstpage:
---

# Basic Usage

Test

# Installation

To install A2Perf, the easiest way is to use `pip`. You can install specific
domains or all domains depending on your needs:

```bash
# Install all domains
pip install a2perf[all]

# Install specific domains
pip install a2perf[circuit-training]
pip install a2perf[web-navigation]
pip install a2perf[quadruped-locomotion]
```

## Installing from source

If you would like to install A2Perf from source, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/Farama-Foundation/A2Perf.git
cd A2Perf
git submodule update --init --recursive
pip install .
```

If you want to install the package in development mode, use the `-e` flag:

```bash
pip install -e .
```
