# Gauge Detection Module

## Purpose

This module detects the **sharpness gauge** in Monster Hunter World and classifies it as **red** or **non-red**. The distinction matters because attack strategy changes depending on gauge color — specifically, detecting red gauge and estimating how much red remains helps decide **when to use Helm Breaker**.

## Gauge States

| State | Description | Reference |
|-------|-------------|-----------|
| **Red Gauge (红刃)** | Gauge bar is red/pink, indicating Spirit Gauge is charged. The remaining red length determines Helm Breaker timing. | `gauge photo/红刃.png`, `红刃2.png`, `红刃3.png` |
| **Yellow Gauge** | Non-red state. Normal gauge color. | `gauge photo/yelloGauge.png` |
| **White Gauge** | Non-red state. Base gauge color. | `gauge photo/whiteGauge.png` |
| **No Gauge** | Gauge bar is empty / not visible. | `gauge photo/NoGauge.png` |

## Detection Goals

1. **Binary classification**: Red gauge vs. Non-red gauge (yellow, white, empty)
2. **Red gauge remaining estimation**: When red is detected, measure how much red is left in the bar (percentage or pixel length) to determine optimal Helm Breaker timing

## Approach

The gauge occupies a fixed screen region (bottom-left HUD area). Detection can leverage:

- **Color-based segmentation**: The red gauge has a distinct red/pink hue that separates it clearly from yellow, white, and empty states
- **Bar length measurement**: Once red is confirmed, measure the red pixel extent relative to the full gauge bar length to estimate remaining percentage

## Directory Structure

```
Gauge/
├── README.md              # This file
├── gauge photo/           # Reference screenshots of each gauge state
│   ├── 红刃.png           # Red gauge (full)
│   ├── 红刃2.png          # Red gauge (partial)
│   ├── 红刃3.png          # Red gauge (low)
│   ├── yelloGauge.png     # Yellow gauge
│   ├── whiteGauge.png     # White gauge
│   └── NoGauge.png        # Empty gauge
└── (detector code TBD)
```

## Usage Context

This module is part of the `mh_w_2.1` project, which combines:
- **Hit number detection** (YOLO + CRNN) — detect and recognize damage numbers
- **Health bar detection** — track monster health
- **Gauge detection** (this module) — monitor sharpness gauge state for Helm Breaker timing