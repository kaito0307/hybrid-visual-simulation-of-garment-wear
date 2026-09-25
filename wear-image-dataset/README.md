# Wear Image Dataset

This folder contains images of worn fabrics captured with our custom abrasion device. It covers 12 fabric types at 10 wear levels each.

## Structure

There is one folder per fabric type, named in lowercase with underscores (e.g., `denim_7oz`, `cotton_corduroy`). Each folder holds 10 images, `1.jpg` (unworn) to `10.jpg` (most worn), one for each wear level.

```
wear-image-dataset/
├── denim_7oz/
│   ├── 1.jpg     # wear level 1 (unworn)
│   ├── 2.jpg
│   ├── ...
│   └── 10.jpg    # wear level 10 (most worn)
├── denim_11oz/
└── ...
```

## Fabric Types

| | Plain weave | Twill weave | Knit |
|---|---|---|---|
| **Polyester** | Taffeta | Polyester twill | Polyester knit |
| **Cotton** | Broadcloth | Denim 7oz. / Denim 11oz. / Chino | Jersey knit |
| **Wool** | Tropical | Gabardine | Wool knit |

The dataset also includes Cotton corduroy, which does not fit into the table above.

## Capture Conditions

- Abrasion: sandpaper of grit #120, constant load of 5 N, belt conveyor speed of 7.6 cm/s.
- Camera: SONY α7 III with a SONY SEL30M35 macro lens, top-down view, under standard fluorescent ceiling lighting.
- Resolution: 1000 × 1000 pixels, covering 5.0 cm × 5.0 cm of fabric surface.

## Wear Levels

The 10 wear levels are not evenly spaced in abrasion cycles. Appearance changes fastest early in abrasion, so the early levels are spaced more finely. Each fabric has a different abrasion resistance, so the cycle counts differ per fabric:

| Fabric | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Denim 7oz. | 0 | 100 | 200 | 320 | 440 | 680 | 860 | 1060 | 1450 | 2020 |
| Denim 11oz. | 0 | 60 | 130 | 240 | 410 | 680 | 1120 | 1450 | 2050 | 2870 |
| Taffeta | 0 | 20 | 170 | 220 | 310 | 430 | 1250 | 1450 | 1620 | 2290 |
| Polyester twill | 0 | 40 | 90 | 210 | 420 | 570 | 910 | 1020 | 1280 | 1420 |
| Polyester knit | 0 | 490 | 810 | 930 | 1010 | 1100 | 1280 | 1460 | 1610 | 1810 |
| Broadcloth | 0 | 20 | 40 | 80 | 160 | 320 | 490 | 820 | 1210 | 1650 |
| Chino | 0 | 20 | 80 | 210 | 320 | 430 | 620 | 810 | 1210 | 1520 |
| Jersey knit | 0 | 160 | 220 | 350 | 530 | 620 | 870 | 1000 | 1130 | 1670 |
| Tropical | 0 | 50 | 90 | 160 | 470 | 730 | 920 | 1310 | 1630 | 1730 |
| Gabardine | 0 | 90 | 130 | 180 | 240 | 310 | 520 | 660 | 1060 | 1800 |
| Wool knit | 0 | 100 | 120 | 190 | 500 | 650 | 1080 | 1190 | 1350 | 2450 |
| Cotton corduroy | 0 | 100 | 340 | 790 | 1870 | 3020 | 4040 | 5650 | 7410 | 10050 |
