#!/usr/bin/env python
"""Recompute the effective-receptive-field radii the manuscript reports.

The two archived arrays hold raw gradient fields, not radii. A summary radius
read off a raw field is NOT the number the paper prints, and the difference is
not small: every field sits on a near-uniform floor, and for two of the four
models that floor is what an uncorrected r75 measures. Subtracting it moves the
four r75 values from 41 / 191 / 151 / 165 pixels to 38 / 140 / 39 / 40. This
script is shipped so the archive reproduces the published numbers rather than
the uncorrected ones, and it asserts them, so a later change to either side is
caught instead of being discovered by a reader.

    python erf_radii.py            # print the table and check it

Needs numpy, which reading an .npz requires in any case; nothing else.

Method (Luo et al. 2016, gradient definition). For each model, the gradient of
the crack logit at a probe pixel with respect to the input, magnitude summed
over input channels and averaged over the same 50 DeepCrack test images.
`benchmarks/erf_arrays.npz` probes the geometric centre of the image, which is
the protocol the figure and the text report; it also carries three TANGO
ablation arms that the text reports as numbers rather than panels.
`benchmarks/erf_arrays_crackprobe.npz` repeats the four panel models with the
probe at the foreground pixel nearest the centre, each gradient map shifted back
to the centre before averaging, as a check that probing off the crack does not
drive the summary. Both files record the probe statistics in their `meta`.

The floor is estimated per field as the median of its radial profile between 220
and 250 pixels: outside every corrected r75, and inside the largest centred
circle that a 512 x 512 field samples completely. Widening the band to
[180, 250] or [200, 256] gives r75 = 37 / 127 / 36 / 41 and 38 / 135 / 37 / 41,
so the conclusion does not depend on the choice.

r50 and r75 are descriptive summaries -- the radii of the centred circles
containing 50 and 75 percent of the remaining gradient mass -- not the two-sigma
Gaussian fit of the original method. Quantiles above 75 percent are not reported
because the residual floor dominates them.
"""
import json
import sys

import numpy as np

FLOOR_BAND = (220, 250)

# What the manuscript prints, per file. Rounded to whole pixels, as there.
PUBLISHED = {
    'erf_arrays.npz': {
        'ConvNeXt-XLarge': (18, 38),
        'TANGO': (87, 140),
        'DTrC-Net': (18, 39),
        'MixerCSeg faithful': (21, 40),
        'TANGO reference (seed 42)': (87, 141),
        'TANGO a2_noprop (seed 42)': (87, 140),
        'TANGO a5_uniform (seed 42)': (89, 143),
    },
    'erf_arrays_crackprobe.npz': {
        'ConvNeXt-XLarge': (18, 38),
        'TANGO': (94, 143),
        'DTrC-Net': (23, 39),
        'MixerCSeg faithful': (15, 30),
    },
}
# The uncorrected r75 values, quoted in the text to show the size of the
# correction. Panel models only, in panel order.
PUBLISHED_RAW_R75 = (41, 191, 151, 165)
PANEL_MODELS = ('ConvNeXt-XLarge', 'TANGO', 'DTrC-Net', 'MixerCSeg faithful')


def radial_profile(field):
    """Mean of the field in each integer-radius bin, measured from the centre."""
    yy, xx = np.mgrid[0:field.shape[0], 0:field.shape[1]]
    rings = np.hypot(yy - field.shape[0] // 2,
                     xx - field.shape[1] // 2).astype(int)
    total = np.bincount(rings.ravel(), field.ravel())
    count = np.bincount(rings.ravel())
    return total / np.maximum(count, 1)


def floor_level(field, band=FLOOR_BAND):
    return float(np.median(radial_profile(field)[band[0]:band[1] + 1]))


def mass_radius(field, frac, floor=None):
    """Radius of the centred circle holding `frac` of the gradient mass.

    `floor=0` reproduces the uncorrected value; the default subtracts the floor,
    which is what the manuscript reports.
    """
    if floor is None:
        floor = floor_level(field)
    e = np.clip(field - floor, 0, None) if floor else field
    yy, xx = np.mgrid[0:e.shape[0], 0:e.shape[1]]
    r = np.hypot(yy - e.shape[0] // 2, xx - e.shape[1] // 2)
    order = np.argsort(r.ravel())
    mass = np.cumsum(e.ravel()[order])
    k = np.searchsorted(mass, frac * mass[-1])
    return float(r.ravel()[order][min(k, order.size - 1)])


def main():
    bad = []
    for name, published in PUBLISHED.items():
        path = 'benchmarks/' + name
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['meta']))
            fields = {lab: z['erf%d' % i].astype(np.float64)
                      for i, lab in enumerate(meta['models'])}
        stats = meta['probe_stats']
        print('%s  (probe: %s; %d of %d images have a crack at the centre, '
              'median distance to the nearest %.1f px)'
              % (name, meta['probe'], stats['centre_pixel_is_crack'],
                 stats['images'],
                 stats['distance_to_nearest_crack_px']['median']))
        for label, field in fields.items():
            r50, r75 = (mass_radius(field, f) for f in (0.50, 0.75))
            raw75 = mass_radius(field, 0.75, 0)
            got, want = (round(r50), round(r75)), published[label]
            flag = '' if got == want else '   <-- expected %d / %d' % want
            print('  %-28s r50 %3d  r75 %3d   (uncorrected r75 %3d)%s'
                  % (label, got[0], got[1], round(raw75), flag))
            if got != want:
                bad.append('%s: %s gives %d/%d, paper reports %d/%d'
                           % (name, label, got[0], got[1], want[0], want[1]))
        if name == 'erf_arrays.npz':
            raw = tuple(round(mass_radius(fields[m], 0.75, 0))
                        for m in PANEL_MODELS)
            if raw != PUBLISHED_RAW_R75:
                bad.append('uncorrected r75 %s, paper reports %s'
                           % (raw, PUBLISHED_RAW_R75))
            crop = min(512, 2 * int(np.ceil(
                max(mass_radius(fields[m], 0.75) for m in PANEL_MODELS) * 1.15)))
            print('  figure crop %d px (fitted to the data, not a constant)'
                  % (crop - crop % 2))
        print()

    if bad:
        print('MISMATCH against the published values:')
        for line in bad:
            print('  ' + line)
        return 1
    print('All radii match the values reported in the manuscript.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
