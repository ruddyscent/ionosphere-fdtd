# FDTD on a Planet, Part 8: Reproducing Simpson–Heikes–Taflove 2006

The 2006 paper by Simpson, Heikes, and Taflove extends global geodesic FDTD
from propagation to remote sensing. Its first two comparison figures revisit
the global ELF experiment discussed in [Part 7](07-reproducing-simpson-taflove-2004.md).
Its third asks a more ambitious question: can a 20 Hz transmitter in Wisconsin
detect the electromagnetic effect of a major subsurface oil deposit in
Alaska?[^paper-2006]

This reproduction separates those claims figure by figure. It recovers timing
and qualitative waveform structure, and one tangential perturbation statistic
passes. It does not reproduce all relative amplitudes, attenuation curves, or
the published magnetic-component scaling. The complete Figures 5–7 result is
therefore a FAIL.[^verification-2006]

## Three figures, two experiments

Figures 5 and 6 repeat the global propagation logic:

- Figure 5 compares near and far time-domain receiver fields along two paths;
- Figure 6 derives frequency-dependent attenuation from those traces.

Figure 7 is a differential radar experiment. A tangential 20 Hz source near
Clam Lake drives two otherwise matched simulations. One contains the reference
Earth model; the other adds a low-conductivity oil body below Alaska. The
observable is the difference between their magnetic fields at a receiver.

This paired design is powerful because common source and propagation effects
can cancel. It is also numerically demanding: subtracting two large fields to
isolate a small perturbation magnifies sensitivity to grid placement,
interpolation, material support, phase error, and any mismatch between the two
runs.

## Production geometry and material support

The production configuration uses a subdivision-7 geodesic mesh with 163,842
surface cells, 40 nominal radial cells at $5\ \mathrm{km}$ spacing, PyTorch
CUDA, and `float64`. The oil body covers $4800\ \mathrm{km}^2$, spans a
$1250\ \mathrm{m}$ vertical interval centered near $1200\ \mathrm{m}$ depth,
and reduces conductivity by a factor of 0.1.

A thin anomaly is not faithfully represented by asking only whether an
electric-field point lies inside it. The reproduction assigns horizontal
support conservatively to both radial-electric dual cells and
tangential-electric edge diamonds. Radial material fractions represent the
overlap between the anomaly and each staggered electric control volume.

The transmitter–receiver separation is global, while the target is local and
subsurface. Resolving both in one grid is a much harder requirement than
resolving the 20 Hz wavelength alone.

Even with conservative support, the reconstruction is not identical to the
authors' model. The paper does not publish the exact Mesquite optimization
parameters or the complete three-dimensional Hermance-derived conductivity
data. Those missing inputs are treated as uncertainty, not tuned until the
curves agree.

## Figures 5 and 6: global propagation

![Published and reproduced Simpson–Heikes–Taflove 2006 Figure 5 waveforms](../verification/images/simpson-taflove-2006-fig-5-comparison.png)

*The reproduced result preserves the pulse morphology and receiver ordering,
but its relative path amplitudes differ from the publication.*

Figure 5 passes morphology and arrival ordering. Its far-receiver normalized
peaks are 0.31141 and 0.35571, while path-shape RMS differences are 37.41% and
18.47%. Those amplitude and path-similarity values fail the declared
criterion.

![Published and reproduced Simpson–Heikes–Taflove 2006 Figure 6 attenuation curves](../verification/images/simpson-taflove-2006-fig-6-comparison.png)

*Figure 6 derives attenuation from the receiver pairs. Agreement in arrival
time does not guarantee agreement in spectral ratios.*

The attenuation errors are:

| Path | Mean absolute error | Maximum absolute error | Verdict |
|---|---:|---:|---|
| East | $0.921\ \mathrm{dB/Mm}$ | $3.020\ \mathrm{dB/Mm}$ | **FAIL** |
| West | $0.284\ \mathrm{dB/Mm}$ | $2.125\ \mathrm{dB/Mm}$ | **FAIL** |

These results are consistent with the 2004 reproduction: the solver captures
the broad propagation response, but a strict frequency-by-frequency match is
more sensitive to spatial dispersion and the reconstructed Earth model.

## Figure 7: the differential oil-body signal

For magnetic component $q$, the normalized perturbation is

$$
\Delta H_q(t)=20\log_{10}
\left(
\frac{|H_q^{\mathrm{oil}}(t)-H_q^{\mathrm{ref}}(t)|}
{\max_t |H_q^{\mathrm{ref}}(t)|}
\right).
$$

The denominator uses the maximum reference-field magnitude for that component.
The numerator is a pointwise difference between matched oil and reference
runs. This definition must be preserved exactly: normalizing each run
independently or comparing unmatched phases would change the physical
question.

![Published and reproduced Simpson–Heikes–Taflove 2006 Figure 7 magnetic perturbations](../verification/images/simpson-taflove-2006-fig-7-comparison.png)

*The tangential perturbation reaches the expected weak-signal scale for part of
the trace, but the duration below threshold and radial-component scale do not
match the published result.*

The reproduced tangential median is $-43.253\ \mathrm{dB}$, which passes its
level criterion. However, 92.469% of samples lie below $-25\ \mathrm{dB}$, so
the perturbation does not remain above the required scale for enough of the
window. The radial perturbation median is $+126.000\ \mathrm{dB}$, a decisive
component-scaling mismatch.

| Figure 7 criterion | Reproduced result | Verdict |
|---|---:|---|
| Tangential perturbation median | $-43.253\ \mathrm{dB}$ | **PASS** |
| Fraction below $-25\ \mathrm{dB}$ | 92.469% | **FAIL** |
| Radial perturbation median | $+126.000\ \mathrm{dB}$ | **FAIL** |

## Why a differential experiment is unforgiving

Suppose the oil and reference fields each contain a small phase error. Their
absolute waveforms may still look plausible. When subtracted, however, that
phase error can dominate the true material perturbation. The same applies to a
target whose support changes abruptly when the mesh is refined.

The implementation addresses several avoidable sources of inconsistency:

- both runs share the same mesh, time step, source, and receiver projector;
- anomaly overlap is assigned conservatively rather than by a single nearest
  point;
- tangential source azimuth is projected onto oriented geodesic edges;
- observations remain backend-native during stepping and are compared after a
  matched run;
- double precision removes arithmetic precision as the primary explanation.

These controls make the failure interpretable. They do not substitute for the
unpublished mesh optimization and conductivity volume.

## The final verdict

| Criterion | Verdict |
|---|---|
| Figure 5 morphology and arrival order | **PASS** |
| Figure 5 relative amplitudes and path similarity | **FAIL** |
| Figure 6 east attenuation | **FAIL** |
| Figure 6 west attenuation | **FAIL** |
| Figure 7 tangential median | **PASS** |
| Figure 7 threshold duration | **FAIL** |
| Figure 7 radial scale | **FAIL** |
| Complete Figures 5–7 reproduction | **FAIL** |

The result does not support a claim that this implementation has reproduced
the proposed oil-radar response. It does support narrower conclusions: global
timing and pulse morphology are present, the target is represented by a
conservative staggered material projector, and the remaining quantitative
differences are recorded without tuning unknown inputs to force a pass.

That is the useful role of reproduction work. It turns a visually appealing
simulation into a list of claims that can be accepted, rejected, or revisited
when better source data become available.

## Reproducing the workflow

The 2004 setup supplies the shared global propagation inputs; the 2006 command
adds the tangential transmitter and oil/reference pair:

```bash
python -m verification.simpson_taflove_2004 --help
python -m verification.simpson_taflove_2006 --help
```

Production archives retain configuration, checksums, and run signatures.
Published panels are included only for technical comparison.[^verification-2006]

## References

[^paper-2006]: J. J. Simpson, R. P. Heikes, and A. Taflove, “FDTD Modeling of a Novel ELF Radar for Major Oil Deposits Using a Three-Dimensional Geodesic Grid of the Earth-Ionosphere Waveguide,” *IEEE Transactions on Antennas and Propagation*, 54(6), 1734–1741, 2006, [doi:10.1109/TAP.2006.875504](https://doi.org/10.1109/TAP.2006.875504).

[^verification-2006]: Ionosphere FDTD project, “[Simpson–Heikes–Taflove 2006 Reproduction Verification](../verification/simpson-taflove-2006.md),” accessed 2026-08-14.
