# FDTD on a Planet, Part 6: Reproducing Simpson–Taflove 2004

[Part 5](05-verifying-the-solver-with-analytic-solutions.md) established that
the solver converges to known Maxwell solutions. The next test is less tidy:
can it reproduce a published global Earth–ionosphere simulation?

Simpson and Taflove's 2004 paper is an unusually demanding reference. It sends
an impulsive ELF signal around the complete Earth, records radial electric
fields at quarter- and half-circumference receivers, and derives attenuation
from pairs of those traces.[^paper-2004] A result can therefore have the right
arrival order and still fail in amplitude, spectral attenuation, or east–west
asymmetry.

This reproduction reaches that mixed outcome. The principal waveform
morphology and arrival order pass. The published relative path amplitudes and
strict pointwise attenuation do not.[^verification-2004]

## The reference experiment

The paper places a vertical Gaussian current on the equator and records four
radial-electric receiver traces. A and A′ lie approximately one quarter of the
way around the globe from the source along opposite directions. B and B′ lie
near the half-circumference distance. The primed and unprimed paths sample
different Earth material and relief even when their great-circle lengths are
similar.

Figure 7 tests the time domain. Its essential observable is not an absolute
field calibration but a structured pulse: a negative main excursion, a
positive overshoot, and a slowly decaying tail. Quarter-arc receivers should
respond before half-arc receivers.

Figure 8 tests the frequency domain. For two receivers separated by distance
$d$, the reported attenuation is

$$
\alpha_{\mathrm{dB/Mm}}(f)
=\frac{20}{d_{\mathrm{Mm}}}
\log_{10}\!\left|\frac{E_1(f)}{E_2(f)}\right|.
$$

Taking a spectral ratio cancels the unknown absolute source amplitude, but it
does not cancel directional numerical dispersion, material differences along
the paths, or choices in time-window truncation.

## Reconstructing the numerical model

The reproduction uses the paper's equatorial source/receiver geometry, a
$3\ \mu\mathrm{s}$ step, 40 radial cells, and the published exponential
ionosphere. The production surface grid is subdivision 8, giving 655,362 dual
cells. It runs 35,000 updates in compiled PyTorch CUDA `float64` so the complete
time extent is available and arithmetic precision is not an easy explanation
for disagreement.

The largest uncertainty is below and at the surface. The paper describes NOAA
NGDC relief but does not identify the exact data edition, preprocessing,
coastline convention, or full three-dimensional conductivity realization. The
reproduction uses ETOPO5 as a period-appropriate reconstruction. That is a
defensible input, not a claim to possess the authors' original grid.

This distinction matters. Analytic verification fixes every coefficient and
boundary. A paper reproduction must sometimes reconstruct inputs that were
never archived.

## Figure 7: timing passes, relative amplitudes fail

![Published and reproduced Simpson–Taflove 2004 Figure 7 receiver traces](images/simpson-taflove-2004-fig-7-comparison.png)

*Published traces are on the left and reproduced traces are on the right. The
comparison preserves the complete axes and matches the plotted data-frame
geometry rather than resizing plots by their outer whitespace.*

The reproduced records contain the same broad temporal structure as the
published figure:

- a negative main pulse;
- a positive overshoot;
- a persistent slow tail;
- A/A′ arrivals before B/B′ arrivals;
- nonidentical eastward and westward paths.

Those observations are meaningful. They show that the source launches, the
wave circulates through the cavity, and the receiver geometry produces the
expected causal ordering.

But the published east/west peak ordering and separation are not preserved.
That failure prevents a full Figure 7 pass. A waveform can be qualitatively
recognizable while its directional amplitudes remain quantitatively wrong.

## Figure 8: attenuation misses strict tolerances

![Published and reproduced Simpson–Taflove 2004 Figure 8 attenuation curves](images/simpson-taflove-2004-fig-8-comparison.png)

*The reproduced curves use the archived receiver traces and the declared
spectral procedure. No smoothing or rescaling is introduced to improve visual
agreement.*

The pointwise error over 50–500 Hz is:

| Path pair | Mean absolute error | Maximum absolute error | Verdict |
|---|---:|---:|---|
| A–B | $1.104\ \mathrm{dB/Mm}$ | $2.538\ \mathrm{dB/Mm}$ | **FAIL** |
| A′–B′ | $0.242\ \mathrm{dB/Mm}$ | $3.258\ \mathrm{dB/Mm}$ | **FAIL** |

The smaller mean error on A′–B′ does not rescue the comparison because its
largest local deviation is still substantial. The acceptance test is based on
the declared pointwise behavior, not a favorable average alone.

## What was ruled out

Several plausible software explanations do not account for the residual:

- NumPy and PyTorch double-precision fields agree under matched tests;
- moving production execution to CUDA `float64` does not repair attenuation;
- source staggering preserves the configured current moment and location;
- the analytic A2 and A4 studies show second-order spherical-mode convergence;
- changing the source amplitude cannot fix a receiver spectral ratio.

The remaining error is most sensitive to horizontal spatial dispersion and to
the incompletely specified three-dimensional conductivity and relief model.
That is not proof that either one is the unique cause. It is the boundary of
what the available evidence supports.

## The final verdict

| Criterion | Result |
|---|---|
| Figure 7 morphology | **PASS** |
| Quarter-arc before half-arc arrival | **PASS** |
| Published east/west relative amplitudes | **FAIL** |
| Figure 8 A–B attenuation | **FAIL** |
| Figure 8 A′–B′ attenuation | **FAIL** |
| Complete Figures 7–8 reproduction | **FAIL** |

Calling the complete reproduction a failure does not erase the successful
physics. It keeps two claims separate:

1. the implementation produces a physically recognizable global ELF response;
2. it does not reproduce every published quantitative observable within the
   declared tolerance.

That wording is more useful than either “the plots look similar, therefore it
works” or “one metric failed, therefore nothing was learned.”

## Reproducing the workflow

The paper-specific workflow remains outside the installable runtime package:

```bash
python -m verification.simpson_taflove_2004 --help
```

Each production archive records its complete configuration and checksums with
the receiver traces. Published panels are retained only for technical
comparison.[^verification-2004]

Part 7 follows the same discipline for the 2006 extension: two global
propagation figures and a proposed ELF radar response to a subsurface oil body.

## References

[^paper-2004]: J. J. Simpson and A. Taflove, “Three-Dimensional FDTD Modeling of Impulsive ELF Propagation About the Earth-Sphere,” *IEEE Transactions on Antennas and Propagation*, 52(2), 443–451, 2004, [doi:10.1109/TAP.2004.823953](https://doi.org/10.1109/TAP.2004.823953).

[^verification-2004]: Ionosphere FDTD project, “[Simpson–Taflove 2004 Reproduction Verification](../verification/simpson-taflove-2004.md),” accessed 2026-08-14.

