# Study: the 0.119 Hz antisymmetric vertical mode

The ~0.12 Hz antisymmetric vertical mode behaves differently from all other vertical modes: its
damping lies above the quasi-steady line, and in the very gusty evening of storm Urd (26 Dec 2016)
its frequency drops by about 3 %. These scripts investigate why. They use the **two-event data set**
(September 2014 + storm Urd, 34 recordings), i.e. the step 5 results in `results/step5/`
(run `pipeline/step5_wind.py` with `DATA_SETS = ["2014_9", "StormUrd_LowFreq"]` and
`OUTPUT_FOLDER = RESULTS_FOLDER / "step5"`), and `results/step5d/` from step 5d on the same data.

| Script | Question | Answer |
|---|---|---|
| `inspect_0117.py` | What do spectra, mode shape and damping look like at low / mid / high wind? | Spectrum broadens in wind; deck x-signals are sensor tilt, not longitudinal motion |
| `band_ssi_antisym.py` | Can a dedicated 2-mode SSI (spatial filter + band-pass) separate it from the 0.105 Hz lateral mode? | Yes; damping 1-6 %, frequency 0.114-0.123 Hz |
| `inspect_0117_amplitude.py` | Is the frequency drop an amplitude effect? | No: at equal response amplitude the low-turbulence storm night gives 0.119-0.123 Hz |
| `simulation_ssi_bias.py` | Could gusty, non-stationary wind bias the SSI? | No: < 0.7 % in frequency and < 0.7 %-points in damping |
| `aeroelastic_model.py` | Does aerodynamic coupling with the lateral mode explain it? | No: < 0.1 %-points (validated against Oiseth et al. 2010, flutter 92.8/80.3 m/s) |

Conclusion: the behaviour is most likely real and aerodynamic, linked to strong gustiness, but it is
confounded with a single storm evening (one wind direction), so it cannot be generalised.
