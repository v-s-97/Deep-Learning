# IFF-AR: Conditional Autoregressive Audio Generation via Instantaneous Frequency Modeling

**Author:** Valerio Santini  
**Course:** Deep Learning & Applied AI (DLAI 2024/2025) — Sapienza University of Rome    

---

## Overview

**IFF-AR** (Instantaneous Frequency Flow–Autoregressive) is a generative model for **audio synthesis in the frequency domain**,  
designed to jointly predict **log-magnitude** and **instantaneous frequency (IF)** rather than reconstructing phase iteratively.  

Unlike models that rely on Griffin–Lim or separate phase estimation, IFF-AR treats the phase as a **learnable probabilistic variable**,  
modelled via a **conditional normalizing flow** (RealNVP-style) conditioned on both **context** and **spectral energy**.

The architecture combines:
- a **causal Temporal Convolutional Encoder (TCN)** for context representation,  
- a **lightweight magnitude decoder** (*MagHead*),  
- a **Conditional Normalizing Flow** (*IF-Flow*) for phase dynamics,  
- and a **Phase Reconstructor** to recover the waveform via cumulative integration of phase.

## Core pipeline
1. **Preprocessing** — Converts waveforms into STFT representations, extracting log-magnitude and instantaneous frequency (IF) per frequency bin.
2. **Encoding** — A causal dilated TCN processes context frames to capture temporal–spectral dependencies.  
3. **Magnitude prediction** — MagHead reconstructs normalized log-magnitude spectra.  
4. **Conditional flow** — IFConditionalFlow learns the distribution of phase differences given context and magnitude.  
5. **Reconstruction** — The phase is integrated cumulatively, and iSTFT converts complex spectra back to waveform audio.  

The file [`Demo.ipynb`](./Demo.ipynb) provides an **end-to-end demonstration**:
- Loads model weights and validation data  
- Visualizes waveform and spectrograms  
- Generates autoregressive predictions  
- Reconstructs the waveform and plays synthesized audio  
