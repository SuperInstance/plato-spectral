# plato-spectral

> Spectral mathematics for PLATO — DFT alert analysis, Fisher metric rooms, conservation laws, and Sinkhorn optimal transport

## What This Does

plato-spectral applies mathematical tools from signal processing and information geometry to PLATO's monitoring pipeline. It answers questions like: "Do my alerts have a periodic pattern?", "Are two rooms configured similarly?", and "Is energy conserved across my processing chain?"

Instead of treating monitoring data as flat numbers, plato-spectral treats it as *signals* — with frequency content, energy budgets, and geometric structure.

## The Key Idea

Think of your alert stream as music. A raw list of alert counts is noise. But run it through a Fourier transform and suddenly you see the rhythm — maybe alerts spike every 4 hours (a cron job), or every Monday (a capacity planning issue). plato-spectral gives you that lens.

Beyond frequency analysis, it borrows from information geometry (Fisher metrics to measure how "far apart" two room configurations are in probability space) and optimal transport (Sinkhorn's algorithm to compute the cheapest way to move probability mass from one distribution to another — which turns out to be the mathematically correct way to compare distributions).

## Install

```bash
cargo add plato-spectral
```

## Quick Start

```rust
use plato_spectral::{compute_dft, identify_dominant_frequencies, alert_spectral_energy};

// Alert counts over 16 time windows
let alerts: Vec<f64> = (0..16)
    .map(|n| 10.0 + 5.0 * (2.0 * std::f64::consts::PI * 3.0 * n as f64 / 16.0).sin())
    .collect();

// Find which frequencies dominate the alert pattern
let dominant = identify_dominant_frequencies(&alerts, 0.5);
println!("Dominant frequency: {:.3}", dominant[0].0);

// Total spectral energy (Parseval's theorem)
let energy = alert_spectral_energy(&alerts);
println!("Spectral energy: {:.2}", energy);
```

## API Reference

### DFT Alert Analysis (`alert_dft`)

| Type/Function | Description |
|---|---|
| `FrequencyBin { frequency, amplitude, phase }` | One frequency component from the DFT |
| `AlertDftResult { bins, total_energy }` | Complete DFT analysis result |
| `compute_dft(samples: &[f64]) -> Vec<FrequencyBin>` | Compute the Discrete Fourier Transform. Returns N/2+1 bins from DC to Nyquist. |
| `identify_dominant_frequencies(samples, threshold_ratio) -> Vec<(f64, f64)>` | Returns (frequency, amplitude) pairs sorted descending by amplitude, filtered above threshold. |
| `alert_spectral_energy(samples) -> f64` | Total spectral energy via Parseval's theorem. High energy = strong periodic patterns. |

### Fisher Metric Rooms (`fisher_rooms`)

| Function | Description |
|---|---|
| `room_fisher_metric(room_a: &[f64], room_b: &[f64]) -> f64` | Symmetric KL divergence between two room probability distributions. 0 = identical, ∞ = disjoint. |
| `room_natural_gradient(params, gradient, fisher) -> Vec<f64>` | Natural gradient F⁻¹·∇. Falls back to Euclidean when Fisher info is near-zero. |

### Conservation Laws (`conservation`)

| Type/Function | Description |
|---|---|
| `TransformStep { name, input_energy, output_energy }` | A step in a processing chain with energy accounting |
| `tile_flow_conservation(incoming, outgoing) -> bool` | Checks Σin == Σout within tolerance (default 1e-9) |
| `energy_conservation(chain: &[TransformStep]) -> bool` | Verifies every step conserves energy AND chain continuity |

### Sinkhorn Optimal Transport (`distill_transport`)

| Type/Function | Description |
|---|---|
| `SinkhornResult { transport_plan, distance }` | The optimal transport plan (matrix) and Wasserstein distance |
| `sinkhorn_step(cost, marginal_a, marginal_b, lambda, iterations)` | Full Sinkhorn algorithm. `lambda` = entropic regularization. |
| `sinkhorn_distance(...) -> f64` | Returns just the transport distance, not the full plan. |

### Health Spectral Decomposition (`health_spectral`)

| Type/Function | Description |
|---|---|
| `HealthDecomposition { trend, seasonal, residual, spectral_energy_trend, spectral_energy_residual }` | Decomposed time series. Growing residual energy = degradation. |
| `health_spectral_decomposition(time_series: &[f64])` | Extracts trend, seasonal, and residual via moving average + DFT. |

## How It Works

**DFT**: Standard DFT X[k] = Σ x[n]·e^(-2πikn/N), no external dependencies.

**Fisher Metric**: Symmetric KL divergence D_sym = (KL(p‖q) + KL(q‖p))/2 between probability distributions.

**Sinkhorn**: Entropic regularized OT. Kernel K[i,j] = e^(-λC[i,j]), dual variables u,v updated iteratively until plan T = diag(u)·K·diag(v) satisfies marginals.

**Conservation**: Verifies processing chains don't create or destroy data.

## The Math

- **Parseval's Theorem**: Σ|X[k]|²/N = Σ|x[n]|²
- **KL Divergence**: D_KL(p‖q) = Σ p_i · ln(p_i/q_i)
- **Natural Gradient**: ∇̃ = F⁻¹·∇ where F is Fisher information matrix
- **Wasserstein Distance**: W(p,q) = min_T Σ T[i,j]·C[i,j] subject to marginals

## Testing

50 tests covering: DFT correctness (DC, Nyquist, known frequencies), Parseval's theorem, Fisher metric symmetry, conservation laws, Sinkhorn convergence/marginals, health decomposition additivity, and full pipeline integration.

## License

MIT
