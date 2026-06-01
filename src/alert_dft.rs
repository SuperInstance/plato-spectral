/// DFT result for a single frequency bin.
#[derive(Debug, Clone)]
pub struct FrequencyBin {
    pub frequency: f64,
    pub amplitude: f64,
    pub phase: f64,
}

/// Result of alert DFT analysis.
#[derive(Debug, Clone)]
pub struct AlertDftResult {
    pub bins: Vec<FrequencyBin>,
    pub total_energy: f64,
}

/// Compute the Discrete Fourier Transform manually (no external deps).
///
/// Given a time series of alert counts, returns frequency bins with amplitude and phase.
/// Uses the standard DFT formula:
///   X[k] = Σ_{n=0}^{N-1} x[n] * exp(-2πi * k * n / N)
pub fn compute_dft(samples: &[f64]) -> Vec<FrequencyBin> {
    let n = samples.len();
    if n == 0 {
        return Vec::new();
    }

    let mut bins = Vec::with_capacity(n / 2 + 1);

    for k in 0..=n / 2 {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        let two_pi_k_over_n = 2.0 * std::f64::consts::PI * k as f64 / n as f64;

        for (idx, &sample) in samples.iter().enumerate() {
            let angle = two_pi_k_over_n * idx as f64;
            re += sample * angle.cos();
            im -= sample * angle.sin();
        }

        let amplitude = re.hypot(im);
        let phase = im.atan2(re);

        bins.push(FrequencyBin {
            frequency: k as f64 / n as f64,
            amplitude,
            phase,
        });
    }

    bins
}

/// Identify dominant frequencies in the alert time series.
///
/// Returns frequencies sorted by amplitude (descending), filtered to those above
/// `threshold_ratio * max_amplitude`.
pub fn identify_dominant_frequencies(samples: &[f64], threshold_ratio: f64) -> Vec<(f64, f64)> {
    let bins = compute_dft(samples);
    if bins.is_empty() {
        return Vec::new();
    }

    let max_amp = bins.iter().map(|b| b.amplitude).fold(0.0_f64, f64::max);
    let threshold = max_amp * threshold_ratio;

    let mut dominant: Vec<(f64, f64)> = bins
        .iter()
        .filter(|b| b.amplitude >= threshold)
        .map(|b| (b.frequency, b.amplitude))
        .collect();

    dominant.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    dominant
}

/// Compute the total spectral energy of the alert time series.
///
/// Spectral energy = Σ |X[k]|² / N² (Parseval's theorem normalized).
/// High spectral energy indicates regular patterns worth investigating.
pub fn alert_spectral_energy(samples: &[f64]) -> f64 {
    let bins = compute_dft(samples);
    let n = samples.len() as f64;
    if n == 0.0 {
        return 0.0;
    }

    // Parseval's: total energy = Σ|X[k]|² / N
    bins.iter().map(|b| b.amplitude * b.amplitude / n).sum()
}
