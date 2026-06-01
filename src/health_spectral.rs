/// Decomposed health metric time series.
#[derive(Debug, Clone)]
pub struct HealthDecomposition {
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub residual: Vec<f64>,
    pub spectral_energy_trend: f64,
    pub spectral_energy_residual: f64,
}

/// Spectral decomposition of health metrics using moving-average trend extraction
/// and DFT-based seasonal detection.
///
/// Returns {trend, seasonal, residual} components.
/// Degradation is detected when residual spectral energy grows over time.
pub fn health_spectral_decomposition(time_series: &[f64]) -> HealthDecomposition {
    let n = time_series.len();
    if n < 3 {
        return HealthDecomposition {
            trend: time_series.to_vec(),
            seasonal: vec![0.0; n],
            residual: vec![0.0; n],
            spectral_energy_trend: 0.0,
            spectral_energy_residual: 0.0,
        };
    }

    // Step 1: Extract trend using centered moving average (window = sqrt(n) rounded to odd)
    let window = {
        let w = (n as f64).sqrt().floor() as usize | 1; // ensure odd
        w.max(3).min(n)
    };
    let half_w = window / 2;
    let trend: Vec<f64> = (0..n)
        .map(|i| {
            let lo = i.saturating_sub(half_w);
            let hi = (i + half_w + 1).min(n);
            let count = hi - lo;
            time_series[lo..hi].iter().sum::<f64>() / count as f64
        })
        .collect();

    // Step 2: Detrend
    let detrended: Vec<f64> = time_series.iter().zip(&trend).map(|(&x, &t)| x - t).collect();

    // Step 3: Extract seasonal component via DFT
    let bins = crate::alert_dft::compute_dft(&detrended);

    // Find the dominant non-DC frequency
    let dominant = bins
        .iter()
        .skip(1) // skip DC
        .max_by(|a, b| a.amplitude.partial_cmp(&b.amplitude).unwrap_or(std::cmp::Ordering::Equal));

    let seasonal: Vec<f64> = if let Some(dom) = dominant {
        let freq = dom.frequency;
        let amp = dom.amplitude / n as f64;
        let phase = dom.phase;
        (0..n)
            .map(|i| {
                amp * (2.0 * std::f64::consts::PI * freq * i as f64 + phase).cos()
            })
            .collect()
    } else {
        vec![0.0; n]
    };

    // Step 4: Residual = original - trend - seasonal
    let residual: Vec<f64> = time_series
        .iter()
        .zip(&trend)
        .zip(&seasonal)
        .map(|((&x, &t), &s)| x - t - s)
        .collect();

    // Compute spectral energies
    let spectral_energy_trend = crate::alert_dft::alert_spectral_energy(&trend);
    let spectral_energy_residual = crate::alert_dft::alert_spectral_energy(&residual);

    HealthDecomposition {
        trend,
        seasonal,
        residual,
        spectral_energy_trend,
        spectral_energy_residual,
    }
}
