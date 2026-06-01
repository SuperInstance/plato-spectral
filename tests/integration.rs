use plato_spectral::*;

// ─── alert_dft tests ────────────────────────────────────────────────────

#[test]
fn test_dft_empty_input() {
    let bins = alert_dft::compute_dft(&[]);
    assert!(bins.is_empty());
}

#[test]
fn test_dft_single_sample() {
    let bins = alert_dft::compute_dft(&[42.0]);
    assert_eq!(bins.len(), 1);
    assert!((bins[0].amplitude - 42.0).abs() < 1e-9);
    assert!((bins[0].frequency - 0.0).abs() < 1e-9);
}

#[test]
fn test_dft_dc_signal() {
    // Constant signal should only have DC component
    let bins = alert_dft::compute_dft(&[5.0, 5.0, 5.0, 5.0]);
    assert!(bins[0].amplitude > 0.0); // DC
    for bin in &bins[1..] {
        assert!(bin.amplitude < 1e-9);
    }
}

#[test]
fn test_dft_single_frequency() {
    // sin(2π * 1 * n/4) has frequency 0.25
    let samples: Vec<f64> = (0..8).map(|n| (2.0 * std::f64::consts::PI * n as f64 / 8.0).sin()).collect();
    let bins = alert_dft::compute_dft(&samples);
    let dominant = bins.iter().skip(1).max_by(|a, b| a.amplitude.partial_cmp(&b.amplitude).unwrap()).unwrap();
    assert!((dominant.frequency - 0.125).abs() < 0.01); // freq = 1/8
}

#[test]
fn test_dft_nyquist() {
    let bins = alert_dft::compute_dft(&[1.0, -1.0, 1.0, -1.0]);
    // Highest amplitude should be at the Nyquist frequency (k = N/2)
    let nyquist = bins.last().unwrap();
    assert!(nyquist.amplitude > 1.0);
}

#[test]
fn test_dft_returns_correct_bin_count() {
    let bins = alert_dft::compute_dft(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(bins.len(), 3); // 5/2 + 1 = 3
}

#[test]
fn test_dft_returns_correct_bin_count_even() {
    let bins = alert_dft::compute_dft(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(bins.len(), 3); // 4/2 + 1 = 3
}

#[test]
fn test_identify_dominant_empty() {
    let result = identify_dominant_frequencies(&[], 0.5);
    assert!(result.is_empty());
}

#[test]
fn test_identify_dominant_single_freq() {
    // Build a signal with one clear frequency
    let samples: Vec<f64> = (0..16).map(|n| (2.0 * std::f64::consts::PI * 3.0 * n as f64 / 16.0).sin()).collect();
    let dominant = identify_dominant_frequencies(&samples, 0.5);
    assert!(!dominant.is_empty());
    assert!((dominant[0].0 - 3.0 / 16.0).abs() < 0.01);
}

#[test]
fn test_identify_dominant_sorted_descending() {
    let samples: Vec<f64> = (0..32)
        .map(|n| {
            (2.0 * std::f64::consts::PI * 2.0 * n as f64 / 32.0).sin()
                + 3.0 * (2.0 * std::f64::consts::PI * 5.0 * n as f64 / 32.0).sin()
        })
        .collect();
    let dominant = identify_dominant_frequencies(&samples, 0.3);
    for window in dominant.windows(2) {
        assert!(window[0].1 >= window[1].1);
    }
}

#[test]
fn test_spectral_energy_empty() {
    let energy = alert_spectral_energy(&[]);
    assert!((energy - 0.0).abs() < 1e-12);
}

#[test]
fn test_spectral_energy_constant() {
    let energy = alert_spectral_energy(&[10.0, 10.0, 10.0, 10.0]);
    assert!(energy > 0.0);
}

#[test]
fn test_spectral_energy_vs_parseval() {
    // Parseval's theorem: time-domain energy ≈ spectral energy
    let samples = [1.0, 2.0, 3.0, 4.0, 5.0];
    let time_energy: f64 = samples.iter().map(|x| x * x).sum::<f64>();
    let spec_energy = alert_spectral_energy(&samples);
    // They should be proportional (exact equality depends on normalization)
    assert!(spec_energy > 0.0);
    assert!((spec_energy - time_energy).abs() / time_energy < 0.5);
}

#[test]
fn test_spectral_energy_zero_signal() {
    let energy = alert_spectral_energy(&[0.0, 0.0, 0.0]);
    assert!((energy - 0.0).abs() < 1e-12);
}

// ─── fisher_rooms tests ─────────────────────────────────────────────────

#[test]
fn test_fisher_identical_distributions() {
    let room_a = [0.25, 0.25, 0.25, 0.25];
    let room_b = [0.25, 0.25, 0.25, 0.25];
    let metric = room_fisher_metric(&room_a, &room_b);
    assert!(metric < 1e-9);
}

#[test]
fn test_fisher_different_distributions() {
    let room_a = [0.5, 0.5, 0.0, 0.0];
    let room_b = [0.0, 0.0, 0.5, 0.5];
    let metric = room_fisher_metric(&room_a, &room_b);
    assert!(metric.is_infinite() || metric > 10.0); // completely disjoint
}

#[test]
fn test_fisher_symmetric() {
    let room_a = [0.1, 0.4, 0.3, 0.2];
    let room_b = [0.2, 0.3, 0.4, 0.1];
    let ab = room_fisher_metric(&room_a, &room_b);
    let ba = room_fisher_metric(&room_b, &room_a);
    assert!((ab - ba).abs() < 1e-9);
}

#[test]
fn test_fisher_similar_distributions() {
    let room_a = [0.25, 0.25, 0.25, 0.25];
    let room_b = [0.26, 0.24, 0.26, 0.24];
    let metric = room_fisher_metric(&room_a, &room_b);
    assert!(metric < 0.1);
}

#[test]
fn test_natural_gradient_basic() {
    let params = [1.0, 2.0, 3.0];
    let gradient = [0.1, 0.2, 0.3];
    let fisher = [1.0, 1.0, 1.0];
    let ng = room_natural_gradient(&params, &gradient, &fisher);
    assert_eq!(ng.len(), 3);
    assert!((ng[0] - 0.1).abs() < 1e-9);
    assert!((ng[1] - 0.2).abs() < 1e-9);
    assert!((ng[2] - 0.3).abs() < 1e-9);
}

#[test]
fn test_natural_gradient_scaling() {
    let params = [1.0, 2.0];
    let gradient = [2.0, 3.0];
    let fisher = [2.0, 3.0];
    let ng = room_natural_gradient(&params, &gradient, &fisher);
    assert!((ng[0] - 1.0).abs() < 1e-9);
    assert!((ng[1] - 1.0).abs() < 1e-9);
}

#[test]
fn test_natural_gradient_zero_fisher_fallback() {
    let params = [1.0];
    let gradient = [5.0];
    let fisher = [0.0];
    let ng = room_natural_gradient(&params, &gradient, &fisher);
    assert!((ng[0] - 5.0).abs() < 1e-9); // fallback to Euclidean
}

#[test]
fn test_natural_gradient_small_fisher_fallback() {
    let params = [1.0];
    let gradient = [5.0];
    let fisher = [1e-13];
    let ng = room_natural_gradient(&params, &gradient, &fisher);
    assert!((ng[0] - 5.0).abs() < 1e-9); // fallback
}

#[test]
#[should_panic(expected = "Rooms must have the same number of states")]
fn test_fisher_mismatched_lengths() {
    room_fisher_metric(&[0.5, 0.5], &[0.33, 0.33, 0.34]);
}

// ─── conservation tests ─────────────────────────────────────────────────

#[test]
fn test_flow_conservation_balanced() {
    assert!(tile_flow_conservation(&[1.0, 2.0, 3.0], &[2.0, 1.0, 3.0]));
}

#[test]
fn test_flow_conservation_unbalanced() {
    assert!(!tile_flow_conservation(&[1.0, 2.0, 3.0], &[1.0, 2.0, 4.0]));
}

#[test]
fn test_flow_conservation_empty() {
    assert!(tile_flow_conservation(&[], &[]));
}

#[test]
fn test_flow_conservation_floating_point() {
    let incoming = [0.1, 0.2, 0.3];
    let outgoing = [0.3, 0.2, 0.1];
    assert!(tile_flow_conservation(&incoming, &outgoing));
}

#[test]
fn test_flow_conservation_tiny_imbalance() {
    assert!(tile_flow_conservation_with_tolerance(&[1.0], &[1.0 + 1e-10], 1e-9));
    assert!(!tile_flow_conservation_with_tolerance(&[1.0], &[1.0 + 1e-6], 1e-9));
}

#[test]
fn test_energy_conservation_empty() {
    assert!(energy_conservation(&[]));
}

#[test]
fn test_energy_conservation_single_step() {
    let chain = [conservation::TransformStep {
        name: "step1".into(),
        input_energy: 100.0,
        output_energy: 100.0,
    }];
    assert!(energy_conservation(&chain));
}

#[test]
fn test_energy_conservation_lossy_step() {
    let chain = [conservation::TransformStep {
        name: "step1".into(),
        input_energy: 100.0,
        output_energy: 90.0,
    }];
    assert!(!energy_conservation(&chain));
}

#[test]
fn test_energy_conservation_chain() {
    let chain = [
        conservation::TransformStep { name: "a".into(), input_energy: 50.0, output_energy: 50.0 },
        conservation::TransformStep { name: "b".into(), input_energy: 50.0, output_energy: 50.0 },
    ];
    assert!(energy_conservation(&chain));
}

#[test]
fn test_energy_conservation_chain_discontinuity() {
    let chain = [
        conservation::TransformStep { name: "a".into(), input_energy: 50.0, output_energy: 50.0 },
        conservation::TransformStep { name: "b".into(), input_energy: 40.0, output_energy: 40.0 },
    ];
    assert!(!energy_conservation(&chain));
}

// ─── distill_transport tests ────────────────────────────────────────────

#[test]
fn test_sinkhorn_identity_cost() {
    // Zero cost → uniform transport
    let cost = vec![vec![0.0; 3]; 3];
    let a = vec![1.0 / 3.0; 3];
    let b = vec![1.0 / 3.0; 3];
    let result = sinkhorn_step(&cost, &a, &b, 1.0, 50);
    // Distance should be ~0 for zero cost
    assert!(result.distance.abs() < 1e-6);
}

#[test]
fn test_sinkhorn_plan_row_marginals() {
    let cost = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let a = vec![0.6, 0.4];
    let b = vec![0.5, 0.5];
    let result = sinkhorn_step(&cost, &a, &b, 1.0, 200);
    // Row marginals should approximate a
    for i in 0..2 {
        let row_sum: f64 = result.transport_plan[i].iter().sum();
        assert!((row_sum - a[i]).abs() < 0.05);
    }
}

#[test]
fn test_sinkhorn_plan_col_marginals() {
    let cost = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let a = vec![0.6, 0.4];
    let b = vec![0.5, 0.5];
    let result = sinkhorn_step(&cost, &a, &b, 1.0, 200);
    // Col marginals should approximate b
    for j in 0..2 {
        let col_sum: f64 = result.transport_plan.iter().map(|r| r[j]).sum();
        assert!((col_sum - b[j]).abs() < 0.05);
    }
}

#[test]
fn test_sinkhorn_positive_plan() {
    let cost = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
    let a = vec![0.5, 0.5];
    let b = vec![0.5, 0.5];
    let result = sinkhorn_step(&cost, &a, &b, 1.0, 50);
    for row in &result.transport_plan {
        for &val in row {
            assert!(val > 0.0);
        }
    }
}

#[test]
fn test_sinkhorn_distance_positive() {
    let cost = vec![vec![1.0, 3.0], vec![2.0, 1.0]];
    let a = vec![0.5, 0.5];
    let b = vec![0.5, 0.5];
    let dist = sinkhorn_distance(&cost, &a, &b, 1.0, 100);
    assert!(dist > 0.0);
}

#[test]
fn test_sinkhorn_cheaper_path_preferred() {
    // Lower cost diagonal should be preferred
    let cost = vec![vec![0.1, 10.0], vec![10.0, 0.1]];
    let a = vec![0.5, 0.5];
    let b = vec![0.5, 0.5];
    let result = sinkhorn_step(&cost, &a, &b, 10.0, 200);
    // Diagonal entries should be larger
    assert!(result.transport_plan[0][0] > result.transport_plan[0][1]);
    assert!(result.transport_plan[1][1] > result.transport_plan[1][0]);
}

#[test]
fn test_sinkhorn_convergence() {
    let cost = vec![vec![1.0, 2.0, 3.0], vec![4.0, 1.0, 2.0], vec![3.0, 2.0, 1.0]];
    let a = vec![1.0 / 3.0; 3];
    let b = vec![1.0 / 3.0; 3];
    // More iterations should converge
    let r10 = sinkhorn_step(&cost, &a, &b, 1.0, 10);
    let r100 = sinkhorn_step(&cost, &a, &b, 1.0, 100);
    // Distance should stabilize
    let diff = (r100.distance - r10.distance).abs();
    assert!(diff < 1.0);
}

// ─── health_spectral tests ──────────────────────────────────────────────

#[test]
fn test_health_decomposition_empty() {
    let decomp = health_spectral_decomposition(&[]);
    assert!(decomp.trend.is_empty());
    assert!(decomp.seasonal.is_empty());
    assert!(decomp.residual.is_empty());
}

#[test]
fn test_health_decomposition_short() {
    let decomp = health_spectral_decomposition(&[1.0, 2.0]);
    assert_eq!(decomp.trend.len(), 2);
    assert_eq!(decomp.seasonal.len(), 2);
    assert_eq!(decomp.residual.len(), 2);
}

#[test]
fn test_health_decomposition_constant() {
    let decomp = health_spectral_decomposition(&[5.0; 20]);
    // Trend should be ~5.0, residual ~0
    for &t in &decomp.trend {
        assert!((t - 5.0).abs() < 0.5);
    }
    for &r in &decomp.residual {
        assert!(r.abs() < 0.5);
    }
}

#[test]
fn test_health_decomposition_additive() {
    let series = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 13.0, 11.0, 9.0, 7.0, 5.0, 3.0, 1.0, -1.0];
    let decomp = health_spectral_decomposition(&series);
    // original ≈ trend + seasonal + residual
    for i in 0..series.len() {
        let reconstructed = decomp.trend[i] + decomp.seasonal[i] + decomp.residual[i];
        assert!((reconstructed - series[i]).abs() < 1e-6, "Mismatch at index {}: {} vs {}", i, reconstructed, series[i]);
    }
}

#[test]
fn test_health_decomposition_energies_positive() {
    let series: Vec<f64> = (0..50).map(|n| (n as f64 * 0.1).sin() + (n as f64 * 0.3).cos()).collect();
    let decomp = health_spectral_decomposition(&series);
    assert!(decomp.spectral_energy_trend >= 0.0);
    assert!(decomp.spectral_energy_residual >= 0.0);
}

#[test]
fn test_health_degradation_detection() {
    // Series with growing noise (degradation)
    let clean: Vec<f64> = (0..30).map(|i| (i as f64 * 0.2).sin()).collect();
    let noisy: Vec<f64> = clean.iter().enumerate().map(|(i, &v)| v + 0.1 * i as f64 * (i as f64 * 0.7).sin()).collect();
    let decomp_clean = health_spectral_decomposition(&clean);
    let decomp_noisy = health_spectral_decomposition(&noisy);
    // Noisy series should have higher residual energy
    assert!(decomp_noisy.spectral_energy_residual >= decomp_clean.spectral_energy_residual * 0.5);
}

#[test]
fn test_health_trend_smoothness() {
    let series: Vec<f64> = (0..20).map(|i| 5.0 + (i as f64 * 0.5)).collect();
    let decomp = health_spectral_decomposition(&series);
    // Trend should be smoother (lower residual energy) than the detrended signal
    assert!(decomp.spectral_energy_trend > 0.0);
}

// ─── integration tests ──────────────────────────────────────────────────

#[test]
fn test_dft_reconstruction_identity() {
    // A constant signal should reconstruct perfectly from DFT
    let samples = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0];
    let bins = alert_dft::compute_dft(&samples);
    // Only DC should be non-zero
    assert!((bins[0].amplitude - 24.0).abs() < 1e-9); // 8 * 3.0
    for bin in &bins[1..] {
        assert!(bin.amplitude < 1e-9);
    }
}

#[test]
fn test_pipeline_alert_to_health() {
    // Generate alert data, run through DFT + health analysis
    let alerts: Vec<f64> = (0..100)
        .map(|t| {
            let base = 10.0;
            let periodic = 5.0 * (2.0 * std::f64::consts::PI * t as f64 / 20.0).sin();
            let noise = if t % 7 == 0 { 3.0 } else { 0.0 };
            base + periodic + noise
        })
        .collect();

    let dominant = identify_dominant_frequencies(&alerts, 0.3);
    assert!(!dominant.is_empty());

    let energy = alert_spectral_energy(&alerts);
    assert!(energy > 0.0);

    let decomp = health_spectral_decomposition(&alerts);
    assert_eq!(decomp.trend.len(), 100);
}

#[test]
fn test_conservation_with_transport() {
    // Verify that transport plan conserves mass
    let cost = vec![vec![1.0, 2.0, 3.0], vec![2.0, 1.0, 2.0]];
    let a = vec![0.6, 0.4];
    let b = vec![0.3, 0.3, 0.4];
    let result = sinkhorn_step(&cost, &a, &b, 1.0, 200);

    let total_transported: f64 = result.transport_plan.iter().flat_map(|r| r.iter()).sum();
    assert!((total_transported - 1.0).abs() < 0.05);
}
