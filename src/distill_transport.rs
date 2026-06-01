/// Result of the Sinkhorn optimal transport algorithm.
#[derive(Debug, Clone)]
pub struct SinkhornResult {
    pub transport_plan: Vec<Vec<f64>>,
    pub distance: f64,
}

/// Perform one full Sinkhorn iteration for optimal transport.
///
/// `cost_matrix` - C[i][j] = cost of moving from i to j
/// `marginal_a` - source distribution (row marginals)
/// `marginal_b` - target distribution (column marginals)
/// `lambda` - entropic regularization parameter (higher = more diffused plan)
/// `iterations` - number of Sinkhorn iterations
///
/// Returns the optimal transport plan and the transport distance (Wasserstein-1 approximation).
pub fn sinkhorn_step(
    cost_matrix: &[Vec<f64>],
    marginal_a: &[f64],
    marginal_b: &[f64],
    lambda: f64,
    iterations: usize,
) -> SinkhornResult {
    let n = marginal_a.len();
    let m = marginal_b.len();
    assert_eq!(cost_matrix.len(), n);
    for row in cost_matrix {
        assert_eq!(row.len(), m);
    }

    // Kernel matrix: K[i][j] = exp(-lambda * C[i][j])
    let kernel: Vec<Vec<f64>> = cost_matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(|&c| (-lambda * c).exp())
                .collect()
        })
        .collect();

    // Dual variables
    let mut u = vec![1.0 / n as f64; n];
    let mut v = vec![1.0 / m as f64; m];

    for _ in 0..iterations {
        // u = a ./ (K * v)
        for i in 0..n {
            let kv: f64 = (0..m).map(|j| kernel[i][j] * v[j]).sum();
            u[i] = if kv.abs() > 1e-30 {
                marginal_a[i] / kv
            } else {
                marginal_a[i]
            };
        }

        // v = b ./ (K^T * u)
        for j in 0..m {
            let ktu: f64 = (0..n).map(|i| kernel[i][j] * u[i]).sum();
            v[j] = if ktu.abs() > 1e-30 {
                marginal_b[j] / ktu
            } else {
                marginal_b[j]
            };
        }
    }

    // Transport plan: T[i][j] = diag(u) * K * diag(v)
    let transport_plan: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..m)
                .map(|j| u[i] * kernel[i][j] * v[j])
                .collect()
        })
        .collect();

    // Transport distance = Σ T[i][j] * C[i][j]
    let distance: f64 = transport_plan
        .iter()
        .zip(cost_matrix.iter())
        .map(|(t_row, c_row)| {
            t_row.iter().zip(c_row.iter()).map(|(&t, &c)| t * c).sum::<f64>()
        })
        .sum();

    SinkhornResult {
        transport_plan,
        distance,
    }
}

/// Convenience function: compute Sinkhorn distance without returning the full plan.
pub fn sinkhorn_distance(
    cost_matrix: &[Vec<f64>],
    marginal_a: &[f64],
    marginal_b: &[f64],
    lambda: f64,
    iterations: usize,
) -> f64 {
    sinkhorn_step(cost_matrix, marginal_a, marginal_b, lambda, iterations).distance
}
