/// Compute the Fisher information metric (KL divergence approximation) between two room configurations.
///
/// Each room is represented as a probability distribution over discrete states.
/// Returns the symmetric KL divergence: (KL(a||b) + KL(b||a)) / 2.
pub fn room_fisher_metric(room_a: &[f64], room_b: &[f64]) -> f64 {
    assert_eq!(
        room_a.len(),
        room_b.len(),
        "Rooms must have the same number of states"
    );
    assert!(!room_a.is_empty(), "Rooms must not be empty");

    let kl_ab = kl_divergence(room_a, room_b);
    let kl_ba = kl_divergence(room_b, room_a);
    (kl_ab + kl_ba) / 2.0
}

/// Natural gradient descent for room configuration optimization.
///
/// Adjusts the gradient by the inverse Fisher information matrix (diagonal approximation).
/// `params` - current configuration parameters
/// `gradient` - Euclidean gradient
/// `fisher` - diagonal of the Fisher information matrix
///
/// Returns the natural gradient: F^{-1} * ∇
pub fn room_natural_gradient(
    params: &[f64],
    gradient: &[f64],
    fisher: &[f64],
) -> Vec<f64> {
    assert_eq!(params.len(), gradient.len());
    assert_eq!(params.len(), fisher.len());

    gradient
        .iter()
        .zip(fisher.iter())
        .map(|(&g, &f)| {
            if f.abs() > 1e-12 {
                g / f
            } else {
                g // No Fisher info → fall back to Euclidean gradient
            }
        })
        .collect()
}

/// KL divergence D_KL(p || q) = Σ p_i * ln(p_i / q_i)
fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            if pi < 1e-15 {
                0.0
            } else if qi < 1e-15 {
                // Divergent — penalize heavily
                f64::INFINITY
            } else {
                pi * (pi / qi).ln()
            }
        })
        .sum()
}
