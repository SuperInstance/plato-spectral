const DEFAULT_TOLERANCE: f64 = 1e-9;

/// Check tile flow conservation: incoming flow must equal outgoing flow within tolerance.
///
/// Returns true if |Σ incoming - Σ outgoing| ≤ tolerance.
pub fn tile_flow_conservation(incoming: &[f64], outgoing: &[f64]) -> bool {
    tile_flow_conservation_with_tolerance(incoming, outgoing, DEFAULT_TOLERANCE)
}

/// Check tile flow conservation with a custom tolerance.
pub fn tile_flow_conservation_with_tolerance(
    incoming: &[f64],
    outgoing: &[f64],
    tolerance: f64,
) -> bool {
    let sum_in: f64 = incoming.iter().copied().sum();
    let sum_out: f64 = outgoing.iter().copied().sum();
    (sum_in - sum_out).abs() <= tolerance
}

/// A transformation step in a processing chain, with input and output energy.
#[derive(Debug, Clone)]
pub struct TransformStep {
    pub name: String,
    pub input_energy: f64,
    pub output_energy: f64,
}

/// Check energy conservation across a chain of transformations.
///
/// Returns true if the total output energy of each step equals the input energy of the next,
/// and the initial input equals the final output (within tolerance).
pub fn energy_conservation(transform_chain: &[TransformStep]) -> bool {
    energy_conservation_with_tolerance(transform_chain, DEFAULT_TOLERANCE)
}

/// Check energy conservation with a custom tolerance.
pub fn energy_conservation_with_tolerance(
    transform_chain: &[TransformStep],
    tolerance: f64,
) -> bool {
    if transform_chain.is_empty() {
        return true;
    }

    // Check that each step conserves energy locally
    for step in transform_chain {
        if (step.input_energy - step.output_energy).abs() > tolerance {
            return false;
        }
    }

    // Check chain continuity: output of step i ≈ input of step i+1
    for window in transform_chain.windows(2) {
        if (window[0].output_energy - window[1].input_energy).abs() > tolerance {
            return false;
        }
    }

    true
}
