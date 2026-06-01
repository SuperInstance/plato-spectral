#![deny(unsafe_code)]

pub mod alert_dft;
pub mod conservation;
pub mod distill_transport;
pub mod fisher_rooms;
pub mod health_spectral;

pub use alert_dft::{
    alert_spectral_energy, compute_dft, identify_dominant_frequencies, AlertDftResult,
};
pub use conservation::{energy_conservation, tile_flow_conservation, tile_flow_conservation_with_tolerance};
pub use distill_transport::{sinkhorn_distance, sinkhorn_step, SinkhornResult};
pub use fisher_rooms::{room_fisher_metric, room_natural_gradient};
pub use health_spectral::{health_spectral_decomposition, HealthDecomposition};
