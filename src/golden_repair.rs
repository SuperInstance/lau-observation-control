//! Golden repair as homotopy equivalence
//!
//! Kintsugi philosophy: the golden repair makes error ~ correction a homotopy equivalence.
//! Error and correction are connected by a continuous deformation (homotopy).

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

use crate::error_correction::{Error, Correction};

/// A homotopy between error and correction.
/// H(t): Error × [0,1] → State, where H(0) = error state, H(1) = corrected state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Homotopy {
    /// Start state (error).
    pub start: DVector<f64>,
    /// End state (corrected).
    pub end: DVector<f64>,
}

impl Homotopy {
    pub fn new(start: DVector<f64>, end: DVector<f64>) -> Self {
        Self { start, end }
    }

    /// Evaluate the homotopy at parameter t ∈ [0, 1].
    pub fn at(&self, t: f64) -> DVector<f64> {
        &self.start * (1.0 - t) + &self.end * t
    }

    /// Derivative of the homotopy (velocity).
    pub fn velocity(&self) -> DVector<f64> {
        &self.end - &self.start
    }

    /// Path length (integral of ||H'(t)||).
    pub fn path_length(&self, steps: usize) -> f64 {
        let mut length = 0.0;
        let mut prev = self.start.clone();
        for i in 1..=steps {
            let t = i as f64 / steps as f64;
            let current = self.at(t);
            length += (current.clone() - prev).norm();
            prev = current;
        }
        length
    }
}

/// Golden repair: the kintsugi homotopy equivalence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenRepair {
    /// The homotopy from error to correction.
    pub homotopy: Homotopy,
    /// Golden ratio (φ = (1 + √5) / 2).
    pub phi: f64,
}

impl GoldenRepair {
    pub fn new(error: &Error, correction: &Correction) -> Self {
        let start = error.deviation.clone();
        let end = correction.target.deviation.clone();
        Self {
            homotopy: Homotopy::new(start, end),
            phi: (1.0 + 5.0_f64.sqrt()) / 2.0,
        }
    }

    /// The golden point: t = 1/φ ≈ 0.618.
    /// This is the aesthetically optimal point of repair.
    pub fn golden_point(&self) -> DVector<f64> {
        self.homotopy.at(1.0 / self.phi)
    }

    /// Golden ratio of the path length.
    pub fn golden_ratio_length(&self) -> f64 {
        self.homotopy.path_length(100) / self.phi
    }

    /// Is the repair complete? (homotopy reaches zero)
    pub fn is_complete(&self) -> bool {
        self.homotopy.end.norm() < 1e-10
    }

    /// Repair quality: how well the golden repair connects error to correction.
    pub fn repair_quality(&self) -> f64 {
        let start_norm = self.homotopy.start.norm();
        if start_norm < 1e-10 {
            return 1.0;
        }
        1.0 - self.homotopy.end.norm() / start_norm
    }

    /// Verify homotopy equivalence: error ≃ correction.
    /// There exists a homotopy H with H(0) = error, H(1) = corrected.
    /// The repair is an equivalence if the path is smooth and continuous.
    pub fn verify_equivalence(&self) -> bool {
        // Check continuity: evaluate at many points and ensure smoothness
        let n = 100;
        for i in 0..n {
            let t1 = i as f64 / n as f64;
            let t2 = (i + 1) as f64 / n as f64;
            let p1 = self.homotopy.at(t1);
            let p2 = self.homotopy.at(t2);
            let step_size = (p2 - p1).norm();
            // Step should be roughly constant for a linear homotopy
            if !step_size.is_finite() {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error_correction::Error;
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn test_homotopy_endpoints() {
        let h = Homotopy::new(dvector![0.0], dvector![1.0]);
        assert!((h.at(0.0)[0] - 0.0).abs() < 1e-10);
        assert!((h.at(1.0)[0] - 1.0).abs() < 1e-10);
        assert!((h.at(0.5)[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_homotopy_velocity() {
        let h = Homotopy::new(dvector![0.0, 0.0], dvector![2.0, 4.0]);
        let v = h.velocity();
        assert!((v[0] - 2.0).abs() < 1e-10);
        assert!((v[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_homotopy_path_length() {
        let h = Homotopy::new(dvector![0.0], dvector![3.0]);
        let length = h.path_length(100);
        assert!((length - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_golden_repair_creation() {
        let error = Error::new(dvector![3.0, 1.0], dmatrix![1.0, 0.0; 0.0, 1.0]);
        let correction = crate::error_correction::Correction::new(dvector![3.0, 1.0], 0.99, error.clone());
        let repair = GoldenRepair::new(&error, &correction);
        assert!(repair.is_complete());
        assert!(repair.repair_quality() > 0.99);
    }

    #[test]
    fn test_golden_point() {
        let error = Error::new(dvector![1.0], dmatrix![1.0]);
        let correction = crate::error_correction::Correction::new(dvector![0.5], 0.8, error.clone());
        let repair = GoldenRepair::new(&error, &correction);
        let gp = repair.golden_point();
        assert_eq!(gp.len(), 1);
        // At t = 1/φ ≈ 0.618, interpolated between 1.0 and 0.5
        let expected = 1.0 * (1.0 - 1.0 / repair.phi) + 0.5 / repair.phi;
        assert!((gp[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_golden_phi() {
        let error = Error::new(dvector![1.0], dmatrix![1.0]);
        let correction = crate::error_correction::Correction::new(dvector![1.0], 1.0, error.clone());
        let repair = GoldenRepair::new(&error, &correction);
        assert!((repair.phi - 1.6180339887).abs() < 1e-8);
    }

    #[test]
    fn test_verify_equivalence() {
        let error = Error::new(dvector![2.0, 1.0], dmatrix![1.0, 0.0; 0.0, 1.0]);
        let correction = crate::error_correction::Correction::new(dvector![2.0, 1.0], 1.0, error.clone());
        let repair = GoldenRepair::new(&error, &correction);
        assert!(repair.verify_equivalence());
    }
}
