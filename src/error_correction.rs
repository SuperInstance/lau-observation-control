//! Error as 1-morphism, correction as 2-morphism
//!
//! From lau-error-correcting-codes + kintsugi philosophy:
//! Errors are 1-morphisms in a 2-category, corrections are 2-morphisms between them.
//! The golden repair makes error ~ correction a homotopy equivalence.

use nalgebra::{DVector, DMatrix};
use serde::{Deserialize, Serialize};

/// An error as a 1-morphism: a deviation from expected state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Error {
    /// Error vector: deviation from expected.
    pub deviation: DVector<f64>,
    /// Error covariance.
    pub covariance: DMatrix<f64>,
    /// Error magnitude.
    pub magnitude: f64,
}

impl Error {
    pub fn new(deviation: DVector<f64>, covariance: DMatrix<f64>) -> Self {
        let magnitude = deviation.norm();
        Self { deviation, covariance, magnitude }
    }

    /// Compose errors (1-morphism composition): e₂ ∘ e₁.
    pub fn compose(&self, other: &Error) -> Error {
        Error::new(
            &self.deviation + &other.deviation,
            &self.covariance + &other.covariance,
        )
    }

    /// Identity error (zero error).
    pub fn identity(n: usize) -> Self {
        Error::new(
            DVector::zeros(n),
            DMatrix::zeros(n, n),
        )
    }

    /// Is this the identity (zero error)?
    pub fn is_identity(&self) -> bool {
        self.magnitude < 1e-10
    }
}

/// A correction as a 2-morphism: maps between errors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Correction {
    /// The correction applied.
    pub delta: DVector<f64>,
    /// Confidence in the correction.
    pub confidence: f64,
    /// Source error.
    pub source: Error,
    /// Target error (after correction).
    pub target: Error,
}

impl Correction {
    pub fn new(delta: DVector<f64>, confidence: f64, source: Error) -> Self {
        let target_deviation = &source.deviation - &delta;
        let target = Error::new(
            target_deviation,
            source.covariance.clone(),
        );
        Self { delta, confidence, source, target }
    }

    /// Compose corrections (vertical composition of 2-morphisms).
    pub fn compose_vertical(&self, other: &Correction) -> Correction {
        let combined_delta = &self.delta + &other.delta;
        let combined_confidence = self.confidence * other.confidence;
        Correction::new(combined_delta, combined_confidence, self.source.clone())
    }

    /// Is this correction an isomorphism? (error → zero)
    pub fn is_isomorphism(&self) -> bool {
        self.target.is_identity()
    }

    /// Quality: how much of the error was corrected.
    pub fn quality(&self) -> f64 {
        if self.source.magnitude < 1e-10 {
            return 1.0;
        }
        1.0 - (self.target.magnitude / self.source.magnitude)
    }
}

/// Error-correcting code context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorCorrectingContext {
    /// Code dimension.
    pub dimension: usize,
    /// Minimum distance.
    pub min_distance: usize,
    /// Parity check matrix.
    pub parity: DMatrix<f64>,
}

impl ErrorCorrectingContext {
    pub fn new(parity: DMatrix<f64>) -> Self {
        let dimension = parity.ncols();
        let min_distance = 2; // simplified
        Self { dimension, min_distance, parity }
    }

    /// Detect error syndrome.
    pub fn syndrome(&self, error: &Error) -> DVector<f64> {
        &self.parity * &error.deviation
    }

    /// Can this error be corrected?
    pub fn is_correctable(&self, error: &Error) -> bool {
        error.magnitude > 0.0 && (error.magnitude as usize) < self.min_distance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn test_error_creation() {
        let e = Error::new(dvector![1.0, 0.0], DMatrix::identity(2, 2));
        assert!((e.magnitude - 1.0).abs() < 1e-10);
        assert!(!e.is_identity());
    }

    #[test]
    fn test_error_identity() {
        let e = Error::identity(3);
        assert!(e.is_identity());
        assert_eq!(e.deviation.len(), 3);
    }

    #[test]
    fn test_error_composition() {
        let e1 = Error::new(dvector![1.0], dmatrix![1.0]);
        let e2 = Error::new(dvector![2.0], dmatrix![1.0]);
        let composed = e1.compose(&e2);
        assert!((composed.deviation[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_correction_basic() {
        let e = Error::new(dvector![3.0], dmatrix![1.0]);
        let c = Correction::new(dvector![3.0], 0.95, e);
        assert!(c.is_isomorphism());
        assert!(c.quality() > 0.99);
    }

    #[test]
    fn test_correction_partial() {
        let e = Error::new(dvector![4.0], dmatrix![1.0]);
        let c = Correction::new(dvector![2.0], 0.8, e);
        assert!(!c.is_isomorphism());
        assert!((c.quality() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_correction_vertical_composition() {
        let e = Error::new(dvector![5.0], dmatrix![1.0]);
        let c1 = Correction::new(dvector![2.0], 0.9, e);
        let c2 = Correction::new(dvector![1.0], 0.8, c1.target.clone());
        let composed = c1.compose_vertical(&c2);
        assert!((composed.delta[0] - 3.0).abs() < 1e-10);
        assert!((composed.confidence - 0.72).abs() < 1e-10);
    }

    #[test]
    fn test_ecc_syndrome() {
        let parity = dmatrix![1.0, 1.0, 0.0; 0.0, 1.0, 1.0];
        let ctx = ErrorCorrectingContext::new(parity);
        let e = Error::new(dvector![1.0, 0.0, 0.0], DMatrix::identity(3, 3));
        let syndrome = ctx.syndrome(&e);
        assert_eq!(syndrome.len(), 2);
    }

    #[test]
    fn test_ecc_correctable() {
        let parity = dmatrix![1.0, 1.0, 0.0; 0.0, 1.0, 1.0];
        let ctx = ErrorCorrectingContext::new(parity);
        let e = Error::new(dvector![0.5], DMatrix::identity(1, 1));
        assert!(ctx.is_correctable(&e));
    }
}
