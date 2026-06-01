//! Observation functor: 𝒞 → 𝒞ᵒᵖ
//!
//! Sheaf pullback / measurement from world to internal model.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A world state in category 𝒞.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldState {
    pub state: DVector<f64>,
    pub covariance: DMatrix<f64>,
}

/// An internal representation in category 𝒞ᵒᵖ.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternalModel {
    pub belief: DVector<f64>,
    pub uncertainty: DMatrix<f64>,
}

/// Measurement model for the observation functor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementModel {
    pub h: DMatrix<f64>,
    pub r: DMatrix<f64>,
}

impl MeasurementModel {
    pub fn new(h: DMatrix<f64>, r: DMatrix<f64>) -> Self {
        Self { h, r }
    }

    /// Identity measurement model for n-dimensional state.
    pub fn identity(n: usize) -> Self {
        Self {
            h: DMatrix::identity(n, n),
            r: DMatrix::identity(n, n) * 0.1,
        }
    }

    /// Apply observation: extract measurement from world state.
    pub fn observe(&self, world: &WorldState) -> InternalModel {
        let belief = &self.h * &world.state;
        let uncertainty = &self.h * &world.covariance * &self.h.transpose() + &self.r;
        InternalModel { belief, uncertainty }
    }
}

/// Observation functor (left adjoint).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationFunctor {
    pub model: MeasurementModel,
}

impl ObservationFunctor {
    pub fn new(model: MeasurementModel) -> Self {
        Self { model }
    }

    pub fn identity(n: usize) -> Self {
        Self { model: MeasurementModel::identity(n) }
    }

    pub fn map_object(&self, world: &WorldState) -> InternalModel {
        self.model.observe(world)
    }

    pub fn map_morphism(&self, transition: &DMatrix<f64>) -> DMatrix<f64> {
        &self.model.h * transition * &self.model.h.transpose()
    }

    pub fn compose_morphisms(&self, f: &DMatrix<f64>, g: &DMatrix<f64>) -> DMatrix<f64> {
        let obs_f = self.map_morphism(f);
        let obs_g = self.map_morphism(g);
        &obs_f * &obs_g
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn test_world_state_creation() {
        let ws = WorldState {
            state: dvector![1.0, 2.0],
            covariance: dmatrix![1.0, 0.0; 0.0, 1.0],
        };
        assert_eq!(ws.state.len(), 2);
        assert_eq!(ws.covariance.nrows(), 2);
    }

    #[test]
    fn test_measurement_model_observe() {
        let model = MeasurementModel::identity(2);
        let world = WorldState {
            state: dvector![3.0, 4.0],
            covariance: dmatrix![1.0, 0.0; 0.0, 1.0],
        };
        let internal = model.observe(&world);
        assert_eq!(internal.belief.len(), 2);
        assert!((internal.belief[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_observation_functor_identity() {
        let obs = ObservationFunctor::identity(3);
        let world = WorldState {
            state: dvector![1.0, 2.0, 3.0],
            covariance: DMatrix::identity(3, 3),
        };
        let model = obs.map_object(&world);
        assert_eq!(model.belief.len(), 3);
    }

    #[test]
    fn test_observation_reverses_morphisms() {
        let obs = ObservationFunctor::identity(2);
        let f = dmatrix![1.0, 0.5; 0.0, 1.0];
        let obs_f = obs.map_morphism(&f);
        assert_eq!(obs_f.nrows(), 2);
    }

    #[test]
    fn test_observation_composition() {
        let obs = ObservationFunctor::identity(2);
        let f = dmatrix![1.0, 0.0; 0.0, 1.0];
        let g = dmatrix![2.0, 0.0; 0.0, 2.0];
        let composed = obs.compose_morphisms(&f, &g);
        assert_eq!(composed.nrows(), 2);
    }

    #[test]
    fn test_measurement_model_custom() {
        let h = dmatrix![1.0, 0.0]; // observe only first component
        let r = dmatrix![0.1];
        let model = MeasurementModel::new(h, r);
        let world = WorldState {
            state: dvector![5.0, 7.0],
            covariance: DMatrix::identity(2, 2),
        };
        let internal = model.observe(&world);
        assert_eq!(internal.belief.len(), 1);
        assert!((internal.belief[0] - 5.0).abs() < 1e-10);
    }
}
