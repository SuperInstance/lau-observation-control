//! Control functor: 𝒞ᵒᵖ → 𝒞
//!
//! Pushforward / actuation from internal model to world.
//! Control is the right adjoint to Observation.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::observation::{InternalModel, WorldState};

/// Control model for the control functor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlModel {
    /// Control input matrix B: maps control space to state space.
    pub b: DMatrix<f64>,
    /// State transition matrix A.
    pub a: DMatrix<f64>,
    /// Cost matrix Q (state cost).
    pub q: DMatrix<f64>,
    /// Cost matrix R (control cost).
    pub r: DMatrix<f64>,
}

impl ControlModel {
    /// Create a new control model.
    pub fn new(a: DMatrix<f64>, b: DMatrix<f64>, q: DMatrix<f64>, r: DMatrix<f64>) -> Self {
        Self { a, b, q, r }
    }

    /// Identity control model for n-dimensional state with m-dimensional control.
    pub fn identity(n: usize, m: usize) -> Self {
        Self {
            a: DMatrix::identity(n, n),
            b: DMatrix::identity(n, m),
            q: DMatrix::identity(n, n),
            r: DMatrix::identity(m, m),
        }
    }
}

/// Control functor (right adjoint).
///
/// Maps objects in 𝒞ᵒᵖ back to objects in 𝒞 and morphisms in 𝒞ᵒᵖ to morphisms in 𝒞
/// (re-reversing direction).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlFunctor {
    pub model: ControlModel,
}

impl ControlFunctor {
    pub fn new(model: ControlModel) -> Self {
        Self { model }
    }

    /// Identity control functor.
    pub fn identity(n: usize, m: usize) -> Self {
        Self {
            model: ControlModel::identity(n, m),
        }
    }

    /// Map an object: InternalModel → WorldState (actuation).
    pub fn map_object(&self, internal: &InternalModel, control: &DVector<f64>) -> WorldState {
        let state = &self.model.a * &internal.belief + &self.model.b * control;
        let covariance = &self.model.a * &internal.uncertainty * &self.model.a.transpose();
        WorldState { state, covariance }
    }

    /// Map a morphism in 𝒞ᵒᵖ to a morphism in 𝒞.
    pub fn map_morphism(&self, morphism: &DMatrix<f64>) -> DMatrix<f64> {
        &self.model.a * morphism * &self.model.a.transpose()
    }

    /// Compute LQR-optimal gain K.
    /// Solves the discrete algebraic Riccati equation iteratively.
    pub fn lqr_gain(&self) -> DMatrix<f64> {
        let _n = self.model.a.nrows();
        let m = self.model.b.ncols();
        let mut p = self.model.q.clone();

        // Iterate the DARE
        for _ in 0..100 {
            let bt_p = self.model.b.transpose() * &p;
            let s = &self.model.r + &bt_p * &self.model.b;
            let k = s.clone().try_inverse().unwrap_or(DMatrix::identity(m, m)) * &bt_p * &self.model.a;

            let new_p = &self.model.q
                + self.model.a.transpose() * &p * &self.model.a
                - self.model.a.transpose() * &p * &self.model.b * &k;

            if (&new_p - &p).norm() < 1e-10 {
                break;
            }
            p = new_p;
        }

        let bt_p = self.model.b.transpose() * &p;
        let s = &self.model.r + &bt_p * &self.model.b;
        s.clone().try_inverse().unwrap_or(DMatrix::identity(m, m)) * bt_p * &self.model.a
    }

    /// Apply LQR control: map internal model to world state via optimal control.
    pub fn lqr_actuate(&self, internal: &InternalModel) -> WorldState {
        let k = self.lqr_gain();
        let u = -&k * &internal.belief;
        self.map_object(internal, &u)
    }
}

/// LQR controller (counit of the adjunction).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LQRController {
    /// Gain matrix K.
    pub k: DMatrix<f64>,
    /// Control model reference.
    pub model: ControlModel,
}

impl LQRController {
    /// Create LQR controller by computing optimal gain.
    pub fn new(model: ControlModel) -> Self {
        let cf = ControlFunctor::new(model.clone());
        let k = cf.lqr_gain();
        Self { k, model }
    }

    /// Compute optimal control input.
    pub fn compute_control(&self, belief: &DVector<f64>) -> DVector<f64> {
        -&self.k * belief
    }

    /// Apply control and return next world state.
    pub fn step(&self, internal: &InternalModel) -> WorldState {
        let u = self.compute_control(&internal.belief);
        let state = &self.model.a * &internal.belief + &self.model.b * &u;
        let covariance = &self.model.a * &internal.uncertainty * &self.model.a.transpose();
        WorldState { state, covariance }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn test_control_model_identity() {
        let model = ControlModel::identity(2, 2);
        assert_eq!(model.a.nrows(), 2);
        assert_eq!(model.b.nrows(), 2);
    }

    #[test]
    fn test_control_functor_actuate() {
        let ctrl = ControlFunctor::identity(2, 2);
        let internal = InternalModel {
            belief: dvector![1.0, 0.0],
            uncertainty: DMatrix::identity(2, 2),
        };
        let u = dvector![0.0, 0.0];
        let world = ctrl.map_object(&internal, &u);
        assert_eq!(world.state.len(), 2);
    }

    #[test]
    fn test_control_reverses_morphisms() {
        let ctrl = ControlFunctor::identity(2, 2);
        let f = dmatrix![1.0, 0.5; 0.0, 1.0];
        let ctrl_f = ctrl.map_morphism(&f);
        assert_eq!(ctrl_f.nrows(), 2);
    }

    #[test]
    fn test_lqr_gain_computation() {
        let model = ControlModel::new(
            dmatrix![1.0, 0.1; 0.0, 1.0],
            dmatrix![0.0; 1.0],
            DMatrix::identity(2, 2),
            dmatrix![0.1],
        );
        let ctrl = ControlFunctor::new(model);
        let k = ctrl.lqr_gain();
        assert_eq!(k.nrows(), 1);
        assert_eq!(k.ncols(), 2);
    }

    #[test]
    fn test_lqr_controller_step() {
        let model = ControlModel::new(
            dmatrix![1.0, 0.1; 0.0, 1.0],
            dmatrix![0.0; 1.0],
            DMatrix::identity(2, 2),
            dmatrix![0.1],
        );
        let lqr = LQRController::new(model);
        let internal = InternalModel {
            belief: dvector![1.0, 0.0],
            uncertainty: DMatrix::identity(2, 2),
        };
        let next = lqr.step(&internal);
        assert_eq!(next.state.len(), 2);
    }

    #[test]
    fn test_lqr_stabilizes() {
        let model = ControlModel::new(
            dmatrix![1.0, 0.1; 0.0, 1.0],
            dmatrix![0.0; 1.0],
            DMatrix::identity(2, 2),
            dmatrix![0.1],
        );
        let lqr = LQRController::new(model);
        let mut belief = dvector![10.0, 5.0];
        let a = dmatrix![1.0, 0.1; 0.0, 1.0];
        let b = dmatrix![0.0; 1.0];
        // Simulate closed-loop: x_{k+1} = (A - B*K) * x_k
        for _ in 0..100 {
            let u = -&lqr.k * &belief;
            belief = &a * &belief + &b * &u;
        }
        assert!(belief.norm() < 1.0, "LQR should stabilize the system, got norm = {}", belief.norm());
    }
}
