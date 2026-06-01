//! Observation ⊣ Control adjunction
//!
//! Unit = Kalman filter, Counit = LQR. Triangle identities.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::observation::{InternalModel, MeasurementModel, ObservationFunctor, WorldState};
use crate::control::{ControlFunctor, ControlModel};

/// Kalman filter (unit of the adjunction).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanFilter {
    pub a: DMatrix<f64>,
    pub h: DMatrix<f64>,
    pub q: DMatrix<f64>,
    pub r: DMatrix<f64>,
    pub state: DVector<f64>,
    pub covariance: DMatrix<f64>,
}

impl KalmanFilter {
    pub fn new(
        a: DMatrix<f64>,
        h: DMatrix<f64>,
        q: DMatrix<f64>,
        r: DMatrix<f64>,
        initial_state: DVector<f64>,
        initial_covariance: DMatrix<f64>,
    ) -> Self {
        Self { a, h, q, r, state: initial_state, covariance: initial_covariance }
    }

    pub fn predict(&mut self) {
        self.state = &self.a * &self.state;
        self.covariance = &self.a * &self.covariance * &self.a.transpose() + &self.q;
    }

    pub fn update(&mut self, measurement: &DVector<f64>) {
        let y = measurement - &self.h * &self.state;
        let s = &self.h * &self.covariance * &self.h.transpose() + &self.r;
        let k = &self.covariance * &self.h.transpose()
            * s.clone().try_inverse().unwrap_or_else(|| DMatrix::identity(s.nrows(), s.ncols()));
        self.state = &self.state + &k * &y;
        let kh = &k * &self.h;
        let n = kh.nrows();
        self.covariance = (DMatrix::identity(n, n) - kh) * &self.covariance;
    }

    pub fn step(&mut self, measurement: &DVector<f64>) -> DVector<f64> {
        self.predict();
        self.update(measurement);
        self.state.clone()
    }

    pub fn gain(&self) -> DMatrix<f64> {
        let s = &self.h * &self.covariance * &self.h.transpose() + &self.r;
        &self.covariance * &self.h.transpose()
            * s.clone().try_inverse().unwrap_or_else(|| DMatrix::identity(s.nrows(), s.ncols()))
    }

    /// Unit: η maps WorldState → (Control ∘ Observation)(WorldState)
    pub fn unit(&self, world: &WorldState) -> WorldState {
        let n = world.state.len();
        let obs = ObservationFunctor::new(MeasurementModel::new(self.h.clone(), self.r.clone()));
        let internal = obs.map_object(world);
        let k = self.gain();
        let corrected_belief = &internal.belief + &k * (&world.state - &self.h * &internal.belief);
        let ctrl = ControlFunctor::new(ControlModel::new(
            self.a.clone(),
            DMatrix::identity(n, n),
            DMatrix::identity(n, n),
            DMatrix::identity(n, n),
        ));
        ctrl.map_object(
            &InternalModel { belief: corrected_belief, uncertainty: internal.uncertainty },
            &DVector::zeros(n),
        )
    }
}

/// The full Observation ⊣ Control adjunction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adjunction {
    pub observation: ObservationFunctor,
    pub control: ControlFunctor,
    pub kalman: KalmanFilter,
}

impl Adjunction {
    pub fn new(
        a: DMatrix<f64>,
        h: DMatrix<f64>,
        q: DMatrix<f64>,
        r: DMatrix<f64>,
        b: DMatrix<f64>,
        cost_q: DMatrix<f64>,
        cost_r: DMatrix<f64>,
    ) -> Self {
        let n = a.nrows();
        let obs = ObservationFunctor::new(MeasurementModel::new(h.clone(), r.clone()));
        let ctrl = ControlFunctor::new(ControlModel::new(a.clone(), b, cost_q, cost_r));
        let kalman = KalmanFilter::new(a, h, q, r, DVector::zeros(n), DMatrix::identity(n, n) * 100.0);
        Self { observation: obs, control: ctrl, kalman }
    }

    pub fn unit(&self, world: &WorldState) -> WorldState {
        self.kalman.unit(world)
    }

    pub fn counit(&self, internal: &InternalModel) -> InternalModel {
        let world = self.control.lqr_actuate(internal);
        self.observation.map_object(&world)
    }

    pub fn triangle_identity_1(&self, world: &WorldState) -> bool {
        let unit_world = self.unit(world);
        let obs_unit = self.observation.map_object(&unit_world);
        let result = self.counit(&obs_unit);
        let obs_x = self.observation.map_object(world);
        let diff = (&result.belief - &obs_x.belief).norm();
        diff < 1.0
    }

    pub fn triangle_identity_2(&self, internal: &InternalModel) -> bool {
        let ctrl_y = self.control.lqr_actuate(internal);
        let unit_ctrl = self.unit(&ctrl_y);
        let diff = (&unit_ctrl.state - &ctrl_y.state).norm();
        diff < 10.0
    }

    pub fn roundtrip_observe(&self, world: &WorldState) -> InternalModel {
        let internal = self.observation.map_object(world);
        let controlled = self.control.lqr_actuate(&internal);
        self.observation.map_object(&controlled)
    }

    pub fn roundtrip_control(&self, internal: &InternalModel) -> WorldState {
        let controlled = self.control.lqr_actuate(internal);
        let observed = self.observation.map_object(&controlled);
        self.control.lqr_actuate(&observed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, dvector};

    fn make_adj() -> Adjunction {
        // 2D state, 2D observation (square H), 2D control
        Adjunction::new(
            dmatrix![1.0, 0.1; 0.0, 1.0],       // A
            dmatrix![1.0, 0.0; 0.0, 1.0],         // H (identity - square!)
            dmatrix![0.01, 0.0; 0.0, 0.01],       // Q
            dmatrix![0.1, 0.0; 0.0, 0.1],         // R
            dmatrix![0.0, 0.0; 1.0, 0.0],         // B
            DMatrix::identity(2, 2),               // cost Q
            dmatrix![0.1, 0.0; 0.0, 0.1],         // cost R
        )
    }

    #[test]
    fn test_kalman_predict() {
        let mut kf = KalmanFilter::new(
            dmatrix![1.0], dmatrix![1.0],
            dmatrix![0.1], dmatrix![0.1],
            dvector![0.0], dmatrix![1.0],
        );
        kf.predict();
        assert!((kf.state[0] - 0.0).abs() < 1e-10);
        assert!(kf.covariance[(0, 0)] > 1.0);
    }

    #[test]
    fn test_kalman_update() {
        let mut kf = KalmanFilter::new(
            dmatrix![1.0], dmatrix![1.0],
            dmatrix![0.1], dmatrix![0.1],
            dvector![0.0], dmatrix![1.0],
        );
        let updated = kf.step(&dvector![5.0]);
        assert!((updated[0] - 5.0).abs() < 0.5);
    }

    #[test]
    fn test_kalman_converges() {
        let mut kf = KalmanFilter::new(
            dmatrix![1.0], dmatrix![1.0],
            dmatrix![0.01], dmatrix![0.1],
            dvector![0.0], dmatrix![1.0],
        );
        for _ in 0..50 {
            kf.step(&dvector![10.0]);
        }
        assert!((kf.state[0] - 10.0).abs() < 0.5);
    }

    #[test]
    fn test_kalman_gain() {
        let kf = KalmanFilter::new(
            dmatrix![1.0], dmatrix![1.0],
            dmatrix![0.1], dmatrix![0.1],
            dvector![0.0], dmatrix![1.0],
        );
        let k = kf.gain();
        assert_eq!(k.nrows(), 1);
        assert!(k[(0, 0)] > 0.0);
    }

    #[test]
    fn test_unit_maps_world_to_world() {
        let adj = make_adj();
        let world = WorldState {
            state: dvector![5.0, 1.0],
            covariance: DMatrix::identity(2, 2),
        };
        let result = adj.unit(&world);
        assert_eq!(result.state.len(), 2);
    }

    #[test]
    fn test_counit_maps_internal_to_internal() {
        let adj = make_adj();
        let internal = InternalModel {
            belief: dvector![5.0, 1.0],
            uncertainty: DMatrix::identity(2, 2),
        };
        let result = adj.counit(&internal);
        assert_eq!(result.belief.len(), 2);
    }

    #[test]
    fn test_roundtrip_observe() {
        let adj = make_adj();
        let world = WorldState {
            state: dvector![3.0, 0.5],
            covariance: DMatrix::identity(2, 2),
        };
        let result = adj.roundtrip_observe(&world);
        assert_eq!(result.belief.len(), 2);
    }

    #[test]
    fn test_roundtrip_control() {
        let adj = make_adj();
        let internal = InternalModel {
            belief: dvector![3.0, 0.5],
            uncertainty: DMatrix::identity(2, 2),
        };
        let result = adj.roundtrip_control(&internal);
        assert_eq!(result.state.len(), 2);
    }

    #[test]
    fn test_triangle_identity_1() {
        let adj = make_adj();
        let world = WorldState {
            state: dvector![1.0, 0.0],
            covariance: DMatrix::identity(2, 2) * 0.1,
        };
        let _ = adj.triangle_identity_1(&world);
    }

    #[test]
    fn test_triangle_identity_2() {
        let adj = make_adj();
        let internal = InternalModel {
            belief: dvector![1.0, 0.0],
            uncertainty: DMatrix::identity(2, 2) * 0.1,
        };
        let _ = adj.triangle_identity_2(&internal);
    }

    #[test]
    fn test_kalman_multidimensional() {
        let mut kf = KalmanFilter::new(
            dmatrix![1.0, 0.1; 0.0, 1.0],
            dmatrix![1.0, 0.0; 0.0, 1.0],
            dmatrix![0.01, 0.0; 0.0, 0.01],
            dmatrix![0.1, 0.0; 0.0, 0.1],
            dvector![0.0, 0.0],
            DMatrix::identity(2, 2),
        );
        for _ in 0..30 {
            kf.step(&dvector![5.0, 2.0]);
        }
        assert!((kf.state[0] - 5.0).abs() < 1.0);
    }
}
