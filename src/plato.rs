//! PLATO observe-predict-control loop

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::observation::{InternalModel, MeasurementModel, ObservationFunctor, WorldState};
use crate::control::{ControlFunctor, ControlModel};
use crate::adjunction::{Adjunction, KalmanFilter};
use crate::agent_loop::AgentLoop;
use crate::computability::{ComputabilityFunctor, ComputabilityConfig};
use crate::error_correction::{Error, Correction};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatoConfig {
    pub state_dim: usize,
    pub control_dim: usize,
    pub observation_dim: usize,
    pub bits_per_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatoSystem {
    pub config: PlatoConfig,
    pub adjunction: Adjunction,
    pub agent_loop: AgentLoop,
    pub computability: ComputabilityFunctor,
}

impl PlatoSystem {
    pub fn new(
        state_dim: usize,
        control_dim: usize,
        _observation_dim: usize,
        a: DMatrix<f64>,
        b: DMatrix<f64>,
    ) -> Self {
        let config = PlatoConfig {
            state_dim, control_dim,
            observation_dim: state_dim, // square H
            bits_per_dim: 8,
        };

        let h = DMatrix::identity(state_dim, state_dim);
        let q_process = DMatrix::identity(state_dim, state_dim) * 0.01;
        let r_measure = DMatrix::identity(state_dim, state_dim) * 0.1;
        let cost_q = DMatrix::identity(state_dim, state_dim);
        let cost_r = DMatrix::identity(control_dim, control_dim) * 0.1;

        let adjunction = Adjunction::new(
            a.clone(), h.clone(),
            q_process.clone(), r_measure.clone(),
            b.clone(), cost_q, cost_r,
        );

        let obs = ObservationFunctor::new(MeasurementModel::new(h.clone(), r_measure));
        let ctrl = ControlFunctor::new(ControlModel::new(a.clone(), b, DMatrix::identity(state_dim, state_dim), DMatrix::identity(state_dim, state_dim)));
        let kf = KalmanFilter::new(
            a, h,
            q_process,
            DMatrix::identity(state_dim, state_dim) * 0.1,
            DVector::zeros(state_dim),
            DMatrix::identity(state_dim, state_dim) * 100.0,
        );

        let agent_loop = AgentLoop::new(obs, ctrl, kf);
        let computability = ComputabilityFunctor::new(ComputabilityConfig {
            bits_per_dim: config.bits_per_dim,
            scale: 10.0,
            offset: 0.0,
        });

        Self { config, adjunction, agent_loop, computability }
    }

    pub fn observe(&mut self, world: &WorldState) -> InternalModel {
        self.adjunction.observation.map_object(world)
    }

    pub fn predict(&self, internal: &InternalModel) -> InternalModel {
        let predicted_belief = &self.adjunction.kalman.a * &internal.belief;
        let predicted_uncertainty = &self.adjunction.kalman.a * &internal.uncertainty
            * &self.adjunction.kalman.a.transpose()
            + &self.adjunction.kalman.q;
        InternalModel { belief: predicted_belief, uncertainty: predicted_uncertainty }
    }

    pub fn control(&self, internal: &InternalModel) -> WorldState {
        self.adjunction.control.lqr_actuate(internal)
    }

    pub fn cycle(&mut self, world: &WorldState) -> (InternalModel, InternalModel, WorldState) {
        let obs = self.observe(world);
        let pred = self.predict(&obs);
        let ctrl = self.control(&pred);
        (obs, pred, ctrl)
    }

    pub fn compute_error(&self, observed: &InternalModel, predicted: &InternalModel) -> Error {
        Error::new(&observed.belief - &predicted.belief, &observed.uncertainty + &predicted.uncertainty)
    }

    pub fn apply_correction(&self, error: &Error, internal: &InternalModel) -> InternalModel {
        let correction = Correction::new(error.deviation.clone(), 0.9, error.clone());
        InternalModel {
            belief: &internal.belief + &correction.delta,
            uncertainty: internal.uncertainty.clone(),
        }
    }

    pub fn ground(&self, internal: &InternalModel) -> crate::computability::FinSet {
        self.computability.map_object(&internal.belief)
    }

    pub fn run(&mut self, initial_world: &WorldState, n: usize) -> WorldState {
        let mut world = initial_world.clone();
        for _ in 0..n {
            let (_, _, new_world) = self.cycle(&world);
            world = new_world;
        }
        world
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, dvector};

    fn make_plato() -> PlatoSystem {
        PlatoSystem::new(
            2, 1, 2,
            dmatrix![1.0, 0.1; 0.0, 1.0],
            dmatrix![0.0; 1.0],
        )
    }

    #[test]
    fn test_plato_observe() {
        let mut p = make_plato();
        let world = WorldState { state: dvector![1.0, 0.5], covariance: DMatrix::identity(2, 2) };
        let internal = p.observe(&world);
        assert_eq!(internal.belief.len(), 2);
    }

    #[test]
    fn test_plato_predict() {
        let p = make_plato();
        let internal = InternalModel { belief: dvector![1.0, 0.5], uncertainty: DMatrix::identity(2, 2) };
        let pred = p.predict(&internal);
        assert_eq!(pred.belief.len(), 2);
    }

    #[test]
    fn test_plato_control() {
        let p = make_plato();
        let internal = InternalModel { belief: dvector![1.0, 0.5], uncertainty: DMatrix::identity(2, 2) };
        let world = p.control(&internal);
        assert_eq!(world.state.len(), 2);
    }

    #[test]
    fn test_plato_cycle() {
        let mut p = make_plato();
        let world = WorldState { state: dvector![3.0, 1.0], covariance: DMatrix::identity(2, 2) };
        let (obs, pred, ctrl) = p.cycle(&world);
        assert_eq!(obs.belief.len(), 2);
        assert_eq!(pred.belief.len(), 2);
        assert_eq!(ctrl.state.len(), 2);
    }

    #[test]
    fn test_plato_error() {
        let p = make_plato();
        let obs = InternalModel { belief: dvector![3.0, 1.0], uncertainty: DMatrix::identity(2, 2) };
        let pred = InternalModel { belief: dvector![2.5, 0.8], uncertainty: DMatrix::identity(2, 2) };
        let error = p.compute_error(&obs, &pred);
        assert!((error.deviation[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_plato_correction() {
        let p = make_plato();
        let error = Error::new(dvector![0.5, 0.2], DMatrix::identity(2, 2));
        let internal = InternalModel { belief: dvector![2.5, 0.8], uncertainty: DMatrix::identity(2, 2) };
        let corrected = p.apply_correction(&error, &internal);
        assert!((corrected.belief[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_plato_ground() {
        let p = make_plato();
        let internal = InternalModel { belief: dvector![1.5, 0.5], uncertainty: DMatrix::identity(2, 2) };
        let finset = p.ground(&internal);
        assert_eq!(finset.source_dim, 2);
        assert_eq!(finset.bits.len(), 16);
    }

    #[test]
    fn test_plato_run_multiple_cycles() {
        let mut p = make_plato();
        let world = WorldState { state: dvector![5.0, 2.0], covariance: DMatrix::identity(2, 2) };
        let result = p.run(&world, 10);
        assert_eq!(result.state.len(), 2);
    }
}
