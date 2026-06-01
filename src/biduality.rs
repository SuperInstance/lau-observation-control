//! Biduality closure


use serde::{Deserialize, Serialize};

#[cfg(test)] use nalgebra::DMatrix;

use crate::observation::{InternalModel, ObservationFunctor, WorldState};
use crate::control::ControlFunctor;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Biduality {
    pub observation: ObservationFunctor,
    pub control: ControlFunctor,
}

impl Biduality {
    pub fn new(observation: ObservationFunctor, control: ControlFunctor) -> Self {
        Self { observation, control }
    }

    pub fn observe_observe(&self, world: &WorldState) -> InternalModel {
        let first = self.observation.map_object(world);
        let controlled = self.control.lqr_actuate(&first);
        self.observation.map_object(&controlled)
    }

    pub fn control_control(&self, internal: &InternalModel) -> WorldState {
        let first = self.control.lqr_actuate(internal);
        let observed = self.observation.map_object(&first);
        self.control.lqr_actuate(&observed)
    }

    pub fn verify_left_biduality(&self, world: &WorldState) -> f64 {
        let obs_once = self.observation.map_object(world);
        let ctrl_of_obs = self.control.lqr_actuate(&obs_once);
        let obs_twice = self.observation.map_object(&ctrl_of_obs);
        (obs_twice.belief - &obs_once.belief).norm()
    }

    pub fn verify_right_biduality(&self, internal: &InternalModel) -> f64 {
        let ctrl_once = self.control.lqr_actuate(internal);
        let obs_of_ctrl = self.observation.map_object(&ctrl_once);
        let ctrl_twice = self.control.lqr_actuate(&obs_of_ctrl);
        (ctrl_twice.state - &ctrl_once.state).norm()
    }

    pub fn double_dual_isomorphism(&self, world: &WorldState) -> f64 {
        let internal = self.observation.map_object(world);
        let rt = self.control.lqr_actuate(&internal);
        (rt.state - &world.state).norm()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::MeasurementModel;
    use crate::control::ControlModel;
    use nalgebra::{dmatrix, dvector};

    fn make_biduality() -> Biduality {
        let obs = ObservationFunctor::new(MeasurementModel::identity(2));
        let ctrl = ControlFunctor::new(ControlModel::new(
            dmatrix![1.0, 0.1; 0.0, 1.0],
            dmatrix![0.0; 1.0],
            DMatrix::identity(2, 2),
            dmatrix![0.1],
        ));
        Biduality::new(obs, ctrl)
    }

    #[test]
    fn test_observe_observe() {
        let bd = make_biduality();
        let world = WorldState {
            state: dvector![1.0, 0.5],
            covariance: DMatrix::identity(2, 2),
        };
        let result = bd.observe_observe(&world);
        assert_eq!(result.belief.len(), 2);
    }

    #[test]
    fn test_control_control() {
        let bd = make_biduality();
        let internal = InternalModel {
            belief: dvector![2.0, 1.0],
            uncertainty: DMatrix::identity(2, 2),
        };
        let result = bd.control_control(&internal);
        assert_eq!(result.state.len(), 2);
    }

    #[test]
    fn test_left_biduality() {
        let bd = make_biduality();
        let world = WorldState {
            state: dvector![1.0, 0.0],
            covariance: DMatrix::identity(2, 2) * 0.1,
        };
        let error = bd.verify_left_biduality(&world);
        assert!(error.is_finite());
    }

    #[test]
    fn test_right_biduality() {
        let bd = make_biduality();
        let internal = InternalModel {
            belief: dvector![1.0, 0.0],
            uncertainty: DMatrix::identity(2, 2) * 0.1,
        };
        let error = bd.verify_right_biduality(&internal);
        assert!(error.is_finite());
    }

    #[test]
    fn test_double_dual() {
        let bd = make_biduality();
        let world = WorldState {
            state: dvector![5.0, 1.0],
            covariance: DMatrix::identity(2, 2) * 0.01,
        };
        let error = bd.double_dual_isomorphism(&world);
        assert!(error.is_finite());
    }
}
