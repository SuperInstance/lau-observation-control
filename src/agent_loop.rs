//! 9-step agent loop as trace of identity morphism tr(id_𝒞)

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::observation::{InternalModel, ObservationFunctor, WorldState};
use crate::control::ControlFunctor;
use crate::adjunction::KalmanFilter;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum AgentStep {
    Observe = 0,
    Represent = 1,
    Decompose = 2,
    Optimize = 3,
    Classify = 4,
    Predict = 5,
    Control = 6,
    Adapt = 7,
    Reflect = 8,
}

impl AgentStep {
    pub fn all() -> [AgentStep; 9] {
        [
            AgentStep::Observe, AgentStep::Represent, AgentStep::Decompose,
            AgentStep::Optimize, AgentStep::Classify, AgentStep::Predict,
            AgentStep::Control, AgentStep::Adapt, AgentStep::Reflect,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            AgentStep::Observe => "observe",
            AgentStep::Represent => "represent",
            AgentStep::Decompose => "decompose",
            AgentStep::Optimize => "optimize",
            AgentStep::Classify => "classify",
            AgentStep::Predict => "predict",
            AgentStep::Control => "control",
            AgentStep::Adapt => "adapt",
            AgentStep::Reflect => "reflect",
        }
    }

    pub fn next(&self) -> AgentStep {
        match self {
            AgentStep::Observe => AgentStep::Represent,
            AgentStep::Represent => AgentStep::Decompose,
            AgentStep::Decompose => AgentStep::Optimize,
            AgentStep::Optimize => AgentStep::Classify,
            AgentStep::Classify => AgentStep::Predict,
            AgentStep::Predict => AgentStep::Control,
            AgentStep::Control => AgentStep::Adapt,
            AgentStep::Adapt => AgentStep::Reflect,
            AgentStep::Reflect => AgentStep::Observe,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopState {
    pub step: AgentStep,
    pub world: Option<WorldState>,
    pub internal: Option<InternalModel>,
    pub sub_problems: Vec<DVector<f64>>,
    pub optimal_gain: Option<DMatrix<f64>>,
    pub labels: Vec<usize>,
    pub prediction: Option<DVector<f64>>,
    pub adaptation: Option<DVector<f64>>,
    pub iteration: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentLoop {
    pub observation: ObservationFunctor,
    pub control: ControlFunctor,
    pub kalman: KalmanFilter,
    pub state: LoopState,
}

impl AgentLoop {
    pub fn new(observation: ObservationFunctor, control: ControlFunctor, kalman: KalmanFilter) -> Self {
        Self {
            observation, control, kalman,
            state: LoopState {
                step: AgentStep::Observe,
                world: None, internal: None,
                sub_problems: vec![], optimal_gain: None,
                labels: vec![], prediction: None,
                adaptation: None, iteration: 0,
            },
        }
    }

    pub fn step(&mut self, world_input: Option<&WorldState>) -> AgentStep {
        match self.state.step {
            AgentStep::Observe => {
                if let Some(w) = world_input {
                    self.state.world = Some(w.clone());
                }
                if let Some(ref w) = self.state.world {
                    self.state.internal = Some(self.observation.map_object(w));
                }
            }
            AgentStep::Represent => {
                // Representation is already formed
            }
            AgentStep::Decompose => {
                if let Some(ref internal) = self.state.internal {
                    let n = internal.belief.len();
                    let mut subs = vec![];
                    for i in 0..n {
                        let mut v = DVector::zeros(n);
                        v[i] = internal.belief[i];
                        subs.push(v);
                    }
                    self.state.sub_problems = subs;
                }
            }
            AgentStep::Optimize => {
                self.state.optimal_gain = Some(self.control.lqr_gain());
            }
            AgentStep::Classify => {
                if let Some(ref internal) = self.state.internal {
                    self.state.labels = internal.belief.iter()
                        .map(|v| if v.abs() > 0.5 { 1 } else { 0 })
                        .collect();
                }
            }
            AgentStep::Predict => {
                if let Some(ref internal) = self.state.internal {
                    self.state.prediction = Some(&self.kalman.a * &internal.belief);
                }
            }
            AgentStep::Control => {
                if let Some(ref internal) = self.state.internal {
                    self.state.world = Some(self.control.lqr_actuate(internal));
                }
            }
            AgentStep::Adapt => {
                if let (Some(ref pred), Some(ref internal)) = (&self.state.prediction, &self.state.internal) {
                    let error = (pred - &internal.belief).norm();
                    self.state.adaptation = Some(DVector::from_element(1, error));
                }
            }
            AgentStep::Reflect => {
                self.state.iteration += 1;
                if let Some(ref w) = self.state.world {
                    self.state.internal = Some(self.observation.map_object(w));
                }
            }
        }
        let current = self.state.step;
        self.state.step = current.next();
        current
    }

    pub fn full_cycle(&mut self, world: &WorldState) -> Vec<AgentStep> {
        let mut steps = vec![];
        self.state.world = Some(world.clone());
        // First step: Observe
        steps.push(self.step(Some(world)));
        // Remaining 8 steps don't need new world input
        for _ in 0..8 {
            steps.push(self.step(None));
        }
        steps
    }

    pub fn trace(&mut self, world: &WorldState) -> WorldState {
        self.full_cycle(world);
        self.state.world.clone().unwrap_or_else(|| world.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::MeasurementModel;
    use crate::control::ControlModel;
    use nalgebra::{dmatrix, dvector};

    fn make_agent_loop() -> AgentLoop {
        let n = 2;
        let obs = ObservationFunctor::new(MeasurementModel::identity(n));
        let ctrl = ControlFunctor::new(ControlModel::new(
            dmatrix![1.0, 0.1; 0.0, 1.0],
            dmatrix![0.0; 1.0],
            DMatrix::identity(2, 2),
            dmatrix![0.1],
        ));
        let kf = KalmanFilter::new(
            dmatrix![1.0, 0.1; 0.0, 1.0],
            dmatrix![1.0, 0.0; 0.0, 1.0],
            dmatrix![0.01, 0.0; 0.0, 0.01],
            dmatrix![0.1, 0.0; 0.0, 0.1],
            dvector![0.0, 0.0],
            DMatrix::identity(2, 2),
        );
        AgentLoop::new(obs, ctrl, kf)
    }

    #[test]
    fn test_step_order() {
        assert_eq!(AgentStep::Observe.next(), AgentStep::Represent);
        assert_eq!(AgentStep::Reflect.next(), AgentStep::Observe);
    }

    #[test]
    fn test_all_steps() {
        let all = AgentStep::all();
        assert_eq!(all.len(), 9);
        assert_eq!(all[0], AgentStep::Observe);
        assert_eq!(all[8], AgentStep::Reflect);
    }

    #[test]
    fn test_step_names() {
        assert_eq!(AgentStep::Observe.name(), "observe");
        assert_eq!(AgentStep::Control.name(), "control");
        assert_eq!(AgentStep::Reflect.name(), "reflect");
    }

    #[test]
    fn test_observe_step() {
        let mut al = make_agent_loop();
        let world = WorldState {
            state: dvector![1.0, 0.5],
            covariance: DMatrix::identity(2, 2),
        };
        let step = al.step(Some(&world));
        assert_eq!(step, AgentStep::Observe);
        assert!(al.state.internal.is_some());
    }

    #[test]
    fn test_full_cycle() {
        let mut al = make_agent_loop();
        let world = WorldState {
            state: dvector![3.0, 1.0],
            covariance: DMatrix::identity(2, 2),
        };
        let steps = al.full_cycle(&world);
        assert_eq!(steps.len(), 9);
        assert_eq!(al.state.iteration, 1);
    }

    #[test]
    fn test_trace() {
        let mut al = make_agent_loop();
        let world = WorldState {
            state: dvector![5.0, 2.0],
            covariance: DMatrix::identity(2, 2),
        };
        let result = al.trace(&world);
        assert_eq!(result.state.len(), 2);
    }

    #[test]
    fn test_decompose_step() {
        let mut al = make_agent_loop();
        let world = WorldState {
            state: dvector![3.0, 1.0],
            covariance: DMatrix::identity(2, 2),
        };
        al.step(Some(&world)); // Observe → now at Represent
        al.step(None);          // Represent → now at Decompose
        al.step(None);          // Decompose → now at Optimize
        assert_eq!(al.state.sub_problems.len(), 2);
    }

    #[test]
    fn test_classify_step() {
        let mut al = make_agent_loop();
        let world = WorldState {
            state: dvector![3.0, 0.1],
            covariance: DMatrix::identity(2, 2),
        };
        al.step(Some(&world)); // Observe
        al.step(None); // Represent
        al.step(None); // Decompose
        al.step(None); // Optimize
        al.step(None); // Classify → now at Predict
        assert_eq!(al.state.labels.len(), 2);
        assert_eq!(al.state.labels[0], 1); // 3.0 > 0.5
        assert_eq!(al.state.labels[1], 0); // 0.1 < 0.5
    }

    #[test]
    fn test_predict_step() {
        let mut al = make_agent_loop();
        let world = WorldState {
            state: dvector![3.0, 1.0],
            covariance: DMatrix::identity(2, 2),
        };
        al.step(Some(&world)); // Observe
        al.step(None); // Represent
        al.step(None); // Decompose
        al.step(None); // Optimize
        al.step(None); // Classify
        al.step(None); // Predict → now at Control
        assert!(al.state.prediction.is_some());
    }
}
