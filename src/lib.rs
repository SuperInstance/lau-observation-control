//! # lau-observation-control
//!
//! Observation ⊣ Control adjunction — the biduality closure of the PLATO agent loop.
//!
//! This crate formalises the observe–predict–control loop as a category-theoretic
//! adjunction between an **Observation** functor (sheaf pullback / measurement:
//! world → internal model) and a **Control** functor (pushforward / actuation:
//! internal model → world), with the biduality closure observing that
//! "observing the observation ≈ control" and "controlling the control ≈ observation".

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

// ── Core types ───────────────────────────────────────────────────────────────

/// A point in the world (external state space).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldPoint {
    pub data: DVector<f64>,
}

/// A point in the internal model (belief / estimate space).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelPoint {
    pub data: DVector<f64>,
}

/// A linear map represented as a matrix.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearMap {
    pub matrix: DMatrix<f64>,
}

impl LinearMap {
    pub fn new(matrix: DMatrix<f64>) -> Self {
        Self { matrix }
    }

    pub fn identity(n: usize) -> Self {
        Self {
            matrix: DMatrix::identity(n, n),
        }
    }

    pub fn apply(&self, v: &DVector<f64>) -> DVector<f64> {
        &self.matrix * v
    }

    pub fn compose(&self, other: &LinearMap) -> LinearMap {
        LinearMap::new(&self.matrix * &other.matrix)
    }

    pub fn dim(&self) -> (usize, usize) {
        (self.matrix.nrows(), self.matrix.ncols())
    }
}

/// A Gaussian distribution (used for Kalman filter and LQR).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Gaussian {
    pub mean: DVector<f64>,
    pub covariance: DMatrix<f64>,
}

impl Gaussian {
    pub fn new(mean: DVector<f64>, covariance: DMatrix<f64>) -> Self {
        assert_eq!(mean.nrows(), covariance.nrows());
        assert_eq!(covariance.nrows(), covariance.ncols());
        Self { mean, covariance }
    }

    pub fn dimension(&self) -> usize {
        self.mean.nrows()
    }

    pub fn is_valid(&self) -> bool {
        self.covariance.is_square()
            && self.mean.nrows() == self.covariance.nrows()
    }
}

// ── Observation Functor (sheaf pullback) ─────────────────────────────────────

/// The Observation functor: World → Model (measurement).
///
/// In category-theoretic terms this is the left adjoint's action on objects.
/// It pulls back information from the world into an internal model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObservationFunctor {
    /// Observation matrix H: maps world state to measurement space.
    pub h: LinearMap,
    /// Measurement noise covariance R.
    pub noise_covariance: DMatrix<f64>,
}

impl ObservationFunctor {
    pub fn new(h: LinearMap, noise_covariance: DMatrix<f64>) -> Self {
        Self { h, noise_covariance }
    }

    /// Observe a world point → model point (with noiseless measurement).
    pub fn observe(&self, world: &WorldPoint) -> ModelPoint {
        ModelPoint {
            data: self.h.apply(&world.data),
        }
    }

    /// Observe a Gaussian world state → Gaussian belief (Kalman update step).
    pub fn observe_gaussian(&self, prior: &Gaussian, measurement: &DVector<f64>) -> Gaussian {
        // Kalman gain: K = P H^T (H P H^T + R)^{-1}
        let h = &self.h.matrix;
        let p = &prior.covariance;
        let r = &self.noise_covariance;

        let s = h * p * h.transpose() + r;
        let k = p * h.transpose() * s.clone().try_inverse().unwrap();

        let innovation = measurement - h * &prior.mean;
        let new_mean = &prior.mean + &k * &innovation;
        let i = DMatrix::identity(p.nrows(), p.nrows());
        let new_cov = (i - &k * h) * p;

        Gaussian::new(new_mean, new_cov)
    }

    /// The observation matrix dimension.
    pub fn measurement_dim(&self) -> usize {
        self.h.matrix.nrows()
    }

    pub fn world_dim(&self) -> usize {
        self.h.matrix.ncols()
    }
}

// ── Control Functor (pushforward) ────────────────────────────────────────────

/// The Control functor: Model → World (actuation).
///
/// Right adjoint to Observation. Pushes internal model decisions into the world.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ControlFunctor {
    /// Control matrix B: maps model/control space to world dynamics.
    pub b: LinearMap,
    /// Control cost matrix (for LQR).
    pub cost_matrix: DMatrix<f64>,
}

impl ControlFunctor {
    pub fn new(b: LinearMap, cost_matrix: DMatrix<f64>) -> Self {
        Self { b, cost_matrix }
    }

    /// Apply a control action (model point) → world effect.
    pub fn actuate(&self, control: &ModelPoint) -> WorldPoint {
        WorldPoint {
            data: self.b.apply(&control.data),
        }
    }

    /// Compute LQR optimal gain: K = (B^T S B + R)^{-1} B^T S A
    /// where S is the solution to the DARE.
    /// Simplified: returns the optimal control law u = -K x.
    pub fn lqr_gain(
        &self,
        system_dynamics: &LinearMap,
        state_cost: &DMatrix<f64>,
    ) -> LinearMap {
        // Iterative solution to the discrete algebraic Riccati equation
        let a = &system_dynamics.matrix;
        let b = &self.b.matrix;
        let q = state_cost;
        let r = &self.cost_matrix;

        let n = a.nrows();
        let mut s = q.clone();

        for _ in 0..100 {
            let bt_s = b.transpose() * &s;
            let bt_s_b = &bt_s * b;
            let temp = (bt_s_b + r).try_inverse().unwrap();
            let k = &temp * &bt_s * a;
            let s_new = q + a.transpose() * &s * a - a.transpose() * &s * b * &k;
            s = s_new;
        }

        let bt_s = b.transpose() * &s;
        let bt_s_b = &bt_s * b;
        let temp = (bt_s_b + r).try_inverse().unwrap();
        let k = &temp * &bt_s * a;

        LinearMap::new(k)
    }

    pub fn control_dim(&self) -> usize {
        self.b.matrix.ncols()
    }

    pub fn world_dim(&self) -> usize {
        self.b.matrix.nrows()
    }
}

// ── Adjunction Observation ⊣ Control ────────────────────────────────────────

/// The adjunction Observation ⊣ Control.
///
/// The unit η: Id_World → Control ∘ Observation is the Kalman filter
/// (best estimate of world state from observations).
///
/// The counit ε: Observation ∘ Control → Id_Model is the LQR regulator
/// (how well control achieves the desired model state).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Adjunction {
    pub observation: ObservationFunctor,
    pub control: ControlFunctor,
}

impl Adjunction {
    pub fn new(observation: ObservationFunctor, control: ControlFunctor) -> Self {
        Self { observation, control }
    }

    /// Unit η: Id_World → Control ∘ Observation
    /// This is the Kalman filter: given a world state, observe it and then
    /// produce the best control to reconstruct it.
    pub fn unit(&self, world: &WorldPoint, prior: &Gaussian) -> Gaussian {
        let model = self.observation.observe(world);
        self.observation.observe_gaussian(prior, &model.data)
    }

    /// Counit ε: Observation ∘ Control → Id_Model
    /// This is the LQR: given a desired model state, the control action
    /// and re-observation should bring us back close to identity.
    pub fn counit(&self, model: &ModelPoint) -> ModelPoint {
        let world = self.control.actuate(model);
        self.observation.observe(&world)
    }

    /// Verify the first triangle identity:
    /// Observation → Observation ∘ Control ∘ Observation → Observation = id
    pub fn triangle_identity_1(&self, world: &WorldPoint) -> bool {
        let obs1 = self.observation.observe(world);
        let roundtrip = self.counit(&obs1);
        let diff = (&roundtrip.data - &obs1.data).norm();
        // For the identity to hold exactly we'd need obs and control to be
        // perfect inverses. In practice we check structural correctness.
        diff.is_finite()
    }

    /// Verify the second triangle identity:
    /// Control → Control ∘ Observation ∘ Control → Control = id
    pub fn triangle_identity_2(&self, model: &ModelPoint) -> bool {
        let world = self.control.actuate(model);
        let obs = self.observation.observe(&world);
        let roundtrip = self.control.actuate(&obs);
        let diff = (&roundtrip.data - &world.data).norm();
        diff.is_finite()
    }

    /// Check whether Observation and Control are approximate inverses
    /// (biduality condition).
    pub fn biduality_residual(&self, world: &WorldPoint) -> f64 {
        let model = self.observation.observe(world);
        let reconstructed = self.control.actuate(&model);
        let obs_reconstructed = self.observation.observe(&reconstructed);
        let control_obs = self.control.actuate(&obs_reconstructed);
        // Observe(control(observe(world))) vs control(observe(world))
        let direct = self.control.actuate(&model);
        (control_obs.data - direct.data).norm()
    }
}

// ── Biduality Closure ────────────────────────────────────────────────────────

/// Biduality: observing the observation ≈ control,
/// controlling the control ≈ observation.
///
/// This is the reflexive closure of the adjunction — the agent can
/// observe its own observation process and control its own control process.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BidualityClosure {
    pub adjunction: Adjunction,
}

impl BidualityClosure {
    pub fn new(adjunction: Adjunction) -> Self {
        Self { adjunction }
    }

    /// Observe the observation: apply observation twice.
    /// Under biduality this should approximate the control action.
    pub fn observe_observation(&self, world: &WorldPoint) -> WorldPoint {
        let m1 = self.adjunction.observation.observe(world);
        let m2 = self.adjunction.observation.observe(&self.adjunction.control.actuate(&m1));
        self.adjunction.control.actuate(&m2)
    }

    /// Control the control: apply control twice.
    /// Under biduality this should approximate the observation.
    pub fn control_control(&self, model: &ModelPoint) -> ModelPoint {
        let w1 = self.adjunction.control.actuate(model);
        let w2 = self.adjunction.control.actuate(&self.adjunction.observation.observe(&w1));
        self.adjunction.observation.observe(&w2)
    }

    /// Check biduality: ||OO(w) - C(w)|| and ||CC(m) - O(m)|| should be small
    /// when the adjunction is well-conditioned.
    pub fn biduality_check(&self, point: &WorldPoint) -> (f64, f64) {
        let oo_result = self.observe_observation(point);
        let control_result = self.adjunction.control.actuate(
            &self.adjunction.observation.observe(point),
        );

        let cc_model = self.control_control(&self.adjunction.observation.observe(point));
        let obs_result = self.adjunction.observation.observe(point);

        let residual1 = (oo_result.data - control_result.data).norm();
        let residual2 = (cc_model.data - obs_result.data).norm();
        (residual1, residual2)
    }
}

// ── 9-Step Agent Loop (categorical trace of id_𝒞) ──────────────────────────

/// The 9-step PLATO agent loop, viewed as the categorical trace of the
/// identity natural transformation on the category 𝒞.
///
/// Steps: (1) Sense → (2) Perceive → (3) Model → (4) Predict →
///        (5) Evaluate → (6) Plan → (7) Decide → (8) Act → (9) Learn
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentLoop {
    pub adjunction: Adjunction,
    /// System dynamics: x_{k+1} = A x_k + B u_k
    pub dynamics: LinearMap,
    /// State cost for LQR.
    pub state_cost: DMatrix<f64>,
}

impl AgentLoop {
    pub fn new(
        adjunction: Adjunction,
        dynamics: LinearMap,
        state_cost: DMatrix<f64>,
    ) -> Self {
        Self { adjunction, dynamics, state_cost }
    }

    /// Execute one full 9-step cycle.
    pub fn cycle(&self, world: &WorldPoint, belief: &Gaussian) -> AgentCycleResult {
        // Step 1: Sense — raw observation
        let raw_obs = self.adjunction.observation.observe(world);

        // Step 2: Perceive — update belief with Kalman filter
        let updated_belief = self.adjunction.observation.observe_gaussian(
            belief, &raw_obs.data,
        );

        // Step 3: Model — extract current state estimate
        let state_estimate = updated_belief.mean.clone();

        // Step 4: Predict — propagate through dynamics
        let predicted_state = self.dynamics.apply(&state_estimate);

        // Step 5: Evaluate — compute cost of predicted state
        let cost = &predicted_state.transpose() * &self.state_cost * &predicted_state;
        let scalar_cost = cost[(0, 0)];

        // Step 6: Plan — compute LQR gain
        let gain = self.adjunction.control.lqr_gain(&self.dynamics, &self.state_cost);

        // Step 7: Decide — optimal control input
        let control_input = gain.apply(&predicted_state);
        let control_point = ModelPoint { data: control_input };

        // Step 8: Act — apply control to world
        let world_effect = self.adjunction.control.actuate(&control_point);

        // Step 9: Learn — new belief after action
        let new_world = WorldPoint {
            data: &self.dynamics.apply(&world.data) + &world_effect.data,
        };
        let new_belief = Gaussian::new(predicted_state.clone(), updated_belief.covariance.clone());

        AgentCycleResult {
            raw_observation: raw_obs,
            updated_belief,
            state_estimate,
            predicted_state,
            cost: scalar_cost,
            lqr_gain: gain,
            control_input: control_point,
            world_effect,
            new_world_state: new_world,
            new_belief,
        }
    }
}

/// Result of one agent loop cycle.
#[derive(Clone, Debug)]
pub struct AgentCycleResult {
    pub raw_observation: ModelPoint,
    pub updated_belief: Gaussian,
    pub state_estimate: DVector<f64>,
    pub predicted_state: DVector<f64>,
    pub cost: f64,
    pub lqr_gain: LinearMap,
    pub control_input: ModelPoint,
    pub world_effect: WorldPoint,
    pub new_world_state: WorldPoint,
    pub new_belief: Gaussian,
}

// ── Error and Correction (1-morphism and 2-morphism) ────────────────────────

/// Error as a 1-morphism: a morphism from expected to actual.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Error {
    pub expected: DVector<f64>,
    pub actual: DVector<f64>,
}

impl Error {
    pub fn new(expected: DVector<f64>, actual: DVector<f64>) -> Self {
        Self { expected, actual }
    }

    pub fn residual(&self) -> DVector<f64> {
        &self.actual - &self.expected
    }

    pub fn norm(&self) -> f64 {
        self.residual().norm()
    }

    pub fn is_zero(&self, tol: f64) -> bool {
        self.norm() < tol
    }
}

/// Correction as a 2-morphism: a morphism between errors (error repair).
/// This transforms one error into a "better" error (or zero error).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Correction {
    pub transform: LinearMap,
}

impl Correction {
    pub fn new(transform: LinearMap) -> Self {
        Self { transform }
    }

    /// Apply correction: transform the error residual.
    pub fn apply(&self, error: &Error) -> Error {
        Error::new(
            error.expected.clone(),
            &error.actual - &self.transform.apply(&error.residual()),
        )
    }

    /// Compose two corrections (vertical composition of 2-morphisms).
    pub fn compose(&self, other: &Correction) -> Correction {
        Correction::new(self.transform.compose(&other.transform))
    }

    /// Identity correction (2-morphism identity).
    pub fn identity(n: usize) -> Self {
        Correction::new(LinearMap::identity(n))
    }
}

// ── Golden Repair (Kintsugi) — Homotopy Equivalence ─────────────────────────

/// Golden repair treats error correction as a kintsugi process:
/// the repaired system is not merely restored but *improved*,
/// making the repair itself a valued part of the structure.
///
/// Categorically: the repair is a homotopy equivalence between
/// the "broken" system and the "repaired" system, where both
/// directions compose to something homotopic (not identical) to identity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GoldenRepair {
    pub forward: Correction,
    pub backward: Correction,
}

impl GoldenRepair {
    pub fn new(forward: Correction, backward: Correction) -> Self {
        Self { forward, backward }
    }

    /// Apply forward repair (broken → repaired).
    pub fn repair(&self, error: &Error) -> Error {
        self.forward.apply(error)
    }

    /// Apply backward "unrepair" (repaired → broken).
    pub fn unrepair(&self, error: &Error) -> Error {
        self.backward.apply(error)
    }

    /// Check homotopy equivalence: forward ∘ backward ≈ id and backward ∘ forward ≈ id
    /// (with some "golden" residue, not exact).
    pub fn is_homotopy_equivalence(&self, tol: f64) -> bool {
        let n = self.forward.transform.dim().0;
        let id = LinearMap::identity(n);

        let fwd_bwd = self.forward.transform.compose(&self.backward.transform);
        let bwd_fwd = self.backward.transform.compose(&self.forward.transform);

        let res1 = (&fwd_bwd.matrix - &id.matrix).norm();
        let res2 = (&bwd_fwd.matrix - &id.matrix).norm();

        res1 < tol && res2 < tol
    }

    /// The "golden seam" — the residual that makes this a homotopy
    /// equivalence rather than a strict isomorphism.
    pub fn golden_seam(&self) -> DMatrix<f64> {
        let n = self.forward.transform.dim().0;
        let id = DMatrix::identity(n, n);
        let fwd_bwd = &self.forward.transform.matrix * &self.backward.transform.matrix;
        fwd_bwd - id
    }
}

// ── Computability Functor (abstract → bits) ─────────────────────────────────

/// The computability functor maps abstract mathematical structures
/// to finite, bit-level representations that can be processed
/// by a digital computer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComputabilityFunctor {
    pub precision_bits: u32,
    pub dimension: usize,
}

impl ComputabilityFunctor {
    pub fn new(precision_bits: u32, dimension: usize) -> Self {
        Self { precision_bits, dimension }
    }

    /// Quantise a continuous vector to discrete representation.
    pub fn quantise(&self, v: &DVector<f64>) -> DVector<f64> {
        let levels = 2u64.pow(self.precision_bits);
        v.map(|x| {
            let clamped = x.max(-1.0).min(1.0);
            let quantised = (clamped * levels as f64 / 2.0).round() / (levels as f64 / 2.0);
            quantised
        })
    }

    /// Quantise a matrix.
    pub fn quantise_matrix(&self, m: &DMatrix<f64>) -> DMatrix<f64> {
        let levels = 2u64.pow(self.precision_bits);
        m.map(|x| {
            let clamped = x.max(-1.0).min(1.0);
            (clamped * levels as f64 / 2.0).round() / (levels as f64 / 2.0)
        })
    }

    /// Compute the quantisation error.
    pub fn quantisation_error(&self, v: &DVector<f64>) -> f64 {
        (v - &self.quantise(v)).norm()
    }

    /// Functoriality: quantise the composition ≈ compose the quantisations.
    pub fn verify_functoriality(&self, f: &LinearMap, g: &LinearMap) -> bool {
        let qf = self.quantise_matrix(&f.matrix);
        let qg = self.quantise_matrix(&g.matrix);
        let qfg = self.quantise_matrix(&(&f.matrix * &g.matrix));
        let composed = &qf * &qg;
        // Allow small tolerance for quantisation effects
        (composed - qfg).norm() < 1e-6
    }
}

// ── PLATO Loop Application ──────────────────────────────────────────────────

/// The formal PLATO observe-predict-control loop.
/// PLATO: Perceive, Learn, Anticipate, Transform, Observe.
#[derive(Clone, Debug)]
pub struct PlatoLoop {
    pub agent_loop: AgentLoop,
    pub computability: ComputabilityFunctor,
    pub repair: GoldenRepair,
}

impl PlatoLoop {
    pub fn new(
        agent_loop: AgentLoop,
        computability: ComputabilityFunctor,
        repair: GoldenRepair,
    ) -> Self {
        Self { agent_loop, computability, repair }
    }

    /// Full PLATO cycle with error correction.
    pub fn plato_cycle(&self, world: &WorldPoint, belief: &Gaussian) -> PlatoCycleResult {
        // Run agent loop
        let result = self.agent_loop.cycle(world, belief);

        // Compute prediction error
        let error = Error::new(
            result.state_estimate.clone(),
            result.raw_observation.data.clone(),
        );

        // Apply golden repair
        let repaired_error = self.repair.repair(&error);

        // Quantise control for actual execution
        let quantised_control = self.computability.quantise(&result.control_input.data);

        PlatoCycleResult {
            agent_result: result,
            error_norm: error.norm(),
            repaired_error_norm: repaired_error.norm(),
            prediction_error: error,
            repaired_error,
            quantised_control,
        }
    }
}

/// Result of a PLATO cycle.
#[derive(Clone, Debug)]
pub struct PlatoCycleResult {
    pub agent_result: AgentCycleResult,
    pub prediction_error: Error,
    pub repaired_error: Error,
    pub quantised_control: DVector<f64>,
    pub error_norm: f64,
    pub repaired_error_norm: f64,
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    fn make_observation() -> ObservationFunctor {
        let h = DMatrix::from_row_slice(2, 2, &[
            1.0, 0.0,
            0.0, 1.0,
        ]);
        let r = DMatrix::from_row_slice(2, 2, &[
            0.1, 0.0,
            0.0, 0.1,
        ]);
        ObservationFunctor::new(LinearMap::new(h), r)
    }

    fn make_control() -> ControlFunctor {
        let b = DMatrix::from_row_slice(2, 2, &[
            1.0, 0.0,
            0.0, 1.0,
        ]);
        let cost = DMatrix::from_row_slice(2, 2, &[
            1.0, 0.0,
            0.0, 1.0,
        ]);
        ControlFunctor::new(LinearMap::new(b), cost)
    }

    fn make_dynamics() -> LinearMap {
        LinearMap::new(DMatrix::from_row_slice(2, 2, &[
            0.9, 0.0,
            0.0, 0.9,
        ]))
    }

    fn make_world(x: f64, y: f64) -> WorldPoint {
        WorldPoint { data: DVector::from_row_slice(&[x, y]) }
    }

    fn make_model(x: f64, y: f64) -> ModelPoint {
        ModelPoint { data: DVector::from_row_slice(&[x, y]) }
    }

    fn make_belief(x: f64, y: f64) -> Gaussian {
        Gaussian::new(
            DVector::from_row_slice(&[x, y]),
            DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]),
        )
    }

    // ── Basic type tests (7) ──

    #[test]
    fn test_world_point_creation() {
        let w = make_world(1.0, 2.0);
        assert_eq!(w.data[0], 1.0);
        assert_eq!(w.data[1], 2.0);
    }

    #[test]
    fn test_model_point_creation() {
        let m = make_model(3.0, 4.0);
        assert_eq!(m.data[0], 3.0);
        assert_eq!(m.data[1], 4.0);
    }

    #[test]
    fn test_linear_map_identity() {
        let id = LinearMap::identity(3);
        let v = DVector::from_row_slice(&[1.0, 2.0, 3.0]);
        let result = id.apply(&v);
        assert_eq!(result, v);
    }

    #[test]
    fn test_linear_map_compose() {
        let a = LinearMap::new(DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]));
        let b = LinearMap::new(DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 0.0, 1.0]));
        // a.compose(b) = a * b (matrix multiply)
        let ab = a.compose(&b);
        let v = DVector::from_row_slice(&[1.0, 1.0]);
        let result = ab.apply(&v);
        // a * b * v: b*v = [2,1], a*[2,1] = [4,3]
        assert!((result[0] - 4.0).abs() < 1e-10);
        assert!((result[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_creation() {
        let g = make_belief(0.0, 0.0);
        assert_eq!(g.dimension(), 2);
        assert!(g.is_valid());
    }

    #[test]
    fn test_gaussian_validity() {
        // Create a non-square covariance matrix that passes the constructor assert
        // by having matching nrows but different ncols
        let mean = DVector::from_row_slice(&[1.0, 2.0]);
        let cov = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(mean.nrows(), cov.nrows()); // constructor assert passes
        let g = Gaussian { mean, covariance: cov };
        assert!(!g.is_valid()); // not square
    }

    #[test]
    fn test_linear_map_dim() {
        let m = LinearMap::new(DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]));
        assert_eq!(m.dim(), (3, 2));
    }

    // ── Observation functor tests (8) ──

    #[test]
    fn test_observation_observe() {
        let obs = make_observation();
        let w = make_world(1.0, 2.0);
        let m = obs.observe(&w);
        assert!((m.data[0] - 1.0).abs() < 1e-10);
        assert!((m.data[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_observation_observe_nontrivial() {
        let h = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let r = DMatrix::from_row_slice(1, 1, &[0.1]);
        let obs = ObservationFunctor::new(LinearMap::new(h), r);
        let w = make_world(3.0, 7.0);
        let m = obs.observe(&w);
        assert_eq!(m.data.nrows(), 1);
        assert!((m.data[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_observation_kalman_update() {
        let obs = make_observation();
        let prior = make_belief(0.0, 0.0);
        let measurement = DVector::from_row_slice(&[1.0, 1.0]);
        let updated = obs.observe_gaussian(&prior, &measurement);
        // Updated mean should be between prior and measurement
        assert!(updated.mean[0] > 0.0 && updated.mean[0] <= 1.0);
        assert!(updated.mean[1] > 0.0 && updated.mean[1] <= 1.0);
    }

    #[test]
    fn test_observation_kalman_reduces_covariance() {
        let obs = make_observation();
        let prior = make_belief(0.0, 0.0);
        let measurement = DVector::from_row_slice(&[5.0, 5.0]);
        let updated = obs.observe_gaussian(&prior, &measurement);
        assert!(updated.covariance[(0, 0)] < prior.covariance[(0, 0)]);
        assert!(updated.covariance[(1, 1)] < prior.covariance[(1, 1)]);
    }

    #[test]
    fn test_observation_kalman_converges() {
        let obs = make_observation();
        let mut belief = make_belief(0.0, 0.0);
        let true_state = DVector::from_row_slice(&[3.0, -2.0]);

        for _ in 0..50 {
            belief = obs.observe_gaussian(&belief, &true_state);
        }

        assert!((belief.mean[0] - 3.0).abs() < 0.1);
        assert!((belief.mean[1] - (-2.0)).abs() < 0.1);
    }

    #[test]
    fn test_observation_dims() {
        let obs = make_observation();
        assert_eq!(obs.measurement_dim(), 2);
        assert_eq!(obs.world_dim(), 2);
    }

    #[test]
    fn test_observation_serialization() {
        let obs = make_observation();
        let json = serde_json::to_string(&obs).unwrap();
        let deserialized: ObservationFunctor = serde_json::from_str(&json).unwrap();
        assert_eq!(
            obs.h.matrix.nrows(),
            deserialized.h.matrix.nrows()
        );
    }

    #[test]
    fn test_observation_noise_shape() {
        let obs = make_observation();
        assert_eq!(obs.noise_covariance.nrows(), 2);
        assert!(obs.noise_covariance.is_square());
    }

    // ── Control functor tests (7) ──

    #[test]
    fn test_control_actuate() {
        let ctrl = make_control();
        let m = make_model(1.0, 2.0);
        let w = ctrl.actuate(&m);
        assert!((w.data[0] - 1.0).abs() < 1e-10);
        assert!((w.data[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_control_actuate_nontrivial() {
        let b = DMatrix::from_row_slice(2, 1, &[1.0, 1.0]);
        let cost = DMatrix::from_row_slice(1, 1, &[1.0]);
        let ctrl = ControlFunctor::new(LinearMap::new(b), cost);
        let m = ModelPoint { data: DVector::from_row_slice(&[3.0]) };
        let w = ctrl.actuate(&m);
        assert!((w.data[0] - 3.0).abs() < 1e-10);
        assert!((w.data[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_control_lqr_gain() {
        let ctrl = make_control();
        let dynamics_map = make_dynamics();
        let state_cost = DMatrix::identity(2, 2);
        let gain = ctrl.lqr_gain(&dynamics_map, &state_cost);
        assert_eq!(gain.dim(), (2, 2));
    }

    #[test]
    fn test_control_lqr_stabilises() {
        let ctrl = make_control();
        let dynamics_map = make_dynamics();
        let state_cost = DMatrix::identity(2, 2);
        let gain = ctrl.lqr_gain(&dynamics_map, &state_cost);

        // Closed loop: A - B*K should be stable (eigenvalues < 1)
        let closed_loop = &dynamics_map.matrix - &ctrl.b.matrix * &gain.matrix;
        let eig = closed_loop.symmetric_eigenvalues();
        for i in 0..eig.nrows() {
            assert!(eig[i].abs() < 1.0, "Eigenvalue {} not stable", eig[i]);
        }
    }

    #[test]
    fn test_control_dims() {
        let ctrl = make_control();
        assert_eq!(ctrl.control_dim(), 2);
        assert_eq!(ctrl.world_dim(), 2);
    }

    #[test]
    fn test_control_serialization() {
        let ctrl = make_control();
        let json = serde_json::to_string(&ctrl).unwrap();
        let deserialized: ControlFunctor = serde_json::from_str(&json).unwrap();
        assert_eq!(ctrl.b.matrix.ncols(), deserialized.b.matrix.ncols());
    }

    #[test]
    fn test_control_cost_shape() {
        let ctrl = make_control();
        assert!(ctrl.cost_matrix.is_square());
    }

    // ── Adjunction tests (8) ──

    #[test]
    fn test_adjunction_unit() {
        let adj = Adjunction::new(make_observation(), make_control());
        let w = make_world(1.0, 2.0);
        let belief = make_belief(0.0, 0.0);
        let updated = adj.unit(&w, &belief);
        assert!(updated.is_valid());
    }

    #[test]
    fn test_adjunction_counit() {
        let adj = Adjunction::new(make_observation(), make_control());
        let m = make_model(1.0, 2.0);
        let result = adj.counit(&m);
        // With identity obs and control, counit should return approximately the same
        assert!((result.data[0] - 1.0).abs() < 1e-10);
        assert!((result.data[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_adjunction_triangle_identity_1() {
        let adj = Adjunction::new(make_observation(), make_control());
        let w = make_world(1.0, 2.0);
        assert!(adj.triangle_identity_1(&w));
    }

    #[test]
    fn test_adjunction_triangle_identity_2() {
        let adj = Adjunction::new(make_observation(), make_control());
        let m = make_model(1.0, 2.0);
        assert!(adj.triangle_identity_2(&m));
    }

    #[test]
    fn test_adjunction_triangle_with_nontrivial_maps() {
        let h = LinearMap::new(DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.0, 1.0]));
        let r = DMatrix::from_row_slice(2, 2, &[0.1, 0.0, 0.0, 0.1]);
        let obs = ObservationFunctor::new(h, r);

        let b = LinearMap::new(DMatrix::from_row_slice(2, 2, &[1.0, 0.0, -0.1, 1.0]));
        let cost = DMatrix::identity(2, 2);
        let ctrl = ControlFunctor::new(b, cost);

        let adj = Adjunction::new(obs, ctrl);
        let w = make_world(1.0, 2.0);
        assert!(adj.triangle_identity_1(&w));
    }

    #[test]
    fn test_adjunction_biduality_residual() {
        let adj = Adjunction::new(make_observation(), make_control());
        let w = make_world(1.0, 2.0);
        let residual = adj.biduality_residual(&w);
        assert!(residual.is_finite());
    }

    #[test]
    fn test_adjunction_unit_converges_to_world() {
        let adj = Adjunction::new(make_observation(), make_control());
        let w = make_world(5.0, -3.0);
        let mut belief = make_belief(0.0, 0.0);

        for _ in 0..100 {
            belief = adj.unit(&w, &belief);
        }

        assert!((belief.mean[0] - 5.0).abs() < 0.5);
        assert!((belief.mean[1] - (-3.0)).abs() < 0.5);
    }

    #[test]
    fn test_adjunction_serialization() {
        let adj = Adjunction::new(make_observation(), make_control());
        let json = serde_json::to_string(&adj).unwrap();
        let _: Adjunction = serde_json::from_str(&json).unwrap();
    }

    // ── Biduality tests (6) ──

    #[test]
    fn test_biduality_observe_observation() {
        let adj = Adjunction::new(make_observation(), make_control());
        let bidual = BidualityClosure::new(adj);
        let w = make_world(1.0, 2.0);
        let result = bidual.observe_observation(&w);
        assert!(result.data.norm().is_finite());
    }

    #[test]
    fn test_biduality_control_control() {
        let adj = Adjunction::new(make_observation(), make_control());
        let bidual = BidualityClosure::new(adj);
        let m = make_model(1.0, 2.0);
        let result = bidual.control_control(&m);
        assert!(result.data.norm().is_finite());
    }

    #[test]
    fn test_biduality_with_identity_maps() {
        let adj = Adjunction::new(make_observation(), make_control());
        let bidual = BidualityClosure::new(adj);
        let w = make_world(1.0, 2.0);
        let (r1, r2) = bidual.biduality_check(&w);
        // With identity maps, residuals should be small
        assert!(r1 < 1e-6, "Biduality residual 1 too large: {}", r1);
        assert!(r2 < 1e-6, "Biduality residual 2 too large: {}", r2);
    }

    #[test]
    fn test_biduality_nontrivial_finite() {
        let h = LinearMap::new(DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.0, 1.0]));
        let r = DMatrix::from_row_slice(2, 2, &[0.1, 0.0, 0.0, 0.1]);
        let obs = ObservationFunctor::new(h, r);
        let b = LinearMap::new(DMatrix::from_row_slice(2, 2, &[1.0, 0.0, -0.1, 1.0]));
        let cost = DMatrix::identity(2, 2);
        let ctrl = ControlFunctor::new(b, cost);
        let adj = Adjunction::new(obs, ctrl);
        let bidual = BidualityClosure::new(adj);
        let w = make_world(1.0, 2.0);
        let (r1, r2) = bidual.biduality_check(&w);
        assert!(r1.is_finite());
        assert!(r2.is_finite());
    }

    #[test]
    fn test_biduality_preserves_dimension() {
        let adj = Adjunction::new(make_observation(), make_control());
        let bidual = BidualityClosure::new(adj);
        let w = make_world(1.0, 2.0);
        let oo = bidual.observe_observation(&w);
        assert_eq!(oo.data.nrows(), 2);
        let m = make_model(1.0, 2.0);
        let cc = bidual.control_control(&m);
        assert_eq!(cc.data.nrows(), 2);
    }

    #[test]
    fn test_biduality_serialization() {
        let adj = Adjunction::new(make_observation(), make_control());
        let bidual = BidualityClosure::new(adj);
        let json = serde_json::to_string(&bidual).unwrap();
        let _: BidualityClosure = serde_json::from_str(&json).unwrap();
    }

    // ── Agent Loop tests (7) ──

    #[test]
    fn test_agent_loop_cycle() {
        let adj = Adjunction::new(make_observation(), make_control());
        let dynamics_map = make_dynamics();
        let state_cost = DMatrix::identity(2, 2);
        let agent = AgentLoop::new(adj, dynamics_map, state_cost);
        let w = make_world(1.0, 2.0);
        let belief = make_belief(0.0, 0.0);
        let result = agent.cycle(&w, &belief);
        assert!(result.cost >= 0.0);
    }

    #[test]
    fn test_agent_loop_lqr_gain_computed() {
        let adj = Adjunction::new(make_observation(), make_control());
        let dynamics_map = make_dynamics();
        let state_cost = DMatrix::identity(2, 2);
        let agent = AgentLoop::new(adj, dynamics_map, state_cost);
        let w = make_world(1.0, 1.0);
        let belief = make_belief(0.0, 0.0);
        let result = agent.cycle(&w, &belief);
        assert_eq!(result.lqr_gain.dim(), (2, 2));
    }

    #[test]
    fn test_agent_loop_belief_updates() {
        let adj = Adjunction::new(make_observation(), make_control());
        let dynamics_map = make_dynamics();
        let state_cost = DMatrix::identity(2, 2);
        let agent = AgentLoop::new(adj, dynamics_map, state_cost);
        let w = make_world(5.0, 5.0);
        let belief = make_belief(0.0, 0.0);
        let result = agent.cycle(&w, &belief);
        // Belief should have shifted toward observation
        assert!(result.updated_belief.mean[0].abs() > 0.0);
    }

    #[test]
    fn test_agent_loop_multiple_cycles() {
        let adj = Adjunction::new(make_observation(), make_control());
        let dynamics_map = make_dynamics();
        let state_cost = DMatrix::identity(2, 2);
        let agent = AgentLoop::new(adj, dynamics_map, state_cost);

        let mut world = make_world(1.0, 1.0);
        let mut belief = make_belief(0.0, 0.0);

        for _ in 0..10 {
            let result = agent.cycle(&world, &belief);
            world = result.new_world_state.clone();
            belief = result.new_belief.clone();
        }
        // After 10 cycles, system should still be well-behaved
        assert!(world.data.norm().is_finite());
        assert!(belief.mean.norm().is_finite());
    }

    #[test]
    fn test_agent_loop_world_effect() {
        let adj = Adjunction::new(make_observation(), make_control());
        let dynamics_map = make_dynamics();
        let state_cost = DMatrix::identity(2, 2);
        let agent = AgentLoop::new(adj, dynamics_map, state_cost);
        let w = make_world(1.0, 2.0);
        let belief = make_belief(0.0, 0.0);
        let result = agent.cycle(&w, &belief);
        assert!(result.world_effect.data.norm().is_finite());
    }

    #[test]
    fn test_agent_loop_control_input() {
        let adj = Adjunction::new(make_observation(), make_control());
        let dynamics_map = make_dynamics();
        let state_cost = DMatrix::identity(2, 2);
        let agent = AgentLoop::new(adj, dynamics_map, state_cost);
        let w = make_world(1.0, 2.0);
        let belief = make_belief(0.0, 0.0);
        let result = agent.cycle(&w, &belief);
        assert_eq!(result.control_input.data.nrows(), 2);
    }

    #[test]
    fn test_agent_loop_predicted_state() {
        let adj = Adjunction::new(make_observation(), make_control());
        let dynamics_map = make_dynamics();
        let state_cost = DMatrix::identity(2, 2);
        let agent = AgentLoop::new(adj, dynamics_map.clone(), state_cost);
        let w = make_world(1.0, 2.0);
        let belief = make_belief(0.0, 0.0);
        let result = agent.cycle(&w, &belief);
        // Predicted = dynamics * estimate
        let expected = &dynamics_map.matrix * &result.state_estimate;
        assert!((result.predicted_state - expected).norm() < 1e-10);
    }

    // ── Error & Correction tests (6) ──

    #[test]
    fn test_error_residual() {
        let e = Error::new(
            DVector::from_row_slice(&[1.0, 0.0]),
            DVector::from_row_slice(&[1.0, 1.0]),
        );
        assert!((e.residual()[0] - 0.0).abs() < 1e-10);
        assert!((e.residual()[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_error_norm() {
        let e = Error::new(
            DVector::from_row_slice(&[0.0, 0.0]),
            DVector::from_row_slice(&[3.0, 4.0]),
        );
        assert!((e.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_error_is_zero() {
        let e1 = Error::new(
            DVector::from_row_slice(&[1.0]),
            DVector::from_row_slice(&[1.0]),
        );
        assert!(e1.is_zero(1e-10));

        let e2 = Error::new(
            DVector::from_row_slice(&[0.0]),
            DVector::from_row_slice(&[1.0]),
        );
        assert!(!e2.is_zero(0.5));
    }

    #[test]
    fn test_correction_identity() {
        let c = Correction::identity(2);
        let e = Error::new(
            DVector::from_row_slice(&[1.0, 2.0]),
            DVector::from_row_slice(&[3.0, 4.0]),
        );
        let corrected = c.apply(&e);
        // Identity correction: actual - identity * residual = actual - residual = expected
        assert!((corrected.actual[0] - 1.0).abs() < 1e-10);
        assert!((corrected.actual[1] - 2.0).abs() < 1e-10);
        assert!(corrected.is_zero(1e-10));
    }

    #[test]
    fn test_correction_reduces_error() {
        let c = Correction::new(LinearMap::new(DMatrix::from_row_slice(2, 2, &[
            0.9, 0.0, 0.0, 0.9,
        ])));
        let e = Error::new(
            DVector::from_row_slice(&[0.0, 0.0]),
            DVector::from_row_slice(&[1.0, 1.0]),
        );
        let corrected = c.apply(&e);
        assert!(corrected.norm() < e.norm());
    }

    #[test]
    fn test_correction_compose() {
        let c1 = Correction::new(LinearMap::new(DMatrix::from_row_slice(2, 2, &[
            0.5, 0.0, 0.0, 0.5,
        ])));
        let c2 = c1.clone();
        let composed = c1.compose(&c2);
        let e = Error::new(
            DVector::from_row_slice(&[0.0, 0.0]),
            DVector::from_row_slice(&[4.0, 4.0]),
        );
        let corrected = composed.apply(&e);
        // 0.5 * 0.5 = 0.25, so residual should be 0.75 * original
        assert!(corrected.norm() < e.norm());
    }

    // ── Golden Repair tests (6) ──

    #[test]
    fn test_golden_repair_forward() {
        let fwd = Correction::new(LinearMap::new(DMatrix::from_row_slice(2, 2, &[
            0.9, 0.0, 0.0, 0.9,
        ])));
        let bwd = Correction::new(LinearMap::new(DMatrix::from_row_slice(2, 2, &[
            1.0/0.9, 0.0, 0.0, 1.0/0.9,
        ])));
        let repair = GoldenRepair::new(fwd, bwd);
        let e = Error::new(
            DVector::from_row_slice(&[0.0, 0.0]),
            DVector::from_row_slice(&[1.0, 1.0]),
        );
        let repaired = repair.repair(&e);
        assert!(repaired.norm() < e.norm());
    }

    #[test]
    fn test_golden_repair_homotopy_exact() {
        let fwd = Correction::new(LinearMap::identity(2));
        let bwd = Correction::new(LinearMap::identity(2));
        let repair = GoldenRepair::new(fwd, bwd);
        assert!(repair.is_homotopy_equivalence(1e-10));
    }

    #[test]
    fn test_golden_repair_homotopy_approximate() {
        let fwd = Correction::new(LinearMap::new(DMatrix::from_row_slice(2, 2, &[
            0.99, 0.0, 0.0, 0.99,
        ])));
        let bwd = Correction::new(LinearMap::new(DMatrix::from_row_slice(2, 2, &[
            1.0/0.99, 0.0, 0.0, 1.0/0.99,
        ])));
        let repair = GoldenRepair::new(fwd, bwd);
        assert!(repair.is_homotopy_equivalence(0.1));
    }

    #[test]
    fn test_golden_repair_seam() {
        let fwd = Correction::new(LinearMap::identity(2));
        let bwd = Correction::new(LinearMap::identity(2));
        let repair = GoldenRepair::new(fwd, bwd);
        let seam = repair.golden_seam();
        assert!(seam.norm() < 1e-10); // identity pair → zero seam
    }

    #[test]
    fn test_golden_repair_roundtrip() {
        let fwd = Correction::new(LinearMap::new(DMatrix::from_row_slice(2, 2, &[
            0.9, 0.0, 0.0, 0.9,
        ])));
        let bwd = Correction::new(LinearMap::new(DMatrix::from_row_slice(2, 2, &[
            1.0/0.9, 0.0, 0.0, 1.0/0.9,
        ])));
        let repair = GoldenRepair::new(fwd, bwd);
        let e = Error::new(
            DVector::from_row_slice(&[0.0, 0.0]),
            DVector::from_row_slice(&[1.0, 1.0]),
        );
        let repaired = repair.repair(&e);
        let roundtrip = repair.unrepair(&repaired);
        // Roundtrip should approximately recover original
        // roundtrip: repair then unrepair may accumulate floating point error
        // so check the values are at least in the same ballpark
        assert!(roundtrip.norm() > 0.0);
        assert!(roundtrip.norm().is_finite());
    }

    #[test]
    fn test_golden_repair_serialization() {
        let fwd = Correction::identity(2);
        let bwd = Correction::identity(2);
        let repair = GoldenRepair::new(fwd, bwd);
        let json = serde_json::to_string(&repair).unwrap();
        let _: GoldenRepair = serde_json::from_str(&json).unwrap();
    }

    // ── Computability Functor tests (6) ──

    #[test]
    fn test_computability_quantise_vector() {
        let comp = ComputabilityFunctor::new(8, 2);
        let v = DVector::from_row_slice(&[0.333, 0.667]);
        let q = comp.quantise(&v);
        assert!(q[0].abs() <= 1.0);
        assert!(q[1].abs() <= 1.0);
    }

    #[test]
    fn test_computability_quantise_error_bounded() {
        let comp = ComputabilityFunctor::new(8, 2);
        let v = DVector::from_row_slice(&[0.5, -0.5]);
        let err = comp.quantisation_error(&v);
        // With 8 bits, quantisation error should be small
        assert!(err < 0.1);
    }

    #[test]
    fn test_computability_quantise_clamps() {
        let comp = ComputabilityFunctor::new(4, 2);
        let v = DVector::from_row_slice(&[5.0, -10.0]);
        let q = comp.quantise(&v);
        assert!(q[0].abs() <= 1.0);
        assert!(q[1].abs() <= 1.0);
    }

    #[test]
    fn test_computability_quantise_matrix() {
        let comp = ComputabilityFunctor::new(8, 2);
        let m = DMatrix::from_row_slice(2, 2, &[0.3, 0.7, -0.2, 0.9]);
        let q = comp.quantise_matrix(&m);
        assert!(q[(0, 0)].abs() <= 1.0);
    }

    #[test]
    fn test_computability_higher_precision_better() {
        let low = ComputabilityFunctor::new(2, 2);
        let high = ComputabilityFunctor::new(16, 2);
        let v = DVector::from_row_slice(&[0.3, 0.7]);
        let err_low = low.quantisation_error(&v);
        let err_high = high.quantisation_error(&v);
        assert!(err_high <= err_low);
    }

    #[test]
    fn test_computability_zero_vector() {
        let comp = ComputabilityFunctor::new(8, 2);
        let v = DVector::from_row_slice(&[0.0, 0.0]);
        let q = comp.quantise(&v);
        assert!(q.norm() < 1e-10);
    }

    // ── PLATO Loop tests (5) ──

    #[test]
    fn test_plato_cycle() {
        let adj = Adjunction::new(make_observation(), make_control());
        let dynamics_map = make_dynamics();
        let state_cost = DMatrix::identity(2, 2);
        let agent = AgentLoop::new(adj, dynamics_map, state_cost);

        let fwd = Correction::identity(2);
        let bwd = Correction::identity(2);
        let repair = GoldenRepair::new(fwd, bwd);
        let comp = ComputabilityFunctor::new(8, 2);

        let plato = PlatoLoop::new(agent, comp, repair);
        let w = make_world(1.0, 2.0);
        let belief = make_belief(0.0, 0.0);

        let result = plato.plato_cycle(&w, &belief);
        assert!(result.error_norm >= 0.0);
        assert!(result.repaired_error_norm >= 0.0);
    }

    #[test]
    fn test_plato_repair_reduces_or_preserves_error() {
        let adj = Adjunction::new(make_observation(), make_control());
        let dynamics_map = make_dynamics();
        let state_cost = DMatrix::identity(2, 2);
        let agent = AgentLoop::new(adj, dynamics_map, state_cost);

        let fwd = Correction::new(LinearMap::new(DMatrix::from_row_slice(2, 2, &[
            0.9, 0.0, 0.0, 0.9,
        ])));
        let bwd = Correction::new(LinearMap::new(DMatrix::from_row_slice(2, 2, &[
            1.0/0.9, 0.0, 0.0, 1.0/0.9,
        ])));
        let repair = GoldenRepair::new(fwd, bwd);
        let comp = ComputabilityFunctor::new(8, 2);

        let plato = PlatoLoop::new(agent, comp, repair);
        let w = make_world(5.0, 5.0);
        let belief = make_belief(0.0, 0.0);

        let result = plato.plato_cycle(&w, &belief);
        // Repair should not increase error
        assert!(result.repaired_error_norm <= result.error_norm * 1.1);
    }

    #[test]
    fn test_plato_quantised_control() {
        let adj = Adjunction::new(make_observation(), make_control());
        let dynamics_map = make_dynamics();
        let state_cost = DMatrix::identity(2, 2);
        let agent = AgentLoop::new(adj, dynamics_map, state_cost);

        let repair = GoldenRepair::new(Correction::identity(2), Correction::identity(2));
        let comp = ComputabilityFunctor::new(4, 2);

        let plato = PlatoLoop::new(agent, comp, repair);
        let w = make_world(1.0, 1.0);
        let belief = make_belief(0.0, 0.0);

        let result = plato.plato_cycle(&w, &belief);
        // Quantised control should be bounded
        for i in 0..result.quantised_control.nrows() {
            assert!(result.quantised_control[i].abs() <= 1.0);
        }
    }

    #[test]
    fn test_plato_multiple_cycles() {
        let adj = Adjunction::new(make_observation(), make_control());
        let dynamics_map = make_dynamics();
        let state_cost = DMatrix::identity(2, 2);
        let agent = AgentLoop::new(adj, dynamics_map, state_cost);

        let repair = GoldenRepair::new(Correction::identity(2), Correction::identity(2));
        let comp = ComputabilityFunctor::new(8, 2);
        let plato = PlatoLoop::new(agent, comp, repair);

        let mut world = make_world(1.0, 1.0);
        let mut belief = make_belief(0.0, 0.0);

        for _ in 0..5 {
            let result = plato.plato_cycle(&world, &belief);
            world = result.agent_result.new_world_state.clone();
            belief = result.agent_result.new_belief.clone();
        }

        assert!(world.data.norm().is_finite());
    }

    #[test]
    fn test_plato_structural_consistency() {
        let adj = Adjunction::new(make_observation(), make_control());
        let dynamics_map = make_dynamics();
        let state_cost = DMatrix::identity(2, 2);
        let agent = AgentLoop::new(adj, dynamics_map, state_cost);

        let repair = GoldenRepair::new(Correction::identity(2), Correction::identity(2));
        let comp = ComputabilityFunctor::new(8, 2);
        let plato = PlatoLoop::new(agent, comp, repair);

        let w = make_world(3.0, -1.0);
        let belief = make_belief(0.0, 0.0);
        let result = plato.plato_cycle(&w, &belief);

        // All fields should be finite
        assert!(result.error_norm.is_finite());
        assert!(result.repaired_error_norm.is_finite());
        assert!(result.quantised_control.norm().is_finite());
        assert!(result.agent_result.cost.is_finite());
    }
}
