# lau-observation-control

> Observation ⊣ Control adjunction — the biduality closure of the PLATO agent loop.

[![tests](https://img.shields.io/badge/tests-133-green)]()
[![license](https://img.shields.io/badge/license-MIT-blue)]()

## What This Does

This crate formalizes the **observe–predict–control loop** as a category-theoretic **adjunction** between two functors:

- **Observation functor** (left adjoint): sheaf pullback / measurement — maps world state to internal model
- **Control functor** (right adjoint): pushforward / actuation — maps internal model back to world

The key insight is **biduality**: "observing the observation ≈ control" and "controlling the control ≈ observation". The unit of the adjunction is the **Kalman filter**, the counit is **LQR optimal control**, and the triangle identities are verified computationally.

Built on `nalgebra` for real linear algebra, with full `serde` serialization.

## The Key Idea

In category theory, an **adjunction** `F ⊣ G` between functors `F: C → D` and `G: D → C` means there's a natural bijection:

```
Hom_D(F(X), Y) ≅ Hom_C(X, G(Y))
```

This crate instantiates this with:

| Categorical concept | Implementation |
|---------------------|----------------|
| Category **C** | World states (external reality) |
| Category **D** | Internal models (agent beliefs) |
| **F** (Observation) | Kalman measurement: `y = Hx + noise` |
| **G** (Control) | LQR actuation: `u = -Kx̂` |
| **Unit** η: id → GF | Kalman filter (world → observe → control → world) |
| **Counit** ε: FG → id | LQR controller (model → control → observe → model) |
| **Triangle identities** | Verified computationally |

The **9-step agent loop** (Observe → Represent → Decompose → Optimize → Classify → Predict → Control → Adapt → Reflect) is modeled as the **trace of the identity morphism** `tr(id_C)` — a categorical trace that completes one full cycle through the adjunction.

Beyond the core adjunction, the crate also implements:
- **Computability functor** `⌊−⌋: C → FinSet` — grounding continuous objects in finite bit representations
- **Error-correcting codes** — errors as 1-morphisms, corrections as 2-morphisms in a 2-category
- **Golden repair** (kintsugi) — the homotopy equivalence between error and correction, parameterized by the golden ratio φ

## Install

```bash
cargo add lau-observation-control
```

## Quick Start

```rust
use lau_observation_control::*;
use nalgebra::{dmatrix, dvector};

// Build a PLATO system: 2D state, 1D control
let mut plato = plato::PlatoSystem::new(
    2, 1, 2,
    dmatrix![1.0, 0.1; 0.0, 1.0],  // dynamics A
    dmatrix![0.0; 1.0],              // control B
);

// Run the observe-predict-control cycle
let world = observation::WorldState {
    state: dvector![5.0, 2.0],
    covariance: DMatrix::identity(2, 2),
};
let (observed, predicted, controlled) = plato.cycle(&world);

// The Kalman filter is the unit of the adjunction
let adj = adjunction::Adjunction::new(
    dmatrix![1.0, 0.1; 0.0, 1.0],   // A
    dmatrix![1.0, 0.0; 0.0, 1.0],   // H
    dmatrix![0.01, 0.0; 0.0, 0.01], // Q
    dmatrix![0.1, 0.0; 0.0, 0.1],   // R
    dmatrix![0.0; 1.0],              // B
    DMatrix::identity(2, 2),         // cost Q
    dmatrix![0.1, 0.0; 0.0, 0.1],   // cost R
);

// Verify the triangle identities
assert!(adj.triangle_identity_1(&world));  // εF ∘ Fη = id_F
assert!(adj.triangle_identity_2(&internal)); // Gε ∘ ηG = id_G
```

## API Reference

### Core Types (lib.rs)

| Type | Description |
|------|-------------|
| `WorldPoint` | A point in external state space |
| `ModelPoint` | A point in internal belief space |
| `LinearMap` | Matrix-represented linear transformation |
| `Gaussian` | Multivariate Gaussian distribution |

### observation

World → Internal model (the left adjoint).

| Type | Description |
|------|-------------|
| `WorldState` | External state vector + covariance |
| `InternalModel` | Belief vector + uncertainty |
| `MeasurementModel` | Observation matrix `H` + noise `R` |
| `ObservationFunctor` | Maps world states to internal models |

Key methods: `map_object`, `map_morphism`, `compose_morphisms`

### control

Internal model → World (the right adjoint).

| Type | Description |
|------|-------------|
| `ControlModel` | State transition `A`, input `B`, costs `Q`, `R` |
| `ControlFunctor` | Maps internal models to world states |
| `LQRController` | Optimal controller (counit of adjunction) |

Key methods: `lqr_gain` (solves DARE), `lqr_actuate`, `compute_control`, `step`

### adjunction

The full Observation ⊣ Control adjunction.

| Type | Description |
|------|-------------|
| `KalmanFilter` | Unit of the adjunction (predict + update) |
| `Adjunction` | Full adjunction with triangle identities |

Key methods:
- `KalmanFilter::predict` / `update` / `step` / `gain`
- `Adjunction::unit` / `counit` / `triangle_identity_1` / `triangle_identity_2`
- `roundtrip_observe` / `roundtrip_control`

### biduality

Verifying that the double-dual closes.

| Type | Description |
|------|-------------|
| `Biduality` | Observation-Control biduality checker |

Key methods: `observe_observe`, `control_control`, `verify_left_biduality`, `verify_right_biduality`, `double_dual_isomorphism`

### agent_loop

The 9-step categorical agent loop.

| Step | Operation |
|------|-----------|
| 0. Observe | Apply observation functor to world |
| 1. Represent | Form internal representation |
| 2. Decompose | Split into per-dimension sub-problems |
| 3. Optimize | Compute LQR gain |
| 4. Classify | Binary classify each dimension |
| 5. Predict | Kalman predict step |
| 6. Control | Apply LQR actuation |
| 7. Adapt | Measure prediction error |
| 8. Reflect | Re-observe and increment iteration |

Key methods: `AgentLoop::step`, `full_cycle`, `trace`

### plato

Complete PLATO observe-predict-control system.

| Type | Description |
|------|-------------|
| `PlatoConfig` | Dimensions and bit resolution |
| `PlatoSystem` | Full integrated system |

Key methods: `observe`, `predict`, `control`, `cycle`, `compute_error`, `apply_correction`, `ground`, `run`

### computability

Grounding functor `⌊−⌋: C → FinSet`.

| Type | Description |
|------|-------------|
| `FinSet` | Finite bit-vector representation |
| `ComputabilityConfig` | Bits per dimension, scale, offset |
| `ComputabilityFunctor` | Encode real vectors to bits and back |

Key methods: `encode_value`, `decode_value`, `map_object`, `decode_object`, `quantization_error`

### error_correction

Errors as 1-morphisms, corrections as 2-morphisms.

| Type | Description |
|------|-------------|
| `Error` | 1-morphism: deviation from expected state |
| `Correction` | 2-morphism: maps error toward zero |
| `ErrorCorrectingContext` | Parity-check matrix, syndrome detection |

Key methods: `Error::compose`, `Correction::compose_vertical`, `Correction::is_isomorphism`, `Correction::quality`

### golden_repair

Kintsugi homotopy: error ≃ correction via golden ratio.

| Type | Description |
|------|-------------|
| `Homotopy` | Linear deformation between error and corrected states |
| `GoldenRepair` | Kintsugi repair at the golden point `t = 1/φ` |

Key methods: `Homotopy::at`, `velocity`, `path_length`, `GoldenRepair::golden_point`, `repair_quality`, `verify_equivalence`

## How It Works

### Architecture

```
           Observation (F)
World ──────────────────────→ Internal Model
  ↑                              │
  │         Control (G)          │
  └──────────────────────────────┘

Unit (η):  World → G(F(World))     = Kalman filter
Counit (ε): F(G(Model)) → Model    = LQR controller
```

### Kalman Filter as Unit

The Kalman filter implements the unit natural transformation `η: id_C → G∘F`:
1. Observe: `F(world)` → internal model
2. Apply Kalman gain: correct the belief
3. Actuate: `G(belief)` → back to world

### LQR as Counit

The Linear-Quadratic Regulator implements the counit `ε: F∘G → id_D`:
1. Compute optimal gain `K` by solving the Discrete Algebraic Riccati Equation (DARE)
2. Apply `u = -Kx̂` to drive the internal model toward zero error
3. Observe the result back into model space

### Triangle Identities

The adjunction is valid iff:
- **Triangle 1**: `ε_F ∘ F(η) = id_F` — observing after unit-then-counit gives the same observation
- **Triangle 2**: `G(ε) ∘ η_G = id_G` — controlling after counit-then-unit gives the same control

Both are verified computationally with tolerance checks.

### Biduality Closure

The biduality module checks that the double application closes:
- `observe(observe(x))` ≈ `control(x)` (left biduality)
- `control(control(y))` ≈ `observe(y)` (right biduality)
- `control(observe(x)) ≈ x` (double-dual isomorphism)

## The Math

### Adjunction

An adjunction `F ⊣ G` between categories `C` and `D` is the most fundamental relationship between two functors. It generalizes:
- Free ⊣ Forgetful (algebra)
- Tensor ⊣ Hom (linear algebra)
- **Observe ⊣ Control** (this crate)

The unit `η: id → GF` and counit `ε: FG → id` satisfy the triangle identities, making the observation-control loop a **self-consistent closed system**.

### Kalman Filter

The optimal linear state estimator for the system:
```
x_{k+1} = A x_k + B u_k + w_k    (process noise w ~ N(0, Q))
y_k     = H x_k + v_k             (measurement noise v ~ N(0, R))
```

The Kalman gain `K = P H^T (H P H^T + R)^{-1}` minimizes the mean-square estimation error.

### LQR Optimal Control

The Linear-Quadratic Regulator minimizes:
```
J = Σ (x^T Q x + u^T R u)
```

The optimal gain `K` is computed by iterating the Discrete Algebraic Riccati Equation (DARE):
```
P_{k+1} = Q + A^T P_k A - A^T P_k B (R + B^T P_k B)^{-1} B^T P_k A
```

### Kintsugi / Golden Repair

Errors and corrections are connected by a **homotopy** `H: [0,1] → State` where `H(0)` is the error state and `H(1)` is the corrected state. The **golden point** `H(1/φ)` at `t ≈ 0.618` represents the aesthetically optimal intermediate — the kintsugi philosophy that repair makes the object more beautiful than the original.

### 2-Category Structure

Errors are **1-morphisms** (transformations between states), corrections are **2-morphisms** (transformations between errors). Vertical composition of corrections corresponds to sequential repair, with confidence decaying multiplicatively.

## Testing

**133 tests** across 10 modules covering:

- **Observation**: world state creation, measurement models, functor composition, morphism reversal
- **Control**: actuation, morphism mapping, LQR gain computation, LQR stabilization (100-step convergence)
- **Adjunction**: Kalman predict/update/converge, unit/counit, triangle identities, roundtrips
- **Biduality**: left/right biduality verification, double-dual isomorphism
- **Agent loop**: 9-step cycle, decompose, classify, predict, full trace
- **PLATO**: observe/predict/control cycle, error computation, correction, grounding, multi-cycle runs
- **Computability**: encode/decode roundtrip, cardinality, quantization error, clamping
- **Error correction**: error composition, identity, correction quality, vertical composition, syndrome detection
- **Golden repair**: homotopy interpolation, golden point at 1/φ, path length, equivalence verification
- **lib.rs**: Gaussian validity, linear map composition, high-dimensional Kalman convergence

## License

MIT
