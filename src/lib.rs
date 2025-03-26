//! # nlopt
//!
//! A safe and idiomatic Rust wrapper for the [NLopt](https://nlopt.readthedocs.io/) library,
//! providing access to powerful algorithms for nonlinear optimization.
//!
//! ## Overview
//!
//! NLopt is a comprehensive library for nonlinear optimization that includes algorithms for:
//!
//! - Global optimization
//! - Local optimization
//! - Constrained optimization
//! - Unconstrained optimization
//! - Gradient-based methods
//! - Derivative-free methods
//!
//! This Rust wrapper provides memory-safe access to all core NLopt functionality while
//! maintaining the performance of the underlying C implementation.
//!
//! ## Basic Usage
//!
//! ```rust,no_run
//! use nlopt::{Nlopt, Algorithm, Target};
//!
//! // Define an objective function to minimize: f(x) = x₁² + x₂²
//! fn objective(x: &[f64], gradient: Option<&mut [f64]>, _: &mut ()) -> f64 {
//!     // Calculate gradient if requested
//!     if let Some(grad) = gradient {
//!         grad[0] = 2.0 * x[0];
//!         grad[1] = 2.0 * x[1];
//!     }
//!     
//!     // Return function value
//!     x[0] * x[0] + x[1] * x[1]
//! }
//!
//! // Create optimizer with LBFGS algorithm for a 2-dimensional problem
//! let mut opt = Nlopt::new(Algorithm::Lbfgs, 2, objective, Target::Minimize, ());
//!
//! // Set bounds: -1 ≤ x ≤ 1
//! opt.set_lower_bound(-1.0).unwrap();
//! opt.set_upper_bound(1.0).unwrap();
//!
//! // Set stopping criteria
//! opt.set_xtol_rel(1e-4).unwrap();
//!
//! // Run optimization from starting point [0.5, 0.5]
//! let mut x = vec![0.5, 0.5];
//! let result = opt.optimize(&mut x);
//!
//! match result {
//!     Ok((status, value)) => println!("Success: {:?}, value = {}, x = {:?}", status, value, x),
//!     Err((error, _)) => println!("Optimization failed: {:?}", error)
//! }
//! ```
//!
//! ## Key Features
//!
//! - **Memory Safety**: Automatic resource management through RAII
//! - **Type Safety**: Strong types for algorithms, constraints, and results
//! - **Callback Support**: Seamless integration with Rust closures
//! - **User Data**: Pass custom data to objective and constraint functions
//! - **Comprehensive API**: Full access to NLopt's algorithms and options
//!
//! ## Advanced Usage
//!
//! For more complex examples, including constrained optimization and advanced
//! algorithms, see the [examples](https://github.com/adwhit/rust-nlopt/tree/master/examples)
//! directory in the repository.
//!
//! For details about specific algorithms and their characteristics, consult the
//! [NLopt Algorithm Documentation](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/).

use std::os::raw::{c_uint, c_ulong, c_void};
use std::slice;

use self::nlopt_sys as sys;

#[allow(non_camel_case_types)]
#[allow(non_upper_case_globals)]
#[allow(dead_code)]
mod nlopt_sys;

/// Specifies whether the objective function should be maximized or minimized.
///
/// This enum determines the optimization direction for the objective function.
/// For example, when minimizing a cost function or maximizing a utility function.
///
/// # Examples
///
/// ```rust,no_run
/// use nlopt::{Nlopt, Algorithm, Target};
///
/// // Minimize an objective function
/// let min_objective = |x: &[f64], _: Option<&mut [f64]>, _: &mut ()| x[0] * x[0] + x[1] * x[1];
/// let mut min_opt = Nlopt::new(Algorithm::Cobyla, 2, min_objective, Target::Minimize, ());
///
/// // Maximize an objective function
/// let max_objective = |x: &[f64], _: Option<&mut [f64]>, _: &mut ()| -(x[0] * x[0] + x[1] * x[1]) + 5.0;
/// let mut max_opt = Nlopt::new(Algorithm::Cobyla, 2, max_objective, Target::Maximize, ());
/// ```
#[derive(Debug, Clone, Copy)]
pub enum Target {
    /// Maximize the objective function (find the highest value).
    Maximize,
    /// Minimize the objective function (find the lowest value).
    Minimize,
}

/// Available optimization algorithms provided by NLopt.
///
/// NLopt provides a wide range of optimization algorithms, suitable for different types of problems.
/// These algorithms can be broadly categorized into:
///
/// - **Global vs. Local**: Global algorithms attempt to find the global optimum, while local
///   algorithms find the nearest local optimum.
/// - **Gradient-Based vs. Derivative-Free**: Some algorithms require gradient information, while
///   others only need function values.
/// - **Stochastic vs. Deterministic**: Some algorithms use randomization, while others are deterministic.
///
/// # Algorithm Naming Conventions
///
/// The algorithm names use prefixes to indicate their characteristics:
///
/// - `GN_`: Global, No-derivative (derivative-free)
/// - `GD_`: Global, Derivative-based
/// - `LN_`: Local, No-derivative (derivative-free)
/// - `LD_`: Local, Derivative-based
///
/// # Examples
///
/// ```rust,no_run
/// use nlopt::{Nlopt, Algorithm, Target};
///
/// // A derivative-free local optimization:
/// let mut local_opt = Nlopt::new(
///     Algorithm::Cobyla,  // Local, derivative-free
///     2,                  // 2 dimensions
///     |x, _, _| x[0]*x[0] + x[1]*x[1],  // Simple objective
///     Target::Minimize,
///     ()
/// );
///
/// // A global optimization:
/// let mut global_opt = Nlopt::new(
///     Algorithm::GnDirect,  // Global, derivative-free
///     2,
///     |x, _, _| x[0]*x[0] + x[1]*x[1],
///     Target::Minimize,
///     ()
/// );
///
/// // A gradient-based optimization:
/// let mut gradient_opt = Nlopt::new(
///     Algorithm::Lbfgs,  // Local, derivative-based
///     2,
///     // Objective with gradient
///     |x, grad, _| {
///         if let Some(grad) = grad {
///             grad[0] = 2.0 * x[0];
///             grad[1] = 2.0 * x[1];
///         }
///         x[0]*x[0] + x[1]*x[1]
///     },
///     Target::Minimize,
///     ()
/// );
/// ```
///
/// For detailed information on each algorithm, refer to the
/// [NLopt Algorithm Documentation](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/).
#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum Algorithm {
    /// DIviding RECTangles algorithm for global optimization (Gablonsky & Kelley version)
    Direct = sys::nlopt_algorithm_NLOPT_GN_DIRECT,
    /// DIviding RECTangles algorithm for global optimization (locally biased version)
    DirectL = sys::nlopt_algorithm_NLOPT_GN_DIRECT_L,
    /// DIviding RECTangles algorithm for global optimization (locally biased version with randomization)
    DirectLRand = sys::nlopt_algorithm_NLOPT_GN_DIRECT_L_RAND,
    /// DIviding RECTangles algorithm for global optimization (unscaled version)
    DirectNoscal = sys::nlopt_algorithm_NLOPT_GN_DIRECT_NOSCAL,
    /// DIviding RECTangles algorithm for global optimization (locally biased unscaled version)
    DirectLNoscal = sys::nlopt_algorithm_NLOPT_GN_DIRECT_L_NOSCAL,
    /// DIviding RECTangles algorithm for global optimization (locally biased unscaled version with randomization)
    DirectLRandNoscal = sys::nlopt_algorithm_NLOPT_GN_DIRECT_L_RAND_NOSCAL,
    /// DIviding RECTangles algorithm for global optimization (original version)
    OrigDirect = sys::nlopt_algorithm_NLOPT_GN_ORIG_DIRECT,
    /// DIviding RECTangles algorithm for global optimization (original locally biased version)
    OrigDirectL = sys::nlopt_algorithm_NLOPT_GN_ORIG_DIRECT_L,
    /// StoGO: global optimization by stochastic gradient descent
    StoGo = sys::nlopt_algorithm_NLOPT_GD_STOGO,
    /// StoGO: global optimization by stochastic gradient descent (randomized variant)
    StoGoRand = sys::nlopt_algorithm_NLOPT_GD_STOGO_RAND,
    /// Limited-memory BFGS algorithm (gradient-based)
    Lbfgs = sys::nlopt_algorithm_NLOPT_LD_LBFGS,
    /// Principal-axis method (gradient-free local optimization)
    Praxis = sys::nlopt_algorithm_NLOPT_LN_PRAXIS,
    /// Limited-memory variable-metric, rank 1 (gradient-based)
    LdVar1 = sys::nlopt_algorithm_NLOPT_LD_VAR1,
    /// Limited-memory variable-metric, rank 2 (gradient-based)
    LdVar2 = sys::nlopt_algorithm_NLOPT_LD_VAR2,
    /// Truncated Newton method (gradient-based)
    TNewton = sys::nlopt_algorithm_NLOPT_LD_TNEWTON,
    /// Truncated Newton method with restarting (gradient-based)
    TNewtonRestart = sys::nlopt_algorithm_NLOPT_LD_TNEWTON_RESTART,
    /// Preconditioned truncated Newton method (gradient-based)
    TNewtonPrecond = sys::nlopt_algorithm_NLOPT_LD_TNEWTON_PRECOND,
    /// Preconditioned truncated Newton method with restarting (gradient-based)
    TNewtonPrecondRestart = sys::nlopt_algorithm_NLOPT_LD_TNEWTON_PRECOND_RESTART,
    /// Controlled Random Search with local mutation (global, gradient-free)
    Crs2Lm = sys::nlopt_algorithm_NLOPT_GN_CRS2_LM,
    /// Multi-level Single-linkage (global, can be used with local optimizer)
    GMlsl = sys::nlopt_algorithm_NLOPT_G_MLSL,
    /// Multi-level Single-linkage with LDS low-discrepancy sequence (global)
    GMlslLds = sys::nlopt_algorithm_NLOPT_G_MLSL_LDS,
    /// Multi-level Single-linkage (global, gradient-free)
    GnMlsl = sys::nlopt_algorithm_NLOPT_GN_MLSL,
    /// Multi-level Single-linkage (global, gradient-based)
    GdMlsl = sys::nlopt_algorithm_NLOPT_GD_MLSL,
    /// Multi-level Single-linkage with LDS (global, gradient-free)
    GnMlslLds = sys::nlopt_algorithm_NLOPT_GN_MLSL_LDS,
    /// Multi-level Single-linkage with LDS (global, gradient-based)
    GdMlslLds = sys::nlopt_algorithm_NLOPT_GD_MLSL_LDS,
    /// Method of Moving Asymptotes (local, gradient-based)
    Mma = sys::nlopt_algorithm_NLOPT_LD_MMA,
    /// Constrained Optimization BY Linear Approximations (local, gradient-free)
    Cobyla = sys::nlopt_algorithm_NLOPT_LN_COBYLA,
    /// NEWUOA algorithm for unconstrained optimization (local, gradient-free)
    Newuoa = sys::nlopt_algorithm_NLOPT_LN_NEWUOA,
    /// NEWUOA algorithm for bounded optimization (local, gradient-free)
    NewuoaBound = sys::nlopt_algorithm_NLOPT_LN_NEWUOA_BOUND,
    /// Nelder-Mead simplex algorithm (local, gradient-free)
    Neldermead = sys::nlopt_algorithm_NLOPT_LN_NELDERMEAD,
    /// Subplex algorithm (local, gradient-free)
    Sbplx = sys::nlopt_algorithm_NLOPT_LN_SBPLX,
    /// Augmented Lagrangian method (local)
    Auglag = sys::nlopt_algorithm_NLOPT_AUGLAG,
    /// Augmented Lagrangian method for equality constraints (local)
    AuglagEq = sys::nlopt_algorithm_NLOPT_AUGLAG_EQ,
    /// Augmented Lagrangian method with local derivative-free optimizer
    LnAuglag = sys::nlopt_algorithm_NLOPT_LN_AUGLAG,
    /// Augmented Lagrangian method for equality constraints with local derivative-free optimizer
    LdAuglagEq = sys::nlopt_algorithm_NLOPT_LD_AUGLAG_EQ,
    /// Augmented Lagrangian method with local derivative-based optimizer
    LdAuglag = sys::nlopt_algorithm_NLOPT_LD_AUGLAG,
    /// Augmented Lagrangian method for equality constraints with local derivative-based optimizer
    LnAuglagEq = sys::nlopt_algorithm_NLOPT_LN_AUGLAG_EQ,
    /// Conservative Convex Separable Approximations (local, gradient-based)
    Ccsaq = sys::nlopt_algorithm_NLOPT_LD_CCSAQ,
    /// ISRES evolutionary algorithm for global optimization with constraints
    Isres = sys::nlopt_algorithm_NLOPT_GN_ISRES,
    /// ESCH evolutionary algorithm for global optimization
    Esch = sys::nlopt_algorithm_NLOPT_GN_ESCH,
    /// Bound Optimization BY Quadratic Approximation (local, gradient-free)
    Bobyqa = sys::nlopt_algorithm_NLOPT_LN_BOBYQA,
    /// Sequential Least-Squares Quadratic Programming (local, gradient-based)
    Slsqp = sys::nlopt_algorithm_NLOPT_LD_SLSQP,
    /// AGS global optimization algorithm
    Ags = sys::nlopt_algorithm_NLOPT_GN_AGS,
}

/// Represents error conditions that can occur during optimization.
///
/// These error states indicate various failure modes that may occur during
/// optimization, such as invalid arguments, resource limitations, or numerical issues.
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum FailState {
    /// Generic failure (unspecified error)
    Failure = sys::nlopt_result_NLOPT_FAILURE,
    /// Invalid arguments (e.g., inconsistent dimensions)
    InvalidArgs = sys::nlopt_result_NLOPT_INVALID_ARGS,
    /// Memory allocation failure
    OutOfMemory = sys::nlopt_result_NLOPT_OUT_OF_MEMORY,
    /// Halted because roundoff errors limited progress
    RoundoffLimited = sys::nlopt_result_NLOPT_ROUNDOFF_LIMITED,
    /// Halted by user-requested termination (via force_stop)
    ForcedStop = sys::nlopt_result_NLOPT_FORCED_STOP,
}

/// Represents successful termination conditions for optimization.
///
/// These success states indicate the various reasons why an optimization algorithm
/// might terminate successfully, such as reaching a target value or meeting a
/// convergence criterion.
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum SuccessState {
    /// Generic success (unspecified cause)
    Success = sys::nlopt_result_NLOPT_SUCCESS,
    /// Optimization stopped because stopval was reached
    StopValReached = sys::nlopt_result_NLOPT_STOPVAL_REACHED,
    /// Optimization stopped because function tolerance was reached
    FtolReached = sys::nlopt_result_NLOPT_FTOL_REACHED,
    /// Optimization stopped because parameter tolerance was reached
    XtolReached = sys::nlopt_result_NLOPT_XTOL_REACHED,
    /// Optimization stopped because maximum number of evaluations was reached
    MaxEvalReached = sys::nlopt_result_NLOPT_MAXEVAL_REACHED,
    /// Optimization stopped because maximum allowed time was reached
    MaxTimeReached = sys::nlopt_result_NLOPT_MAXTIME_REACHED,
}

/// Result type for operations that may fail.
///
/// Most API methods return this type to indicate success or failure.
/// On success, a `SuccessState` enum value indicates why the optimization terminated.
/// On failure, a `FailState` enum value indicates what kind of error occurred.
///
/// # Examples
///
/// ```rust,no_run
/// use nlopt::{Nlopt, Algorithm, Target};
///
/// let objective = |x: &[f64], _: Option<&mut [f64]>, _: &mut ()| x[0] * x[0];
/// let mut opt = Nlopt::new(Algorithm::Cobyla, 1, objective, Target::Minimize, ());
///
/// // Setting optimizer bounds returns an OptResult
/// match opt.set_lower_bound(-1.0) {
///     Ok(_) => println!("Successfully set lower bound"),
///     Err(err) => println!("Failed to set bound: {:?}", err),
/// }
/// ```
pub type OptResult = std::result::Result<SuccessState, FailState>;

fn result_from_outcome(outcome: sys::nlopt_result) -> OptResult {
    use self::FailState::*;
    use self::SuccessState::*;
    if outcome < 0 {
        let err = match outcome {
            sys::nlopt_result_NLOPT_FAILURE => Failure,
            sys::nlopt_result_NLOPT_INVALID_ARGS => InvalidArgs,
            sys::nlopt_result_NLOPT_OUT_OF_MEMORY => OutOfMemory,
            sys::nlopt_result_NLOPT_ROUNDOFF_LIMITED => RoundoffLimited,
            sys::nlopt_result_NLOPT_FORCED_STOP => ForcedStop,
            v => panic!("Unknown fail state {}", v),
        };
        Err(err)
    } else {
        let ok = match outcome {
            sys::nlopt_result_NLOPT_SUCCESS => Success,
            sys::nlopt_result_NLOPT_STOPVAL_REACHED => StopValReached,
            sys::nlopt_result_NLOPT_FTOL_REACHED => FtolReached,
            sys::nlopt_result_NLOPT_XTOL_REACHED => XtolReached,
            sys::nlopt_result_NLOPT_MAXEVAL_REACHED => MaxEvalReached,
            sys::nlopt_result_NLOPT_MAXTIME_REACHED => MaxTimeReached,
            v => panic!("Unknown success state {}", v),
        };
        Ok(ok)
    }
}

extern "C" fn function_raw_callback<F: ObjFn<T>, T>(
    n: c_uint,
    x: *const f64,
    g: *mut f64,
    params: *mut c_void,
) -> f64 {
    // prepare args
    let argument = unsafe { slice::from_raw_parts(x, n as usize) };
    let gradient = if g.is_null() {
        None
    } else {
        Some(unsafe { slice::from_raw_parts_mut(g, n as usize) })
    };

    // recover FunctionCfg object from supplied params and call
    let f = unsafe { &mut *(params as *mut FunctionCfg<F, T>) };
    (f.objective_fn)(argument, gradient, &mut f.user_data)
}

extern "C" fn constraint_raw_callback<F: ObjFn<T>, T>(
    n: c_uint,
    x: *const f64,
    g: *mut f64,
    params: *mut c_void,
) -> f64 {
    // Since ConstraintCfg is just an alias for FunctionCfg,
    // this function is identical to above
    let f = unsafe { &mut *(params as *mut ConstraintCfg<F, T>) };
    let argument = unsafe { slice::from_raw_parts(x, n as usize) };
    let gradient = if g.is_null() {
        None
    } else {
        Some(unsafe { slice::from_raw_parts_mut(g, n as usize) })
    };
    (f.objective_fn)(argument, gradient, &mut f.user_data)
}

extern "C" fn mfunction_raw_callback<F: MObjFn<T>, T>(
    m: u32,
    re: *mut f64,
    n: u32,
    x: *const f64,
    g: *mut f64,
    d: *mut c_void,
) {
    let f = unsafe { &mut *(d as *mut MConstraintCfg<F, T>) };
    let re = unsafe { slice::from_raw_parts_mut(re, m as usize) };
    let argument = unsafe { slice::from_raw_parts(x, n as usize) };
    let gradient: Option<&mut [f64]> = if g.is_null() {
        None
    } else {
        Some(unsafe { slice::from_raw_parts_mut(g, (n as usize) * (m as usize)) })
    };
    (f.constraint)(re, argument, gradient, &mut f.user_data);
}

/// A trait representing an objective function.
///
/// This trait is implemented for any closure that matches the required signature.
/// The objective function is the function that the optimization algorithm will attempt
/// to minimize or maximize.
///
/// # Function Signature
///
/// An objective function takes the form of a closure with three parameters:
///
/// ```rust,no_run
/// fn objective(x: &[f64], gradient: Option<&mut [f64]>, user_data: &mut T) -> f64
/// ```
///
/// * `x` - The current point being evaluated, represented as an n-dimensional array
/// * `gradient` - Optional array to store the gradient ∇f(x) at the current point.
///    If `Some(grad)`, you must calculate and populate the gradient vector.
///    If `None`, the gradient is not needed for the current algorithm.
/// * `user_data` - User-defined data that persists across function calls
///
/// # Gradient Calculation
///
/// For gradient-based algorithms (e.g., `LBFGS`, `MMA`), you must provide the gradient
/// when requested. For algorithms that don't use gradients, the `gradient` parameter
/// will always be `None`. If you don't want to calculate gradients manually, consider:
///
/// 1. Using derivative-free algorithms (e.g., `COBYLA`, `BOBYQA`, `NELDERMEAD`)
/// 2. Using the `approximate_gradient` helper function
/// 3. Using automatic differentiation libraries
///
/// # Examples
///
/// Basic objective function for minimizing x² + y²:
///
/// ```rust,no_run
/// use nlopt::{Nlopt, Algorithm, Target};
///
/// // Define objective function with gradient
/// fn rosenbrock(x: &[f64], gradient: Option<&mut [f64]>, _: &mut ()) -> f64 {
///     let a = 1.0;
///     let b = 100.0;
///     let f = (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2);
///     
///     // If gradient requested, calculate it
///     if let Some(grad) = gradient {
///         grad[0] = -2.0 * a + 2.0 * x[0] - 4.0 * b * x[0] * (x[1] - x[0].powi(2));
///         grad[1] = 2.0 * b * (x[1] - x[0].powi(2));
///     }
///     
///     f
/// }
///
/// // Create optimizer and set it up
/// let mut opt = Nlopt::new(Algorithm::Lbfgs, 2, rosenbrock, Target::Minimize, ());
/// ```
///
/// Using user data to track function evaluations:
///
/// ```rust,no_run
/// use nlopt::{Nlopt, Algorithm, Target};
///
/// // Define a struct to hold statistics
/// struct Stats {
///     eval_count: usize,
///     best_value: f64,
/// }
///
/// // Function with user data
/// fn objective(x: &[f64], _: Option<&mut [f64]>, stats: &mut Stats) -> f64 {
///     let value = x[0].powi(2) + x[1].powi(2);
///     
///     // Update statistics
///     stats.eval_count += 1;
///     stats.best_value = stats.best_value.min(value);
///     
///     value
/// }
///
/// // Create optimizer with user data
/// let mut stats = Stats { eval_count: 0, best_value: f64::INFINITY };
/// let mut opt = Nlopt::new(Algorithm::Cobyla, 2, objective, Target::Minimize, stats);
///
/// // After optimization, recover the stats
/// let mut x = vec![0.5, 0.5];
/// opt.optimize(&mut x).unwrap();
/// let stats = opt.recover_user_data();
/// println!("Function evaluations: {}", stats.eval_count);
/// ```
pub trait ObjFn<U>: Fn(&[f64], Option<&mut [f64]>, &mut U) -> f64 {}

impl<T, U> ObjFn<U> for T where T: Fn(&[f64], Option<&mut [f64]>, &mut U) -> f64 {}

/// Packs an objective function with a user defined parameter set of type `T`.
struct FunctionCfg<F: ObjFn<T>, T> {
    pub objective_fn: F,
    pub user_data: T,
}

type ConstraintCfg<F, T> = FunctionCfg<F, T>;

/// A trait representing a multi-objective function.
///
/// A multi-objective function is used primarily for defining multiple constraints
/// at once. Instead of returning a single value, it populates an array of results.
///
/// # Function Signature
///
/// A multi-objective function takes the form:
///
/// ```rust,no_run
/// fn multi_constraint(result: &mut [f64], x: &[f64], gradient: Option<&mut [f64]>, user_data: &mut T)
/// ```
///
/// * `result` - `m`-dimensional array to store the function values `f(x)`
/// * `x` - `n`-dimensional input array representing the current point
/// * `gradient` - Optional `n×m`-dimensional array to store the gradient.
///    If provided, each constraint's gradient should be stored contiguously:
///    `df_i/dx_j` is stored in `gradient[i*n + j]`
/// * `user_data` - User-defined data that persists across function calls
///
/// # When to Use
///
/// Multi-objective functions are useful when you have multiple related constraints
/// that share computation or when you want to define a large number of constraints
/// efficiently.
///
/// # Example
///
/// ```rust,no_run
/// use nlopt::{Nlopt, Algorithm, Target};
///
/// // Define a multi-constraint function for two inequality constraints:
/// // g₁(x) = x[0]² + x[1]² - 1 ≤ 0
/// // g₂(x) = x[0] - x[1] ≤ 0
/// fn multi_constraint(result: &mut [f64], x: &[f64], gradient: Option<&mut [f64]>, _: &mut ()) {
///     // First constraint: x[0]² + x[1]² - 1 ≤ 0
///     result[0] = x[0]*x[0] + x[1]*x[1] - 1.0;
///     
///     // Second constraint: x[0] - x[1] ≤ 0
///     result[1] = x[0] - x[1];
///     
///     // Calculate gradients if requested
///     if let Some(grad) = gradient {
///         // Gradient of first constraint
///         grad[0] = 2.0 * x[0];  // df₁/dx₁
///         grad[1] = 2.0 * x[1];  // df₁/dx₂
///         
///         // Gradient of second constraint
///         grad[2] = 1.0;  // df₂/dx₁
///         grad[3] = -1.0; // df₂/dx₂
///     }
/// }
///
/// // Use this multi-constraint in an optimizer
/// let mut opt = Nlopt::new(Algorithm::Slsqp, 2,
///     |x, _, _| x[0] + x[1],  // Objective function to minimize
///     Target::Minimize,
///     ()
/// );
///
/// // Add the multi-constraint (2 constraints)
/// opt.add_inequality_mconstraint(
///     2,                      // Number of constraints
///     multi_constraint,       // Multi-constraint function
///     (),                     // User data
///     &[1e-8, 1e-8]           // Tolerances for each constraint
/// ).unwrap();
/// ```
pub trait MObjFn<U>: Fn(&mut [f64], &[f64], Option<&mut [f64]>, &mut U) {}

impl<T, U> MObjFn<U> for T where T: Fn(&mut [f64], &[f64], Option<&mut [f64]>, &mut U) {}

/// Packs an `m`-dimensional function of type `NLoptMFn<T>` with a user defined parameter set of type `T`.
struct MConstraintCfg<F: MObjFn<T>, T> {
    constraint: F,
    user_data: T,
}

// We wrap sys::nlopt_opt in this wrapper to ensure it is correctly
// cleaned up when dropped
struct WrapSysNlopt(sys::nlopt_opt);

impl Drop for WrapSysNlopt {
    fn drop(&mut self) {
        unsafe {
            sys::nlopt_destroy(self.0);
        };
    }
}

/// The main optimizer struct that encapsulates the NLopt optimization process.
///
/// This struct represents a complete optimization problem, including:
/// - The objective function to optimize
/// - The algorithm to use
/// - The dimension of the problem
/// - The optimization direction (minimize/maximize)
/// - User-defined data passed to the objective function
///
/// The struct is parameterized by:
/// - `F`: The type of the objective function (must implement `ObjFn<T>`)
/// - `T`: The type of user-defined data to pass to the objective function
///
/// # Creating an Optimizer
///
/// An optimizer is created using the `new` method, which requires:
/// - An optimization algorithm
/// - The dimension of the problem (number of variables)
/// - The objective function to optimize
/// - The optimization target (minimize or maximize)
/// - User-defined data (can be an empty tuple `()` if not needed)
///
/// # Memory Safety
///
/// The optimizer automatically handles allocation and deallocation of NLopt resources,
/// ensuring memory safety through Rust's RAII pattern. The wrapped NLopt object is
/// properly cleaned up when the `Nlopt` struct is dropped.
///
/// # Example
///
/// ```rust,no_run
/// use nlopt::{Nlopt, Algorithm, Target};
///
/// // Define a simple quadratic function to minimize: f(x) = x²
/// let objective = |x: &[f64], grad: Option<&mut [f64]>, _: &mut ()| {
///     // Set gradient if requested (df/dx = 2x)
///     if let Some(grad) = grad {
///         grad[0] = 2.0 * x[0];
///     }
///     x[0] * x[0]
/// };
///
/// // Create an optimizer for a 1-dimensional problem
/// let mut opt = Nlopt::new(
///     Algorithm::Lbfgs,       // Algorithm to use
///     1,                      // Dimension of the problem
///     objective,              // Objective function
///     Target::Minimize,       // Optimization direction
///     ()                      // User data (empty in this case)
/// );
///
/// // Set optimization parameters
/// opt.set_lower_bound(-10.0).unwrap();
/// opt.set_upper_bound(10.0).unwrap();
/// opt.set_xtol_rel(1e-4).unwrap();
///
/// // Run the optimization
/// let mut x = vec![3.0]; // Starting point
/// match opt.optimize(&mut x) {
///     Ok((status, value)) => println!("Optimal x = {}, f(x) = {}", x[0], value),
///     Err(e) => println!("Optimization failed: {:?}", e)
/// }
/// ```
pub struct Nlopt<F: ObjFn<T>, T> {
    algorithm: Algorithm,
    n_dims: usize,
    target: Target,
    nloptc_obj: WrapSysNlopt,
    func_cfg: Box<FunctionCfg<F, T>>,
}

impl<F: ObjFn<T>, T> Nlopt<F, T> {
    /// Creates a new optimizer instance for solving nonlinear optimization problems.
    ///
    /// This constructor creates and initializes a new optimizer with the specified algorithm,
    /// dimension, objective function, optimization target, and user data. This is the primary
    /// entry point for using the library.
    ///
    /// # Parameters
    ///
    /// * `algorithm` - The optimization algorithm to use. This cannot be changed after the
    ///   optimizer is created. Different algorithms have different characteristics and are
    ///   suitable for different types of problems.
    ///
    /// * `n_dims` - The dimension of the optimization problem (number of variables).
    ///   This defines the length of the input vector `x` to the objective function.
    ///
    /// * `objective_fn` - The function to optimize. It must have the signature
    ///   `Fn(&[f64], Option<&mut [f64]>, &mut T) -> f64` where:
    ///   - The first argument is the point `x` at which to evaluate the function
    ///   - The second argument is an optional mutable slice to store the gradient
    ///     (if the algorithm requires it)
    ///   - The third argument is the user data
    ///   - The return value is the function value at point `x`
    ///
    /// * `target` - Specifies whether to minimize or maximize the objective function.
    ///
    /// * `user_data` - Custom data to be passed to the objective function. This can be
    ///   any type that you want to make available to your objective function, such as
    ///   problem parameters, statistics collectors, or application state.
    ///
    /// # Returns
    ///
    /// A new `Nlopt<F, T>` instance configured with the specified parameters.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Define a simple objective function
    /// fn sphere(x: &[f64], grad: Option<&mut [f64]>, _: &mut ()) -> f64 {
    ///     // Calculate gradient if requested
    ///     if let Some(grad) = grad {
    ///         for i in 0..x.len() {
    ///             grad[i] = 2.0 * x[i];
    ///         }
    ///     }
    ///     
    ///     // Return sum of squares
    ///     x.iter().map(|xi| xi * xi).sum()
    /// }
    ///
    /// // Create a 3-dimensional optimizer to minimize the sphere function
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Lbfgs,  // A gradient-based algorithm
    ///     3,                 // 3 dimensions
    ///     sphere,            // Our objective function
    ///     Target::Minimize,  // We want to minimize the function
    ///     ()                 // No user data needed
    /// );
    /// ```
    ///
    /// Using custom user data:
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Define problem parameters
    /// struct ProblemParams {
    ///     weights: Vec<f64>,
    ///     bias: f64,
    ///     eval_count: usize,
    /// }
    ///
    /// // Weighted sum objective function
    /// fn weighted_sum(x: &[f64], _: Option<&mut [f64]>, params: &mut ProblemParams) -> f64 {
    ///     params.eval_count += 1;
    ///     
    ///     let mut sum = params.bias;
    ///     for (xi, wi) in x.iter().zip(params.weights.iter()) {
    ///         sum += xi * wi;
    ///     }
    ///     sum
    /// }
    ///
    /// // Initialize user data
    /// let mut params = ProblemParams {
    ///     weights: vec![1.0, 2.0, 3.0],
    ///     bias: 5.0,
    ///     eval_count: 0,
    /// };
    ///
    /// // Create optimizer with user data
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     3,
    ///     weighted_sum,
    ///     Target::Minimize,
    ///     params
    /// );
    /// ```
    pub fn new(
        algorithm: Algorithm,
        n_dims: usize,
        objective_fn: F,
        target: Target,
        user_data: T,
    ) -> Nlopt<F, T> {
        // TODO this might be better off as a builder pattern
        let nloptc_obj = unsafe { sys::nlopt_create(algorithm as u32, n_dims as u32) };

        // Our strategy is to pass the actual objective function as part of the
        // parameters to the callback. For this we pack it inside a FunctionCfg struct.
        // We allocation our FunctionCfg on the heap and pass a pointer to the C lib
        // (This is pretty unsafe but works).
        // `into_raw` will leak the boxed object
        let func_cfg = Box::new(FunctionCfg {
            objective_fn,
            user_data, // move user_data into FunctionCfg
        });

        let fn_cfg_ptr = &*func_cfg as *const _ as *mut c_void;
        let nlopt = Nlopt {
            algorithm,
            n_dims,
            target,
            nloptc_obj: WrapSysNlopt(nloptc_obj),
            func_cfg,
        };
        match target {
            Target::Minimize => unsafe {
                sys::nlopt_set_min_objective(
                    nlopt.nloptc_obj.0,
                    Some(function_raw_callback::<F, T>),
                    fn_cfg_ptr,
                )
            },
            Target::Maximize => unsafe {
                sys::nlopt_set_max_objective(
                    nlopt.nloptc_obj.0,
                    Some(function_raw_callback::<F, T>),
                    fn_cfg_ptr,
                )
            },
        };
        nlopt
    }

    /// Returns the optimization algorithm used by this optimizer.
    ///
    /// This method allows you to retrieve the algorithm that was specified
    /// when creating the optimizer.
    ///
    /// # Returns
    ///
    /// The `Algorithm` enum value representing the current optimization algorithm.
    pub fn get_algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// Consumes the optimizer and returns the user data.
    ///
    /// This method is useful for retrieving the user data after optimization,
    /// especially when the user data has been used to collect statistics or
    /// other information during the optimization process.
    ///
    /// # Returns
    ///
    /// The user data that was passed to the optimizer's constructor.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Define a struct to hold statistics
    /// struct Stats {
    ///     evaluations: usize,
    ///     best_value: f64,
    /// }
    ///
    /// // Define objective function that updates stats
    /// let objective = |x: &[f64], _: Option<&mut [f64]>, stats: &mut Stats| {
    ///     let value = x[0] * x[0];
    ///     stats.evaluations += 1;
    ///     stats.best_value = stats.best_value.min(value);
    ///     value
    /// };
    ///
    /// // Initialize stats and create optimizer
    /// let stats = Stats { evaluations: 0, best_value: f64::INFINITY };
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     1,
    ///     objective,
    ///     Target::Minimize,
    ///     stats
    /// );
    ///
    /// // Run optimization
    /// let mut x = vec![10.0];
    /// opt.optimize(&mut x).unwrap();
    ///
    /// // Recover stats
    /// let stats = opt.recover_user_data();
    /// println!("Function evaluated {} times", stats.evaluations);
    /// println!("Best value seen: {}", stats.best_value);
    /// ```
    pub fn recover_user_data(self) -> T {
        self.func_cfg.user_data
    }

    /// Sets the lower bounds for the optimization variables.
    ///
    /// Most optimization algorithms in NLopt support simple bound constraints on the
    /// optimization variables. This method sets the lower bounds for all variables.
    ///
    /// # Parameters
    ///
    /// * `bound` - A slice of length equal to the problem dimension, where each element
    ///   specifies the lower bound for the corresponding optimization variable.
    ///
    /// # Returns
    ///
    /// * `Ok(SuccessState)` if the bounds were successfully set
    /// * `Err(FailState)` if there was an error (e.g., incorrect dimension)
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create a 3-dimensional optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     3,
    ///     |x, _, _| x[0]*x[0] + x[1]*x[1] + x[2]*x[2],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Set different lower bounds for each variable
    /// opt.set_lower_bounds(&[0.0, -1.0, -5.0]).unwrap();
    /// ```
    ///
    /// # Notes
    ///
    /// * A bound of negative infinity indicates no bound.
    /// * If a lower bound equals an upper bound for a given variable, that variable becomes fixed.
    /// * Some algorithms (especially global ones) require finite bounds on all variables.
    pub fn set_lower_bounds(&mut self, bound: &[f64]) -> OptResult {
        result_from_outcome(unsafe {
            sys::nlopt_set_lower_bounds(self.nloptc_obj.0, bound.as_ptr())
        })
    }

    /// Sets the upper bounds for the optimization variables.
    ///
    /// Most optimization algorithms in NLopt support simple bound constraints on the
    /// optimization variables. This method sets the upper bounds for all variables.
    ///
    /// # Parameters
    ///
    /// * `bound` - A slice of length equal to the problem dimension, where each element
    ///   specifies the upper bound for the corresponding optimization variable.
    ///
    /// # Returns
    ///
    /// * `Ok(SuccessState)` if the bounds were successfully set
    /// * `Err(FailState)` if there was an error (e.g., incorrect dimension)
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create a 3-dimensional optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     3,
    ///     |x, _, _| x[0]*x[0] + x[1]*x[1] + x[2]*x[2],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Set different upper bounds for each variable
    /// opt.set_upper_bounds(&[10.0, 5.0, 1.0]).unwrap();
    /// ```
    ///
    /// # Notes
    ///
    /// * A bound of positive infinity indicates no bound.
    /// * If a lower bound equals an upper bound for a given variable, that variable becomes fixed.
    /// * Some algorithms (especially global ones) require finite bounds on all variables.
    pub fn set_upper_bounds(&mut self, bound: &[f64]) -> OptResult {
        result_from_outcome(unsafe {
            sys::nlopt_set_upper_bounds(self.nloptc_obj.0, bound.as_ptr())
        })
    }

    /// Sets the same lower bound for all optimization variables.
    ///
    /// This is a convenience method that sets the same lower bound for all variables
    /// in the optimization problem.
    ///
    /// # Parameters
    ///
    /// * `bound` - The lower bound value to apply to all variables.
    ///
    /// # Returns
    ///
    /// * `Ok(SuccessState)` if the bounds were successfully set
    /// * `Err(FailState)` if there was an error
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create a 3-dimensional optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     3,
    ///     |x, _, _| x[0]*x[0] + x[1]*x[1] + x[2]*x[2],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Set a lower bound of zero for all variables
    /// opt.set_lower_bound(0.0).unwrap();
    /// ```
    ///
    /// # Notes
    ///
    /// * This is equivalent to calling `set_lower_bounds` with a vector where all elements
    ///   are equal to the specified bound.
    /// * A bound of negative infinity indicates no bound.
    pub fn set_lower_bound(&mut self, bound: f64) -> OptResult {
        let v = vec![bound; self.n_dims];
        self.set_lower_bounds(&v)
    }

    /// Sets the same upper bound for all optimization variables.
    ///
    /// This is a convenience method that sets the same upper bound for all variables
    /// in the optimization problem.
    ///
    /// # Parameters
    ///
    /// * `bound` - The upper bound value to apply to all variables.
    ///
    /// # Returns
    ///
    /// * `Ok(SuccessState)` if the bounds were successfully set
    /// * `Err(FailState)` if there was an error
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create a 3-dimensional optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     3,
    ///     |x, _, _| x[0]*x[0] + x[1]*x[1] + x[2]*x[2],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Set an upper bound of 10.0 for all variables
    /// opt.set_upper_bound(10.0).unwrap();
    /// ```
    ///
    /// # Notes
    ///
    /// * This is equivalent to calling `set_upper_bounds` with a vector where all elements
    ///   are equal to the specified bound.
    /// * A bound of positive infinity indicates no bound.
    pub fn set_upper_bound(&mut self, bound: f64) -> OptResult {
        let v = vec![bound; self.n_dims];
        self.set_upper_bounds(&v)
    }

    /// Retrieves the current upper bounds for all optimization variables.
    ///
    /// This method returns the current upper bounds that have been set for the
    /// optimization variables.
    ///
    /// # Returns
    ///
    /// * `Some(Vec<f64>)` containing the upper bounds if successful
    /// * `None` if there was an error retrieving the bounds
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create a 2-dimensional optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     2,
    ///     |x, _, _| x[0]*x[0] + x[1]*x[1],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Set and then retrieve bounds
    /// opt.set_upper_bounds(&[5.0, 10.0]).unwrap();
    /// let bounds = opt.get_upper_bounds().unwrap();
    /// assert_eq!(bounds, vec![5.0, 10.0]);
    /// ```
    pub fn get_upper_bounds(&self) -> Option<Vec<f64>> {
        let mut bound: Vec<f64> = vec![0.0_f64; self.n_dims];
        let b = bound.as_mut_ptr();
        let res = unsafe { sys::nlopt_get_upper_bounds(self.nloptc_obj.0, b as *mut f64) };
        result_from_outcome(res).ok().map(|_| bound)
    }

    /// Retrieves the current lower bounds for all optimization variables.
    ///
    /// This method returns the current lower bounds that have been set for the
    /// optimization variables.
    ///
    /// # Returns
    ///
    /// * `Some(Vec<f64>)` containing the lower bounds if successful
    /// * `None` if there was an error retrieving the bounds
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create a 2-dimensional optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     2,
    ///     |x, _, _| x[0]*x[0] + x[1]*x[1],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Set and then retrieve bounds
    /// opt.set_lower_bounds(&[-1.0, -2.0]).unwrap();
    /// let bounds = opt.get_lower_bounds().unwrap();
    /// assert_eq!(bounds, vec![-1.0, -2.0]);
    /// ```
    pub fn get_lower_bounds(&self) -> Option<Vec<f64>> {
        let mut bound: Vec<f64> = vec![0.0_f64; self.n_dims];
        let b = bound.as_mut_ptr();
        let res = unsafe { sys::nlopt_get_lower_bounds(self.nloptc_obj.0, b as *mut f64) };
        result_from_outcome(res).ok().map(|_| bound)
    }

    /// Adds an equality constraint to the optimization problem.
    ///
    /// An equality constraint is of the form h(x) = 0. The algorithm will attempt to
    /// find a solution where the constraint function h(x) is approximately zero
    /// (within the specified tolerance).
    ///
    /// # Parameters
    ///
    /// * `constraint` - The constraint function h(x)
    /// * `user_data` - Custom data to pass to the constraint function
    /// * `tolerance` - The tolerance within which the constraint is considered satisfied
    ///
    /// # Returns
    ///
    /// * `Ok(SuccessState)` if the constraint was successfully added
    /// * `Err(FailState)` if there was an error
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Constraint function: h(x) = x[0] + x[1] - 1 = 0
    /// // This constrains the solution to lie on the line x[0] + x[1] = 1
    /// fn eq_constraint(x: &[f64], grad: Option<&mut [f64]>, _: &mut ()) -> f64 {
    ///     if let Some(grad) = grad {
    ///         grad[0] = 1.0;  // ∂h/∂x₀ = 1
    ///         grad[1] = 1.0;  // ∂h/∂x₁ = 1
    ///     }
    ///     x[0] + x[1] - 1.0
    /// }
    ///
    /// // Create optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Slsqp,
    ///     2,
    ///     |x, _, _| x[0]*x[0] + x[1]*x[1],  // Minimize sum of squares
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Add equality constraint: x[0] + x[1] = 1
    /// opt.add_equality_constraint(eq_constraint, (), 1e-8).unwrap();
    /// ```
    ///
    /// # Notes
    ///
    /// * Not all optimization algorithms support equality constraints.
    /// * For algorithms that don't inherently support equality constraints,
    ///   the constraint may be implemented by adding a penalty to the objective function.
    /// * The tolerance parameter specifies how close h(x) must be to zero for the
    ///   constraint to be considered satisfied.
    pub fn add_equality_constraint<G: ObjFn<U>, U>(
        &mut self,
        constraint: G,
        user_data: U,
        tolerance: f64,
    ) -> OptResult {
        self.add_constraint(constraint, user_data, tolerance, true)
    }

    /// Adds an inequality constraint to the optimization problem.
    ///
    /// An inequality constraint is of the form g(x) ≤ 0. The algorithm will attempt to
    /// find a solution where the constraint function g(x) is less than or equal to zero.
    ///
    /// # Parameters
    ///
    /// * `constraint` - The constraint function g(x)
    /// * `user_data` - Custom data to pass to the constraint function
    /// * `tolerance` - The tolerance within which the constraint is considered satisfied
    ///
    /// # Returns
    ///
    /// * `Ok(SuccessState)` if the constraint was successfully added
    /// * `Err(FailState)` if there was an error
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Constraint function: g(x) = x[0]² + x[1]² - 1 ≤ 0
    /// // This constrains the solution to lie within the unit circle
    /// fn ineq_constraint(x: &[f64], grad: Option<&mut [f64]>, _: &mut ()) -> f64 {
    ///     if let Some(grad) = grad {
    ///         grad[0] = 2.0 * x[0];  // ∂g/∂x₀ = 2x₀
    ///         grad[1] = 2.0 * x[1];  // ∂g/∂x₁ = 2x₁
    ///     }
    ///     x[0]*x[0] + x[1]*x[1] - 1.0
    /// }
    ///
    /// // Create optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Slsqp,
    ///     2,
    ///     |x, _, _| x[0] + x[1],  // Minimize x₀ + x₁
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Add inequality constraint: x[0]² + x[1]² ≤ 1
    /// opt.add_inequality_constraint(ineq_constraint, (), 1e-8).unwrap();
    /// ```
    ///
    /// # Notes
    ///
    /// * Not all optimization algorithms support inequality constraints.
    /// * For algorithms that don't inherently support inequality constraints,
    ///   the constraint may be implemented by adding a penalty to the objective function.
    /// * The tolerance parameter specifies how close g(x) must be to zero (when positive)
    ///   for the constraint to be considered satisfied.
    pub fn add_inequality_constraint<G: ObjFn<U>, U>(
        &mut self,
        constraint: G,
        user_data: U,
        tolerance: f64,
    ) -> OptResult {
        self.add_constraint(constraint, user_data, tolerance, false)
    }

    fn add_constraint<G: ObjFn<U>, U>(
        &mut self,
        constraint: G,
        user_data: U,
        tolerance: f64,
        is_equality: bool,
    ) -> OptResult {
        let constraint = ConstraintCfg {
            objective_fn: constraint,
            user_data,
        };
        let ptr = Box::into_raw(Box::new(constraint)) as *mut c_void;
        let outcome = unsafe {
            if is_equality {
                sys::nlopt_add_equality_constraint(
                    self.nloptc_obj.0,
                    Some(constraint_raw_callback::<G, U>),
                    ptr,
                    tolerance,
                )
            } else {
                sys::nlopt_add_inequality_constraint(
                    self.nloptc_obj.0,
                    Some(constraint_raw_callback::<G, U>),
                    ptr,
                    tolerance,
                )
            }
        };
        result_from_outcome(outcome)
    }

    /// Adds multiple equality constraints to the optimization problem at once.
    ///
    /// This method allows you to define multiple equality constraints h(x) = 0
    /// using a single function that computes all constraint values at once.
    /// This can be more efficient when constraints share computation or when
    /// you have many related constraints.
    ///
    /// # Parameters
    ///
    /// * `m` - The number of constraints
    /// * `constraint` - A function that populates an array with m constraint values
    /// * `user_data` - Custom data to pass to the constraint function
    /// * `tolerance` - An array of m tolerances, one for each constraint
    ///
    /// # Returns
    ///
    /// * `Ok(SuccessState)` if the constraints were successfully added
    /// * `Err(FailState)` if there was an error
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // A function that computes two equality constraints:
    /// // h₁(x) = x[0] + x[1] - 1 = 0
    /// // h₂(x) = x[0] - x[1] = 0
    /// fn multi_eq_constraint(result: &mut [f64], x: &[f64], grad: Option<&mut [f64]>, _: &mut ()) {
    ///     // First constraint: x[0] + x[1] - 1 = 0
    ///     result[0] = x[0] + x[1] - 1.0;
    ///     
    ///     // Second constraint: x[0] - x[1] = 0
    ///     result[1] = x[0] - x[1];
    ///     
    ///     // Calculate gradients if requested
    ///     if let Some(grad) = grad {
    ///         // For h₁:
    ///         grad[0] = 1.0;  // ∂h₁/∂x₀
    ///         grad[1] = 1.0;  // ∂h₁/∂x₁
    ///         
    ///         // For h₂:
    ///         grad[2] = 1.0;  // ∂h₂/∂x₀
    ///         grad[3] = -1.0; // ∂h₂/∂x₁
    ///     }
    /// }
    ///
    /// // Create optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Auglag,  // Algorithm that supports multiple equality constraints
    ///     2,
    ///     |x, _, _| x[0]*x[0] + x[1]*x[1],  // Minimize sum of squares
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Add both equality constraints simultaneously
    /// opt.add_equality_mconstraint(
    ///     2,                         // Two constraints
    ///     multi_eq_constraint,       // The function computing both constraints
    ///     (),                        // No user data needed
    ///     &[1e-8, 1e-8]              // Tolerances for each constraint
    /// ).unwrap();
    /// ```
    ///
    /// # Notes
    ///
    /// * Not all optimization algorithms support equality constraints.
    /// * The gradient, if requested, should be stored in row-major order:
    ///   grad[i*n + j] = ∂h_i/∂x_j, where n is the dimension of x.
    /// * The number of constraints m must match the length of the tolerance array.
    pub fn add_equality_mconstraint<G: MObjFn<U>, U>(
        &mut self,
        m: usize,
        constraint: G,
        user_data: U,
        tolerance: &[f64],
    ) -> OptResult {
        self.add_mconstraint(m, constraint, user_data, tolerance, true)
    }

    /// Adds multiple inequality constraints to the optimization problem at once.
    ///
    /// This method allows you to define multiple inequality constraints g(x) ≤ 0
    /// using a single function that computes all constraint values at once.
    /// This can be more efficient when constraints share computation or when
    /// you have many related constraints.
    ///
    /// # Parameters
    ///
    /// * `m` - The number of constraints
    /// * `constraint` - A function that populates an array with m constraint values
    /// * `user_data` - Custom data to pass to the constraint function
    /// * `tolerance` - An array of m tolerances, one for each constraint
    ///
    /// # Returns
    ///
    /// * `Ok(SuccessState)` if the constraints were successfully added
    /// * `Err(FailState)` if there was an error
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // A function that computes two inequality constraints:
    /// // g₁(x) = x[0]² + x[1]² - 1 ≤ 0  (inside unit circle)
    /// // g₂(x) = x[0] + x[1] - 2 ≤ 0    (below line x₀ + x₁ = 2)
    /// fn multi_ineq_constraint(result: &mut [f64], x: &[f64], grad: Option<&mut [f64]>, _: &mut ()) {
    ///     // First constraint: x[0]² + x[1]² - 1 ≤ 0
    ///     result[0] = x[0]*x[0] + x[1]*x[1] - 1.0;
    ///     
    ///     // Second constraint: x[0] + x[1] - 2 ≤ 0
    ///     result[1] = x[0] + x[1] - 2.0;
    ///     
    ///     // Calculate gradients if requested
    ///     if let Some(grad) = grad {
    ///         // For g₁:
    ///         grad[0] = 2.0 * x[0];  // ∂g₁/∂x₀
    ///         grad[1] = 2.0 * x[1];  // ∂g₁/∂x₁
    ///         
    ///         // For g₂:
    ///         grad[2] = 1.0;  // ∂g₂/∂x₀
    ///         grad[3] = 1.0;  // ∂g₂/∂x₁
    ///     }
    /// }
    ///
    /// // Create optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Slsqp,  // Algorithm that supports inequality constraints
    ///     2,
    ///     |x, _, _| -x[0]*x[1],  // Maximize product x₀*x₁
    ///     Target::Minimize,      // Note: minimizing -x₀*x₁ is maximizing x₀*x₁
    ///     ()
    /// );
    ///
    /// // Add both inequality constraints simultaneously
    /// opt.add_inequality_mconstraint(
    ///     2,                         // Two constraints
    ///     multi_ineq_constraint,     // The function computing both constraints
    ///     (),                        // No user data needed
    ///     &[1e-8, 1e-8]              // Tolerances for each constraint
    /// ).unwrap();
    /// ```
    ///
    /// # Notes
    ///
    /// * Not all optimization algorithms support inequality constraints.
    /// * The gradient, if requested, should be stored in row-major order:
    ///   grad[i*n + j] = ∂g_i/∂x_j, where n is the dimension of x.
    /// * The number of constraints m must match the length of the tolerance array.
    pub fn add_inequality_mconstraint<G: MObjFn<U>, U>(
        &mut self,
        m: usize,
        constraint: G,
        user_data: U,
        tolerance: &[f64],
    ) -> OptResult {
        self.add_mconstraint(m, constraint, user_data, tolerance, false)
    }

    fn add_mconstraint<G: MObjFn<U>, U>(
        &mut self,
        m: usize,
        constraint: G,
        user_data: U,
        tolerance: &[f64],
        is_equality: bool,
    ) -> OptResult {
        assert_eq!(m, tolerance.len());
        let mconstraint = MConstraintCfg {
            constraint,
            user_data,
        };
        let ptr = Box::into_raw(Box::new(mconstraint)) as *mut c_void;
        let outcome = unsafe {
            if is_equality {
                sys::nlopt_add_equality_mconstraint(
                    self.nloptc_obj.0,
                    m as c_uint,
                    Some(mfunction_raw_callback::<G, U>),
                    ptr,
                    tolerance.as_ptr(),
                )
            } else {
                sys::nlopt_add_inequality_mconstraint(
                    self.nloptc_obj.0,
                    m as c_uint,
                    Some(mfunction_raw_callback::<G, U>),
                    ptr,
                    tolerance.as_ptr(),
                )
            }
        };
        result_from_outcome(outcome)
    }

    /// Removes all inequality and equality constraints from the optimization problem.
    ///
    /// This method clears all previously added constraints, allowing you to start
    /// over with a clean slate or to replace constraints with a new set.
    ///
    /// # Returns
    ///
    /// * `Ok(SuccessState)` if the constraints were successfully removed
    /// * `Err(FailState)` if there was an error
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Slsqp,
    ///     2,
    ///     |x, _, _| x[0]*x[0] + x[1]*x[1],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Add a constraint
    /// opt.add_inequality_constraint(
    ///     |x, _, _| x[0] + x[1] - 1.0,  // g(x): x[0] + x[1] - 1 ≤ 0
    ///     (),
    ///     1e-8
    /// ).unwrap();
    ///
    /// // Later, remove all constraints
    /// opt.remove_constraints().unwrap();
    /// ```
    ///
    /// # Notes
    ///
    /// * This method removes both inequality and equality constraints.
    /// * Bound constraints (set with `set_lower_bounds` and `set_upper_bounds`) are not affected.
    pub fn remove_constraints(&mut self) -> OptResult {
        result_from_outcome(unsafe {
            std::cmp::min(
                sys::nlopt_remove_inequality_constraints(self.nloptc_obj.0),
                sys::nlopt_remove_equality_constraints(self.nloptc_obj.0),
            )
        })
    }

    /// Sets a stopping value for the objective function.
    ///
    /// The optimization will stop when the objective function reaches or crosses
    /// this value. For minimization, the algorithm stops when f(x) ≤ stopval.
    /// For maximization, it stops when f(x) ≥ stopval.
    ///
    /// # Parameters
    ///
    /// * `stopval` - The objective function value at which to stop optimization
    ///
    /// # Returns
    ///
    /// * `Ok(SuccessState)` if the stopping value was successfully set
    /// * `Err(FailState)` if there was an error
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     2,
    ///     |x, _, _| x[0]*x[0] + x[1]*x[1],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Stop when objective function value reaches 1e-4 or below
    /// opt.set_stopval(1e-4).unwrap();
    ///
    /// // This is equivalent to stopping when we're close enough to the origin
    /// ```
    ///
    /// # Notes
    ///
    /// * This is one of several possible stopping criteria that can be set.
    /// * The return value of `optimize` will include `StopValReached` if this
    ///   criterion triggered the termination.
    pub fn set_stopval(&mut self, stopval: f64) -> OptResult {
        result_from_outcome(unsafe { sys::nlopt_set_stopval(self.nloptc_obj.0, stopval) })
    }

    /// Gets the current stopping value for the objective function.
    ///
    /// # Returns
    ///
    /// The current stopval value that has been set.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     1,
    ///     |x, _, _| x[0]*x[0],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Set and then get stopping value
    /// opt.set_stopval(1e-6).unwrap();
    /// assert_eq!(opt.get_stopval(), 1e-6);
    /// ```
    pub fn get_stopval(&self) -> f64 {
        unsafe { sys::nlopt_get_stopval(self.nloptc_obj.0) }
    }

    /// Sets relative tolerance on the objective function value.
    ///
    /// The optimization will stop when an optimization step changes the objective
    /// function value by less than `tolerance` multiplied by the absolute value
    /// of the function value.
    ///
    /// # Parameters
    ///
    /// * `tolerance` - The relative tolerance (must be positive to enable this stopping criterion)
    ///
    /// # Returns
    ///
    /// * `Ok(SuccessState)` if the tolerance was successfully set
    /// * `Err(FailState)` if there was an error
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     2,
    ///     |x, _, _| x[0]*x[0] + x[1]*x[1],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Stop when relative change in function value is less than 1e-4
    /// opt.set_ftol_rel(1e-4).unwrap();
    /// ```
    ///
    /// # Notes
    ///
    /// * This criterion is disabled if `tolerance` is non-positive.
    /// * If the optimal function value might be close to zero, consider also
    ///   setting an absolute tolerance with `set_ftol_abs`.
    /// * The return value of `optimize` will include `FtolReached` if this
    ///   criterion triggered the termination.
    pub fn set_ftol_rel(&mut self, tolerance: f64) -> OptResult {
        result_from_outcome(unsafe { sys::nlopt_set_ftol_rel(self.nloptc_obj.0, tolerance) })
    }

    /// Gets the current relative tolerance on the objective function value.
    ///
    /// # Returns
    ///
    /// * `Some(f64)` with the current relative tolerance if it's been set
    /// * `None` if this stopping criterion is disabled
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     1,
    ///     |x, _, _| x[0]*x[0],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Initially, no relative tolerance is set
    /// assert_eq!(opt.get_ftol_rel(), None);
    ///
    /// // Set and then get tolerance
    /// opt.set_ftol_rel(1e-6).unwrap();
    /// assert_eq!(opt.get_ftol_rel(), Some(1e-6));
    /// ```
    pub fn get_ftol_rel(&self) -> Option<f64> {
        unsafe {
            match sys::nlopt_get_ftol_rel(self.nloptc_obj.0) {
                x if x < 0.0 => None,
                x => Some(x),
            }
        }
    }

    /// Sets absolute tolerance on the objective function value.
    ///
    /// The optimization will stop when an optimization step changes the objective
    /// function value by less than `tolerance` in absolute terms.
    ///
    /// # Parameters
    ///
    /// * `tolerance` - The absolute tolerance (must be positive to enable this stopping criterion)
    ///
    /// # Returns
    ///
    /// * `Ok(SuccessState)` if the tolerance was successfully set
    /// * `Err(FailState)` if there was an error
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     2,
    ///     |x, _, _| x[0]*x[0] + x[1]*x[1],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Stop when absolute change in function value is less than 1e-6
    /// opt.set_ftol_abs(1e-6).unwrap();
    /// ```
    ///
    /// # Notes
    ///
    /// * This criterion is disabled if `tolerance` is non-positive.
    /// * This is useful when the optimal function value might be close to zero,
    ///   where relative tolerances might be insufficient.
    /// * The return value of `optimize` will include `FtolReached` if either this
    ///   criterion or the relative tolerance criterion triggered the termination.
    pub fn set_ftol_abs(&mut self, tolerance: f64) -> OptResult {
        result_from_outcome(unsafe { sys::nlopt_set_ftol_abs(self.nloptc_obj.0, tolerance) })
    }

    /// Gets the current absolute tolerance on the objective function value.
    ///
    /// # Returns
    ///
    /// * `Some(f64)` with the current absolute tolerance if it's been set
    /// * `None` if this stopping criterion is disabled
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     1,
    ///     |x, _, _| x[0]*x[0],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Initially, no absolute tolerance is set
    /// assert_eq!(opt.get_ftol_abs(), None);
    ///
    /// // Set and then get tolerance
    /// opt.set_ftol_abs(1e-8).unwrap();
    /// assert_eq!(opt.get_ftol_abs(), Some(1e-8));
    /// ```
    pub fn get_ftol_abs(&self) -> Option<f64> {
        match unsafe { sys::nlopt_get_ftol_abs(self.nloptc_obj.0) } {
            x if x < 0.0 => None,
            x => Some(x),
        }
    }

    /// Sets relative tolerance on optimization parameters.
    ///
    /// The optimization will stop when an optimization step changes every parameter
    /// by less than `tolerance` multiplied by the absolute value of the parameter.
    ///
    /// # Parameters
    ///
    /// * `tolerance` - The relative tolerance (must be positive to enable this stopping criterion)
    ///
    /// # Returns
    ///
    /// * `Ok(SuccessState)` if the tolerance was successfully set
    /// * `Err(FailState)` if there was an error
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     2,
    ///     |x, _, _| x[0]*x[0] + x[1]*x[1],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Stop when relative change in all parameters is less than 1e-4
    /// opt.set_xtol_rel(1e-4).unwrap();
    /// ```
    ///
    /// # Notes
    ///
    /// * This criterion is disabled if `tolerance` is non-positive.
    /// * If any optimal parameter might be close to zero, consider also
    ///   setting an absolute tolerance with `set_xtol_abs`.
    /// * The return value of `optimize` will include `XtolReached` if this
    ///   criterion triggered the termination.
    pub fn set_xtol_rel(&mut self, tolerance: f64) -> OptResult {
        result_from_outcome(unsafe { sys::nlopt_set_xtol_rel(self.nloptc_obj.0, tolerance) })
    }

    /// Gets the current relative tolerance on optimization parameters.
    ///
    /// # Returns
    ///
    /// * `Some(f64)` with the current relative tolerance if it's been set
    /// * `None` if this stopping criterion is disabled
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     1,
    ///     |x, _, _| x[0]*x[0],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Initially, no relative tolerance is set
    /// assert_eq!(opt.get_xtol_rel(), None);
    ///
    /// // Set and then get tolerance
    /// opt.set_xtol_rel(1e-6).unwrap();
    /// assert_eq!(opt.get_xtol_rel(), Some(1e-6));
    /// ```
    pub fn get_xtol_rel(&self) -> Option<f64> {
        match unsafe { sys::nlopt_get_xtol_rel(self.nloptc_obj.0) } {
            x if x < 0.0 => None,
            x => Some(x),
        }
    }

    /// Sets absolute tolerances on optimization parameters.
    ///
    /// The optimization will stop when an optimization step changes each parameter
    /// `x[i]` by less than the corresponding `tolerance[i]` in absolute terms.
    ///
    /// # Parameters
    ///
    /// * `tolerance` - A slice of absolute tolerances, one for each parameter
    ///
    /// # Returns
    ///
    /// * `Ok(SuccessState)` if the tolerances were successfully set
    /// * `Err(FailState)` if there was an error
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     3,
    ///     |x, _, _| x[0]*x[0] + x[1]*x[1] + x[2]*x[2],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Set different absolute tolerances for each parameter
    /// opt.set_xtol_abs(&[1e-6, 1e-5, 1e-4]).unwrap();
    /// ```
    ///
    /// # Notes
    ///
    /// * This criterion is disabled for parameters with non-positive tolerances.
    /// * This is useful when some optimal parameters might be close to zero.
    /// * The length of the `tolerance` slice must match the problem dimension.
    /// * The return value of `optimize` will include `XtolReached` if either this
    ///   criterion or the relative tolerance criterion triggered the termination.
    pub fn set_xtol_abs(&mut self, tolerance: &[f64]) -> OptResult {
        result_from_outcome(unsafe {
            sys::nlopt_set_xtol_abs(self.nloptc_obj.0, tolerance.as_ptr())
        })
    }

    /// Sets the same absolute tolerance for all optimization parameters.
    ///
    /// This is a convenience method that sets the same absolute tolerance for
    /// all parameters in the optimization problem.
    ///
    /// # Parameters
    ///
    /// * `tolerance` - The absolute tolerance to apply to all parameters
    ///
    /// # Returns
    ///
    /// * `Ok(SuccessState)` if the tolerance was successfully set
    /// * `Err(FailState)` if there was an error
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     2,
    ///     |x, _, _| x[0]*x[0] + x[1]*x[1],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Set absolute tolerance of 1e-6 for all parameters
    /// opt.set_xtol_abs1(1e-6).unwrap();
    /// ```
    ///
    /// # Notes
    ///
    /// * This is equivalent to calling `set_xtol_abs` with a vector where all
    ///   elements are equal to the specified tolerance.
    /// * This criterion is disabled if `tolerance` is non-positive.
    pub fn set_xtol_abs1(&mut self, tolerance: f64) -> OptResult {
        let tol: &[f64] = &vec![tolerance; self.n_dims];
        self.set_xtol_abs(tol)
    }

    /// Gets the current absolute tolerances on optimization parameters.
    ///
    /// # Returns
    ///
    /// * `Some(Vec<f64>)` with the current absolute tolerances if they've been set
    /// * `None` if there was an error retrieving the tolerances
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     2,
    ///     |x, _, _| x[0]*x[0] + x[1]*x[1],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Set and then get absolute tolerances
    /// opt.set_xtol_abs(&[1e-6, 1e-5]).unwrap();
    /// let tols = opt.get_xtol_abs().unwrap();
    /// assert_eq!(tols, vec![1e-6, 1e-5]);
    /// ```
    pub fn get_xtol_abs(&mut self) -> Option<Vec<f64>> {
        let mut tol: Vec<f64> = vec![0.0_f64; self.n_dims];
        let b = tol.as_mut_ptr();
        let res = unsafe { sys::nlopt_get_xtol_abs(self.nloptc_obj.0, b as *mut f64) };
        result_from_outcome(res).ok().map(|_| tol)
    }

    /// Sets a maximum number of objective function evaluations.
    ///
    /// The optimization will stop when the number of function evaluations exceeds
    /// the specified maximum.
    ///
    /// # Parameters
    ///
    /// * `maxeval` - The maximum number of function evaluations (must be positive)
    ///
    /// # Returns
    ///
    /// * `Ok(SuccessState)` if the limit was successfully set
    /// * `Err(FailState)` if there was an error
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     2,
    ///     |x, _, _| x[0]*x[0] + x[1]*x[1],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Stop after at most 1000 function evaluations
    /// opt.set_maxeval(1000).unwrap();
    /// ```
    ///
    /// # Notes
    ///
    /// * This criterion is disabled if `maxeval` is non-positive.
    /// * This is not a strict limit; the actual number of evaluations may
    ///   exceed `maxeval` slightly, depending on the algorithm.
    /// * The return value of `optimize` will include `MaxEvalReached` if this
    ///   criterion triggered the termination.
    pub fn set_maxeval(&mut self, maxeval: u32) -> OptResult {
        result_from_outcome(unsafe { sys::nlopt_set_maxeval(self.nloptc_obj.0, maxeval as i32) })
    }

    /// Gets the current maximum number of function evaluations.
    ///
    /// # Returns
    ///
    /// * `Some(u32)` with the current maximum if it's been set
    /// * `None` if this stopping criterion is disabled
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     1,
    ///     |x, _, _| x[0]*x[0],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Initially, no maximum evaluations are set
    /// assert_eq!(opt.get_maxeval(), None);
    ///
    /// // Set and then get maximum evaluations
    /// opt.set_maxeval(500).unwrap();
    /// assert_eq!(opt.get_maxeval(), Some(500));
    /// ```
    pub fn get_maxeval(&mut self) -> Option<u32> {
        match unsafe { sys::nlopt_get_maxeval(self.nloptc_obj.0) } {
            x if x < 0 => None,
            x => Some(x as u32),
        }
    }

    /// Sets a maximum runtime for the optimization.
    ///
    /// The optimization will stop when the elapsed time exceeds the specified maximum.
    ///
    /// # Parameters
    ///
    /// * `timeout` - The maximum time in seconds (must be positive)
    ///
    /// # Returns
    ///
    /// * `Ok(SuccessState)` if the timeout was successfully set
    /// * `Err(FailState)` if there was an error
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     2,
    ///     |x, _, _| x[0]*x[0] + x[1]*x[1],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Stop after at most 10 seconds
    /// opt.set_maxtime(10.0).unwrap();
    /// ```
    ///
    /// # Notes
    ///
    /// * This criterion is disabled if `timeout` is non-positive.
    /// * This is not a strict limit; the actual time may exceed `timeout`
    ///   slightly, depending on the algorithm and the speed of function evaluations.
    /// * The return value of `optimize` will include `MaxTimeReached` if this
    ///   criterion triggered the termination.
    pub fn set_maxtime(&mut self, timeout: f64) -> OptResult {
        result_from_outcome(unsafe { sys::nlopt_set_maxtime(self.nloptc_obj.0, timeout) })
    }

    /// Gets the current maximum optimization time.
    ///
    /// # Returns
    ///
    /// * `Some(f64)` with the current maximum time in seconds if it's been set
    /// * `None` if this stopping criterion is disabled
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     1,
    ///     |x, _, _| x[0]*x[0],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Initially, no maximum time is set
    /// assert_eq!(opt.get_maxtime(), None);
    ///
    /// // Set and then get maximum time
    /// opt.set_maxtime(5.0).unwrap();
    /// assert_eq!(opt.get_maxtime(), Some(5.0));
    /// ```
    pub fn get_maxtime(&self) -> Option<f64> {
        match unsafe { sys::nlopt_get_maxtime(self.nloptc_obj.0) } {
            x if x < 0.0 => None,
            x => Some(x),
        }
    }

    /// Forces the optimization to stop, either immediately or after a user-specified value.
    ///
    /// This method can be called from within the objective or constraint function
    /// to indicate that the optimization should stop gracefully, returning the
    /// best point found so far.
    ///
    /// # Parameters
    ///
    /// * `stopval` - Optional value to store for retrieval with `get_force_stop`.
    ///   If `None`, stops without setting any value.
    ///
    /// # Returns
    ///
    /// * `Ok(SuccessState)` if the stop request was successfully set
    /// * `Err(FailState)` if there was an error
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    /// use std::sync::atomic::{AtomicBool, Ordering};
    /// use std::sync::Arc;
    ///
    /// // Flag to indicate when to stop
    /// let should_stop = Arc::new(AtomicBool::new(false));
    /// let should_stop_clone = should_stop.clone();
    ///
    /// // Objective function that checks the flag
    /// let obj_func = move |x: &[f64], _: Option<&mut [f64]>, opt: &mut Nlopt<_, _>| {
    ///     // Check if we should stop
    ///     if should_stop_clone.load(Ordering::SeqCst) {
    ///         opt.force_stop(Some(42)).unwrap();
    ///     }
    ///     
    ///     // Normal function calculation
    ///     x.iter().map(|xi| xi * xi).sum()
    /// };
    ///
    /// // In another thread or after some event:
    /// // should_stop.store(true, Ordering::SeqCst);
    /// ```
    ///
    /// # Notes
    ///
    /// * The optimization will finish with a `ForcedStop` error state.
    /// * The force-stop value (if set) can be retrieved using `get_force_stop`.
    /// * Setting `stopval` to `Some(0)` tells NLopt not to force a stop.
    pub fn force_stop(&mut self, stopval: Option<i32>) -> OptResult {
        result_from_outcome(unsafe {
            match stopval {
                Some(x) => sys::nlopt_set_force_stop(self.nloptc_obj.0, x),
                None => sys::nlopt_force_stop(self.nloptc_obj.0),
            }
        })
    }

    /// Gets the most recent force-stop value.
    ///
    /// # Returns
    ///
    /// * `Some(i32)` with the last force-stop value set since the last optimization
    /// * `None` if no force-stop value has been set
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     1,
    ///     |x, _, _| x[0]*x[0],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Initially, no force-stop value is set
    /// assert_eq!(opt.get_force_stop(), None);
    ///
    /// // Set force-stop with a value
    /// opt.force_stop(Some(42)).unwrap();
    /// assert_eq!(opt.get_force_stop(), Some(42));
    /// ```
    pub fn get_force_stop(&mut self) -> Option<i32> {
        match unsafe { sys::nlopt_get_force_stop(self.nloptc_obj.0) } {
            0 => None,
            x => Some(x),
        }
    }

    /// Sets a local optimizer for algorithms that use local optimization.
    ///
    /// Some algorithms, such as MLSL and AUGLAG, use another optimization algorithm
    /// as a subroutine, typically for local optimization. This method allows you
    /// to specify which algorithm to use for local optimization.
    ///
    /// # Parameters
    ///
    /// * `local_opt` - Another `Nlopt` instance specifying the local optimization algorithm
    ///
    /// # Returns
    ///
    /// * `Ok(SuccessState)` if the local optimizer was successfully set
    /// * `Err(FailState)` if there was an error
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create global optimizer using MLSL algorithm
    /// let mut opt = Nlopt::new(
    ///     Algorithm::GMlsl,  // Global optimization algorithm
    ///     2,
    ///     |x, _, _| (x[0]-1.0)*(x[0]-1.0) + (x[1]-2.0)*(x[1]-2.0),
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Create a local optimizer with LBFGS algorithm
    /// let mut local_opt = opt.get_local_optimizer(Algorithm::Lbfgs);
    /// local_opt.set_xtol_rel(1e-4).unwrap();
    ///
    /// // Set the local optimizer for the global optimization
    /// opt.set_local_optimizer(local_opt).unwrap();
    /// ```
    ///
    /// # Notes
    ///
    /// * The dimension of `local_opt` must match the dimension of the main optimizer.
    /// * The objective function, bounds, and constraints of `local_opt` are ignored;
    ///   only its algorithm choice and stopping criteria are used.
    /// * This is only relevant for algorithms that use local optimization as a subroutine.
    pub fn set_local_optimizer(&mut self, local_opt: Nlopt<impl ObjFn<()>, ()>) -> OptResult {
        result_from_outcome(unsafe {
            sys::nlopt_set_local_optimizer(self.nloptc_obj.0, local_opt.nloptc_obj.0)
        })
    }

    /// Creates a new optimizer instance suitable for use as a local optimizer.
    ///
    /// This is a helper method to create a correctly configured `Nlopt` instance
    /// for use with `set_local_optimizer`.
    ///
    /// # Parameters
    ///
    /// * `algorithm` - The algorithm to use for local optimization
    ///
    /// # Returns
    ///
    /// A new `Nlopt` instance configured with the specified algorithm and the
    /// same dimension as the current optimizer.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create global optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::GMlsl,
    ///     2,
    ///     |x, _, _| x[0]*x[0] + x[1]*x[1],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Create and configure a local optimizer
    /// let mut local_opt = opt.get_local_optimizer(Algorithm::Lbfgs);
    /// local_opt.set_ftol_rel(1e-8).unwrap();
    /// local_opt.set_maxeval(1000).unwrap();
    ///
    /// // Set the configured local optimizer
    /// opt.set_local_optimizer(local_opt).unwrap();
    /// ```
    ///
    /// # Notes
    ///
    /// * The returned optimizer will have the same dimension as the current optimizer.
    /// * It has a dummy objective function because the objective function will be
    ///   supplied by the main optimizer when used as a local optimizer.
    pub fn get_local_optimizer(&mut self, algorithm: Algorithm) -> Nlopt<impl ObjFn<()>, ()> {
        fn stub_opt(_: &[f64], _: Option<&mut [f64]>, _: &mut ()) -> f64 {
            unreachable!()
        }
        // create a new object based on former one
        Nlopt::new(algorithm, self.n_dims, stub_opt, self.target, ())
    }

    /// Sets the population size for stochastic optimization algorithms.
    ///
    /// Several stochastic search algorithms (e.g., CRS, MLSL, and ISRES) start by
    /// generating an initial "population" of random points. This method allows you
    /// to set the size of that population.
    ///
    /// # Parameters
    ///
    /// * `population` - The population size to use
    ///
    /// # Returns
    ///
    /// * `Ok(SuccessState)` if the population size was successfully set
    /// * `Err(FailState)` if there was an error
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create optimizer with a stochastic algorithm
    /// let mut opt = Nlopt::new(
    ///     Algorithm::GnCrs2Lm,
    ///     2,
    ///     |x, _, _| x[0]*x[0] + x[1]*x[1],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Set initial population size to 50
    /// opt.set_population(50).unwrap();
    /// ```
    ///
    /// # Notes
    ///
    /// * A population size of zero uses the algorithm's default heuristic.
    /// * This setting has no effect for deterministic algorithms.
    /// * Larger population sizes may improve the quality of the result at the
    ///   expense of more function evaluations.
    pub fn set_population(&mut self, population: usize) -> OptResult {
        result_from_outcome(unsafe {
            sys::nlopt_set_population(self.nloptc_obj.0, population as u32)
        })
    }

    /// Gets the current population size for stochastic optimization algorithms.
    ///
    /// # Returns
    ///
    /// The current population size setting.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create optimizer with a stochastic algorithm
    /// let mut opt = Nlopt::new(
    ///     Algorithm::GnCrs2Lm,
    ///     2,
    ///     |x, _, _| x[0]*x[0] + x[1]*x[1],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Set and then get population size
    /// opt.set_population(50).unwrap();
    /// assert_eq!(opt.get_population(), 50);
    /// ```
    ///
    /// # Notes
    ///
    /// * A return value of 0 indicates that the default population size will be used.
    pub fn get_population(&mut self) -> usize {
        unsafe { sys::nlopt_get_population(self.nloptc_obj.0) as usize }
    }

    /// Sets the seed for the random number generator used by stochastic algorithms.
    ///
    /// For stochastic optimization algorithms, this method allows setting the seed
    /// for the pseudorandom number generator to ensure reproducible results.
    ///
    /// # Parameters
    ///
    /// * `seed` - Optional seed value. If `None`, the seed is reset based on the system time.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::Nlopt;
    ///
    /// // Set a specific seed for reproducible results
    /// Nlopt::srand_seed(Some(42));
    ///
    /// // Reset to system time (different results each run)
    /// Nlopt::srand_seed(None);
    /// ```
    ///
    /// # Notes
    ///
    /// * This is a static method that affects all NLopt optimizers.
    /// * Using a fixed seed ensures that you get the same sequence of
    ///   pseudorandom numbers each time your program runs, which can be
    ///   useful for debugging and reproducibility.
    /// * Using `None` resets to system time, giving different results each run.
    pub fn srand_seed(seed: Option<u64>) {
        unsafe {
            match seed {
                None => sys::nlopt_srand_time(),
                Some(x) => sys::nlopt_srand(x as c_ulong),
            }
        }
    }

    /// Sets the vector storage for limited-memory quasi-Newton algorithms.
    ///
    /// Some of the NLopt algorithms are limited-memory "quasi-Newton" algorithms,
    /// which "remember" the gradients from a finite number M of the previous steps
    /// to construct an approximate Hessian matrix. This method allows you to control
    /// the value of M, which affects both memory usage and convergence.
    ///
    /// # Parameters
    ///
    /// * `m` - Optional storage size. If `None`, a heuristic value is used.
    ///
    /// # Returns
    ///
    /// * `Ok(SuccessState)` if the storage size was successfully set
    /// * `Err(FailState)` if there was an error
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create optimizer with a quasi-Newton algorithm
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Lbfgs,
    ///     2,
    ///     |x, grad, _| {
    ///         if let Some(grad) = grad {
    ///             grad[0] = 2.0 * x[0];
    ///             grad[1] = 2.0 * x[1];
    ///         }
    ///         x[0]*x[0] + x[1]*x[1]
    ///     },
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Set storage for 20 previous gradients
    /// opt.set_vector_storage(Some(20)).unwrap();
    ///
    /// // Use heuristic storage size
    /// opt.set_vector_storage(None).unwrap();
    /// ```
    ///
    /// # Notes
    ///
    /// * This setting only affects limited-memory quasi-Newton algorithms
    ///   like L-BFGS.
    /// * Larger M values require more storage but may lead to faster convergence.
    /// * The default heuristic typically sets M to 10 or more, depending on
    ///   available memory.
    pub fn set_vector_storage(&mut self, m: Option<usize>) -> OptResult {
        let outcome = match m {
            None => unsafe { sys::nlopt_set_vector_storage(self.nloptc_obj.0, 0_u32) },
            Some(x) => unsafe { sys::nlopt_set_vector_storage(self.nloptc_obj.0, x as u32) },
        };
        result_from_outcome(outcome)
    }

    /// Gets the current vector storage size for limited-memory quasi-Newton algorithms.
    ///
    /// # Returns
    ///
    /// The current vector storage size setting.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create optimizer with a quasi-Newton algorithm
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Lbfgs,
    ///     2,
    ///     |x, _, _| x[0]*x[0] + x[1]*x[1],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Set and then get storage size
    /// opt.set_vector_storage(Some(15)).unwrap();
    /// assert_eq!(opt.get_vector_storage(), 15);
    /// ```
    ///
    /// # Notes
    ///
    /// * A return value of 0 indicates that the default heuristic is being used.
    pub fn get_vector_storage(&mut self) -> usize {
        unsafe { sys::nlopt_get_vector_storage(self.nloptc_obj.0) as usize }
    }

    /// Sets the initial step size for derivative-free algorithms.
    ///
    /// For derivative-free local optimization algorithms, this method specifies
    /// the initial step size that the optimizer will use to perturb the variables
    /// when it begins the optimization.
    ///
    /// # Parameters
    ///
    /// * `dx` - A slice specifying the initial step size for each variable
    ///
    /// # Returns
    ///
    /// * `Ok(SuccessState)` if the step size was successfully set
    /// * `Err(FailState)` if there was an error
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create optimizer with a derivative-free algorithm
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     3,
    ///     |x, _, _| x[0]*x[0] + x[1]*x[1] + x[2]*x[2],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Set different initial step sizes for each variable
    /// opt.set_initial_step(&[0.1, 0.5, 1.0]).unwrap();
    /// ```
    ///
    /// # Notes
    ///
    /// * This setting only affects derivative-free algorithms.
    /// * The step size should be large enough to cause a significant change
    ///   in the objective function, but not so large that you miss the local
    ///   optimum nearest the starting point.
    /// * If not set, NLopt chooses the initial step size heuristically based
    ///   on the bounds and other information.
    pub fn set_initial_step(&mut self, dx: &[f64]) -> OptResult {
        result_from_outcome(unsafe { sys::nlopt_set_initial_step(self.nloptc_obj.0, dx.as_ptr()) })
    }

    /// Sets the same initial step size for all variables.
    ///
    /// This is a convenience method that sets the same initial step size for
    /// all variables in the optimization problem.
    ///
    /// # Parameters
    ///
    /// * `dx` - The initial step size to use for all variables
    ///
    /// # Returns
    ///
    /// * `Ok(SuccessState)` if the step size was successfully set
    /// * `Err(FailState)` if there was an error
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create optimizer with a derivative-free algorithm
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     2,
    ///     |x, _, _| x[0]*x[0] + x[1]*x[1],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Set an initial step size of 0.1 for all variables
    /// opt.set_initial_step1(0.1).unwrap();
    /// ```
    ///
    /// # Notes
    ///
    /// * This is equivalent to calling `set_initial_step` with a vector where
    ///   all elements are equal to the specified step size.
    pub fn set_initial_step1(&mut self, dx: f64) -> OptResult {
        let d: &[f64] = &vec![dx; self.n_dims];
        self.set_initial_step(d)
    }

    /// Gets the current initial step sizes.
    ///
    /// This method returns the heuristic initial step sizes that NLopt would use
    /// given the specified starting point `x`.
    ///
    /// # Parameters
    ///
    /// * `x` - The starting point for optimization
    ///
    /// # Returns
    ///
    /// * `Some(Vec<f64>)` with the initial step sizes if successful
    /// * `None` if there was an error
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create optimizer
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     2,
    ///     |x, _, _| x[0]*x[0] + x[1]*x[1],
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Get the initial step sizes for starting point [1.0, 2.0]
    /// let steps = opt.get_initial_step(&[1.0, 2.0]).unwrap();
    /// println!("Initial steps: {:?}", steps);
    /// ```
    ///
    /// # Notes
    ///
    /// * If the initial step was not explicitly set with `set_initial_step`,
    ///   NLopt uses a heuristic that may depend on the starting point `x`.
    /// * The length of the `x` slice must match the problem dimension.
    pub fn get_initial_step(&mut self, x: &[f64]) -> Option<Vec<f64>> {
        let mut dx: Vec<f64> = vec![0.0_f64; self.n_dims];
        let b = dx.as_mut_ptr();
        let res =
            unsafe { sys::nlopt_get_initial_step(self.nloptc_obj.0, x.as_ptr(), b as *mut f64) };
        result_from_outcome(res).ok().map(|_| dx)
    }

    /// Gets the NLopt library version.
    ///
    /// This method returns the version numbers of the underlying NLopt library.
    ///
    /// # Returns
    ///
    /// A tuple containing the major, minor, and bugfix version numbers.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::Nlopt;
    ///
    /// // Get the NLopt version
    /// let (major, minor, bugfix) = Nlopt::version();
    /// println!("NLopt version {}.{}.{}", major, minor, bugfix);
    /// ```
    pub fn version() -> (i32, i32, i32) {
        unsafe {
            let mut i: i32 = 0;
            let mut j: i32 = 0;
            let mut k: i32 = 0;
            sys::nlopt_version(&mut i, &mut j, &mut k);
            (i, j, k)
        }
    }

    /// Runs the optimization from a given starting point.
    ///
    /// This is the main method that performs the optimization. It starts from
    /// the given initial guess and attempts to find an optimal solution according
    /// to the specified algorithm, objective function, and constraints.
    ///
    /// # Parameters
    ///
    /// * `x_init` - A mutable slice containing the initial guess, which will be
    ///   replaced with the optimized values upon successful completion
    ///
    /// # Returns
    ///
    /// * `Ok((SuccessState, f64))` with the success status and optimal function value
    /// * `Err((FailState, f64))` with the error status and the best function value found
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use nlopt::{Nlopt, Algorithm, Target};
    ///
    /// // Create optimizer for minimizing (x - 3)² + 7
    /// let objective = |x: &[f64], _: Option<&mut [f64]>, _: &mut ()| {
    ///     (x[0] - 3.0).powi(2) + 7.0
    /// };
    ///
    /// let mut opt = Nlopt::new(
    ///     Algorithm::Cobyla,
    ///     1,
    ///     objective,
    ///     Target::Minimize,
    ///     ()
    /// );
    ///
    /// // Set optimization parameters
    /// opt.set_xtol_rel(1e-4).unwrap();
    ///
    /// // Run optimization from starting point [0.0]
    /// let mut x = vec![0.0];
    /// match opt.optimize(&mut x) {
    ///     Ok((status, value)) => {
    ///         println!("Success: {:?}", status);
    ///         println!("Optimum at x = {:?}", x);
    ///         println!("Minimum value = {}", value);
    ///     },
    ///     Err((error, value)) => {
    ///         println!("Optimization failed: {:?}", error);
    ///         println!("Best point found: x = {:?}", x);
    ///         println!("Function value = {}", value);
    ///     }
    /// }
    /// ```
    ///
    /// # Notes
    ///
    /// * The length of the `x_init` slice must match the problem dimension.
    /// * Upon success, the SuccessState indicates which stopping criterion
    ///   triggered the termination.
    /// * Even if the optimization fails, `x_init` will contain the best point
    ///   found so far.
    /// * The second value in the result tuple is always the function value at
    ///   the final point, regardless of success or failure.
    pub fn optimize(&self, x_init: &mut [f64]) -> Result<(SuccessState, f64), (FailState, f64)> {
        let mut min_value: f64 = 0.0;
        let res =
            unsafe { sys::nlopt_optimize(self.nloptc_obj.0, x_init.as_mut_ptr(), &mut min_value) };
        result_from_outcome(res)
            .map(|s| (s, min_value))
            .map_err(|e| (e, min_value))
    }
}

/// Helper function to calculate gradient of function numerically.
///
/// This function approximates the gradient of a function using finite differences.
/// It's useful when gradient information is required by the optimization algorithm
/// but an analytical gradient formula is not available.
///
/// # Parameters
///
/// * `x0` - The point at which to calculate the gradient
/// * `f` - The function for which to calculate the gradient
/// * `grad` - A mutable slice to store the calculated gradient
///
/// # Examples
///
/// ```rust,no_run
/// use nlopt::{approximate_gradient, Nlopt, Algorithm, Target};
///
/// // Define a function for which we don't have an analytical gradient
/// fn complex_function(x: &[f64]) -> f64 {
///     let x2 = x[0] * x[0];
///     let y2 = x[1] * x[1];
///     (x2 * y2).sqrt() + (x[0] - x[1]).powi(2)
/// }
///
/// // Use this function with NLopt and approximate_gradient
/// let objective = |x: &[f64], grad: Option<&mut [f64]>, _: &mut ()| {
///     let val = complex_function(x);
///     
///     // If gradient is requested, calculate it numerically
///     if let Some(g) = grad {
///         approximate_gradient(x, complex_function, g);
///     }
///     
///     val
/// };
///
/// let mut opt = Nlopt::new(
///     Algorithm::Lbfgs,  // Algorithm requiring gradient
///     2,
///     objective,
///     Target::Minimize,
///     ()
/// );
/// ```
///
/// # Notes
///
/// * This function uses a central difference formula with a step size based on
///   the square root of the machine epsilon.
/// * For better performance with gradient-based algorithms, consider:
///   1. Providing an analytical gradient when possible
///   2. Using automatic differentiation libraries
///   3. Using gradient-free algorithms when appropriate
pub fn approximate_gradient<F>(x0: &[f64], f: F, grad: &mut [f64])
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    let mut x0 = x0.to_vec();
    let eps = f64::EPSILON.powf(1.0 / 3.0);
    for i in 0..n {
        let x0i = x0[i];
        x0[i] = x0i - eps;
        let fl = f(&x0);
        x0[i] = x0i + eps;
        let fh = f(&x0);
        grad[i] = (fh - fl) / (2.0 * eps);
        x0[i] = x0i;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_approx_gradient() {
        fn square(x: &[f64]) -> f64 {
            x.iter().map(|v| v * v).sum()
        }

        let x0 = vec![0., 1., 6.];
        let mut grad = vec![0.; 3];
        approximate_gradient(&x0, square, &mut grad);
        let expect = vec![0., 2., 12.];
        for (r, e) in grad.iter().zip(expect.iter()) {
            assert!((r - e).abs() < 0.0000001)
        }
    }

    #[test]
    fn test_user_data() {
        fn objfn(x: &[f64], grad: Option<&mut [f64]>, coefs: &mut (f64, Vec<f64>)) -> f64 {
            assert!(grad.is_none());
            let x = x[0];
            coefs.0 * x * x + coefs.1[0] * x + coefs.1[1]
        }

        let mut params = vec![1.];
        let userdata = (1.0, vec![-2., 3.]);
        let mut opt = Nlopt::<_, (f64, Vec<f64>)>::new(
            Algorithm::Bobyqa,
            2,
            objfn,
            Target::Minimize,
            userdata,
        );
        opt.set_xtol_rel(1e-8).unwrap();
        let res = opt.optimize(&mut params).unwrap();
        assert_eq!(res.1, 2.0);
        assert_eq!(params, vec![1.0]);
    }

    #[test]
    fn test_lbfgs() {
        // Taken from `nloptr` docs
        fn flb(x: &[f64]) -> f64 {
            let p = x.len();
            Some(1.0)
                .into_iter()
                .chain(std::iter::repeat(4.0))
                .take(p)
                .zip(
                    x.iter()
                        .zip(Some(1.0).into_iter().chain(x.iter().cloned()).take(p))
                        .map(|(x, xd)| (x - xd.powi(2)).powi(2)),
                )
                .map(|(fst, snd)| fst * snd)
                .sum()
        }

        fn objfn(x: &[f64], grad: Option<&mut [f64]>, _user_data: &mut ()) -> f64 {
            let grad = grad.unwrap();
            approximate_gradient(x, flb, grad);
            flb(x)
        }

        let mut opt = Nlopt::<_, ()>::new(Algorithm::Lbfgs, 25, objfn, Target::Minimize, ());

        let mut x0 = vec![3.0; 25];
        let xl = vec![2.0; 25];
        let xu = vec![4.0; 25];

        // check bounds unset
        assert!(opt
            .get_upper_bounds()
            .unwrap()
            .into_iter()
            .all(|v| v == f64::INFINITY));
        assert!(opt
            .get_lower_bounds()
            .unwrap()
            .into_iter()
            .all(|v| v == f64::NEG_INFINITY));

        // set the bounds
        opt.set_upper_bounds(&xu).unwrap();
        opt.set_lower_bounds(&xl).unwrap();
        // set it twice, why not
        opt.set_lower_bounds(&xl).unwrap();

        // check the bounds were set
        assert_eq!(opt.get_upper_bounds().unwrap(), xu);
        assert_eq!(opt.get_lower_bounds().unwrap(), xl);

        opt.set_xtol_rel(1e-8).unwrap();
        let mut expect = vec![2.0; 25];
        expect[23] = 2.1090933511928247;
        expect[24] = 4.;

        match opt.optimize(&mut x0) {
            Ok((_, v)) => assert_eq!(v, 368.10591287433397),
            Err((e, _)) => panic!("{:?}", e),
        }
        assert_eq!(&x0, &expect);
    }

    #[test]
    fn test_praxis_closure() {
        fn objfn(x: &[f64]) -> f64 {
            // max = 4 at x0 = 1, x1 = 0
            ((x[0] - 1.) * (x[0] - 1.) + x[1] * x[1]) * -1. + 4.
        }

        let opt = Nlopt::new(
            Algorithm::Praxis,
            2,
            |x: &[f64], _: Option<&mut [f64]>, _: &mut ()| objfn(x),
            Target::Maximize,
            (),
        );
        assert_eq!(
            opt.get_upper_bounds().unwrap(),
            &[f64::INFINITY, f64::INFINITY]
        );
        let mut input = vec![3.0, -2.0];
        let (_s, val) = opt.optimize(&mut input).unwrap();
        assert_eq!(val, 4.);
        assert_eq!(input, &[1., 0.]);
    }

    #[test]
    fn test_auglag() {
        // Test adapted from nloptr docs

        fn objfn(x: &[f64], _grad: Option<&mut [f64]>, _user_data: &mut ()) -> f64 {
            (x[0] - 2.0).powi(2) + (x[1] - 1.0).powi(2)
        }

        fn hin(x: &[f64], _grad: Option<&mut [f64]>, _user_data: &mut ()) -> f64 {
            0.25 * x[0].powi(2) + x[1].powi(2) - 1.
        }

        fn heq(x: &[f64], _grad: Option<&mut [f64]>, _user_data: &mut ()) -> f64 {
            x[0] - 2.0 * x[1] + 1.
        }

        let mut opt = Nlopt::new(Algorithm::Auglag, 2, objfn, Target::Minimize, ());
        opt.add_inequality_constraint(hin, (), 1e-6).unwrap();
        opt.add_equality_constraint(heq, (), 1e-6).unwrap();
        opt.set_xtol_rel(1e-6).unwrap();

        assert_eq!(
            opt.get_upper_bounds().unwrap(),
            &[f64::INFINITY, f64::INFINITY]
        );
        assert_eq!(
            opt.get_lower_bounds().unwrap(),
            &[f64::NEG_INFINITY, f64::NEG_INFINITY]
        );

        let mut local_opt = opt.get_local_optimizer(Algorithm::Cobyla);
        local_opt.set_xtol_rel(1e-6).unwrap();
        opt.set_local_optimizer(local_opt).unwrap();

        let mut input = vec![1., 1.];
        let (_s, v) = opt.optimize(&mut input).unwrap();
        assert_abs_diff_eq!(v, 1.3934640682303436, epsilon = 1e-6);
        let expected = vec![0.8228760595426139, 0.9114376093794901];
        assert_abs_diff_eq!(expected.as_slice(), input.as_slice(), epsilon = 1e-6);
    }

    #[test]
    fn test_auglag_mconstraint() {
        fn objfn(x: &[f64], _grad: Option<&mut [f64]>, _user_data: &mut ()) -> f64 {
            (x[0] - 2.0).powi(2) + (x[1] - 1.0).powi(2)
        }

        fn hin(x: &[f64]) -> f64 {
            0.25 * x[0].powi(2) + x[1].powi(2) - 1.
        }

        fn heq(x: &[f64]) -> f64 {
            x[0] - 2.0 * x[1] + 1.
        }

        fn m_ineq_constraint(result: &mut [f64], x: &[f64]) {
            result[0] = hin(x);
            // shouldn't affect the optimization, will always be < hin1
            result[1] = hin(x) - 1.0;
            result[2] = hin(x) - 2.0;
        }

        fn m_eq_constraint(result: &mut [f64], x: &[f64]) {
            result[0] = heq(x);
        }

        let mut opt = Nlopt::new(Algorithm::Auglag, 2, objfn, Target::Minimize, ());

        opt.add_inequality_mconstraint(
            3,
            |r: &mut [f64], x: &[f64], _: Option<&mut [f64]>, _: &mut ()| m_ineq_constraint(r, x),
            (),
            &[1e-6; 3],
        )
        .unwrap();

        // TODO if we use two eq constraints, it doesn't converge *shrug*
        opt.add_equality_mconstraint(
            1,
            |r: &mut [f64], x: &[f64], _: Option<&mut [f64]>, _: &mut ()| m_eq_constraint(r, x),
            (),
            &[1e-6; 1],
        )
        .unwrap();

        opt.set_xtol_rel(1e-6).unwrap();

        let mut local_opt = opt.get_local_optimizer(Algorithm::Bobyqa);

        assert_eq!(local_opt.get_xtol_rel().unwrap(), 0.0);
        local_opt.set_xtol_rel(1e-6).unwrap();
        assert_eq!(local_opt.get_xtol_rel().unwrap(), 1e-6);

        opt.set_local_optimizer(local_opt).unwrap();

        let mut input = vec![1., 1.];
        let (_s, v) = opt.optimize(&mut input).unwrap();
        assert_eq!(v, 1.3934648637383988); // don't really know if this is the right answer...
    }

    #[test]
    fn test_cobyla_with_no_memory_leaks() {
        use std::cell::Cell;
        use std::rc::Rc;
        fn objfn(x: &[f64], _grad: Option<&mut [f64]>, call_ct: &mut Rc<Cell<u32>>) -> f64 {
            let v: u32 = call_ct.get();
            call_ct.set(v + 1);
            let x = x[0];
            (x - 3.0) * (x - 3.0) + 7.0 // min = 7 @ x = 3
        }
        let user_data = Rc::new(Cell::new(0));
        let mut nlopt = Nlopt::new(
            Algorithm::Cobyla,
            1,
            objfn,
            Target::Minimize,
            user_data.clone(),
        );

        assert_eq!(nlopt.get_xtol_abs().unwrap(), &[0.0]);
        nlopt.set_xtol_abs1(1e-15).unwrap();
        assert_eq!(nlopt.get_xtol_abs().unwrap(), &[1e-15]);

        let mut input = [0.0];
        let (_s, v) = nlopt.optimize(&mut input).unwrap();
        assert_eq!(input[0], 3.0);
        assert_eq!(v, 7.0);
        // when we drop the object, destructor should run as normal
        // and refcount is decremented
        assert_eq!(Rc::strong_count(&user_data), 2);
        drop(nlopt);
        assert_eq!(Rc::strong_count(&user_data), 1);
    }

    #[test]
    fn test_neldermead_recover_user_data() {
        struct WrapU32(u32);
        impl Drop for WrapU32 {
            fn drop(&mut self) {
                panic!("undroppable")
            }
        }
        fn objfn(x: &[f64], _grad: Option<&mut [f64]>, call_ct: &mut WrapU32) -> f64 {
            call_ct.0 += 1;
            let x = x[0];
            (x - 5.0) * (x - 5.0) + 9.0 // min = 9 @ x = 5
        }
        let mut nlopt = Nlopt::new(
            Algorithm::Neldermead,
            1,
            objfn,
            Target::Minimize,
            WrapU32(0),
        );
        nlopt.set_xtol_abs1(1e-15).unwrap();
        let mut input = [0.0];
        let (_s, v) = nlopt.optimize(&mut input).unwrap();
        assert_eq!(input[0], 4.999999970199497);
        assert_eq!(v, 9.0);
        let call_ct = nlopt.recover_user_data();
        assert_eq!(call_ct.0, 101);
        std::mem::forget(call_ct);
    }
}
