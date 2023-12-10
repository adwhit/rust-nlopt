//! # nlopt
//!
//! This is a wrapper for `nlopt`, a C library of useful optimization
//! algorithms. For details of the various algorithms,
//! consult the [nlopt docs](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/)

use std::os::raw::{c_uint, c_ulong, c_void};
use std::slice;

use self::nlopt_sys as sys;

#[allow(non_camel_case_types)]
#[allow(non_upper_case_globals)]
#[allow(dead_code)]
mod nlopt_sys;

/// Target object function state
#[derive(Debug, Clone, Copy)]
pub enum Target {
    Maximize,
    Minimize,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum Algorithm {
    Direct = sys::nlopt_algorithm_NLOPT_GN_DIRECT,
    DirectL = sys::nlopt_algorithm_NLOPT_GN_DIRECT_L,
    DirectLRand = sys::nlopt_algorithm_NLOPT_GN_DIRECT_L_RAND,
    DirectNoscal = sys::nlopt_algorithm_NLOPT_GN_DIRECT_NOSCAL,
    DirectLNoscal = sys::nlopt_algorithm_NLOPT_GN_DIRECT_L_NOSCAL,
    DirectLRandNoscal = sys::nlopt_algorithm_NLOPT_GN_DIRECT_L_RAND_NOSCAL,
    OrigDirect = sys::nlopt_algorithm_NLOPT_GN_ORIG_DIRECT,
    OrigDirectL = sys::nlopt_algorithm_NLOPT_GN_ORIG_DIRECT_L,
    Crs2Lm = sys::nlopt_algorithm_NLOPT_GN_CRS2_LM,

    GMlsl = sys::nlopt_algorithm_NLOPT_G_MLSL,
    GMlslLds = sys::nlopt_algorithm_NLOPT_G_MLSL_LDS,
    GnMlsl = sys::nlopt_algorithm_NLOPT_GN_MLSL,
    GdMlsl = sys::nlopt_algorithm_NLOPT_GD_MLSL,
    GnMlslLds = sys::nlopt_algorithm_NLOPT_GN_MLSL_LDS,
    GdMlslLds = sys::nlopt_algorithm_NLOPT_GD_MLSL_LDS,

    // TODO I think these only exist when compiled with
    // C++ (which currently is disabled by build.rs)
    // so ... check and possibly disable?
    StoGo = sys::nlopt_algorithm_NLOPT_GD_STOGO,
    StoGoRand = sys::nlopt_algorithm_NLOPT_GD_STOGO_RAND,
    Isres = sys::nlopt_algorithm_NLOPT_GN_ISRES,
    Esch = sys::nlopt_algorithm_NLOPT_GN_ESCH,

    Cobyla = sys::nlopt_algorithm_NLOPT_LN_COBYLA,
    Bobyqa = sys::nlopt_algorithm_NLOPT_LN_BOBYQA,
    Newuoa = sys::nlopt_algorithm_NLOPT_LN_NEWUOA,
    NewuoaBound = sys::nlopt_algorithm_NLOPT_LN_NEWUOA_BOUND,
    Praxis = sys::nlopt_algorithm_NLOPT_LN_PRAXIS,
    Neldermead = sys::nlopt_algorithm_NLOPT_LN_NELDERMEAD,
    Sbplx = sys::nlopt_algorithm_NLOPT_LN_SBPLX,

    Mma = sys::nlopt_algorithm_NLOPT_LD_MMA,
    Slsqp = sys::nlopt_algorithm_NLOPT_LD_SLSQP,
    Lbfgs = sys::nlopt_algorithm_NLOPT_LD_LBFGS,
    LbfgsNocedal = sys::nlopt_algorithm_NLOPT_LD_LBFGS_NOCEDAL,

    LdVar1 = sys::nlopt_algorithm_NLOPT_LD_VAR1,
    LdVar2 = sys::nlopt_algorithm_NLOPT_LD_VAR2,

    TNewton = sys::nlopt_algorithm_NLOPT_LD_TNEWTON,
    TNewtonRestart = sys::nlopt_algorithm_NLOPT_LD_TNEWTON_RESTART,
    TNewtonPrecond = sys::nlopt_algorithm_NLOPT_LD_TNEWTON_PRECOND,
    TNewtonPrecondRestart = sys::nlopt_algorithm_NLOPT_LD_TNEWTON_PRECOND_RESTART,

    Auglag = sys::nlopt_algorithm_NLOPT_AUGLAG,
    AuglagEq = sys::nlopt_algorithm_NLOPT_AUGLAG_EQ,
    LnAuglag = sys::nlopt_algorithm_NLOPT_LN_AUGLAG,
    LdAuglagEq = sys::nlopt_algorithm_NLOPT_LD_AUGLAG_EQ,
    LdAuglag = sys::nlopt_algorithm_NLOPT_LD_AUGLAG,
    LnAuglagEq = sys::nlopt_algorithm_NLOPT_LN_AUGLAG_EQ,

    Ccsaq = sys::nlopt_algorithm_NLOPT_LD_CCSAQ,

    Ags = sys::nlopt_algorithm_NLOPT_GN_AGS,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum FailState {
    Failure = sys::nlopt_result_NLOPT_FAILURE,
    InvalidArgs = sys::nlopt_result_NLOPT_INVALID_ARGS,
    OutOfMemory = sys::nlopt_result_NLOPT_OUT_OF_MEMORY,
    RoundoffLimited = sys::nlopt_result_NLOPT_ROUNDOFF_LIMITED,
    ForcedStop = sys::nlopt_result_NLOPT_FORCED_STOP,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum SuccessState {
    Success = sys::nlopt_result_NLOPT_SUCCESS,
    StopValReached = sys::nlopt_result_NLOPT_STOPVAL_REACHED,
    FtolReached = sys::nlopt_result_NLOPT_FTOL_REACHED,
    XtolReached = sys::nlopt_result_NLOPT_XTOL_REACHED,
    MaxEvalReached = sys::nlopt_result_NLOPT_MAXEVAL_REACHED,
    MaxTimeReached = sys::nlopt_result_NLOPT_MAXTIME_REACHED,
}

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
/// An objective function takes the form of a closure `f(x: &[f64], gradient: Option<&mut [f64], user_data: &mut U) -> f64`
///
/// * `x` - `n`-dimensional array
/// * `gradient` - `n`-dimensional array to store the gradient `grad f(x)`. If `gradient` matches
/// `Some(x)`, the user is required to provide a gradient, otherwise the optimization will
/// probabely fail.
/// * `user_data` - user defined data
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
/// A multi-objective function takes the form of a closure `f(result: &mut [f64], x: &[f64], gradient: Option<&mut [f64], user_data: &mut U)`
///
/// * `result` - `m`-dimensional array to store the value `f(x)`
/// * `x` - `n`-dimensional array
/// * `gradient` - `n×m`-diconstraint array to store the gradient `grad f(x)`. The n dimension of
/// gradient is stored contiguously, so that `df_i / dx_j` is stored in `gradient[i*n + j]`. If
/// `gradient` is `Some(x)`, the user is required to return a valid gradient, otherwise the
/// optimization will most likely fail.
/// * `user_data` - user defined data
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

/// This is the central ```struct``` of this library. It represents an optimization of a given
/// function, called the objective function. The argument `x` to this function is an
/// `n`-dimensional double-precision vector. The dimensions are set at creation of the struct and
/// cannot be changed afterwards. NLopt offers different optimization algorithms. One must be
/// chosen at struct creation and cannot be changed afterwards. Always use ```Nlopt::<T>::new()``` to create an `Nlopt` struct.
pub struct Nlopt<F: ObjFn<T>, T> {
    algorithm: Algorithm,
    n_dims: usize,
    target: Target,
    nloptc_obj: WrapSysNlopt,
    func_cfg: Box<FunctionCfg<F, T>>,
}

impl<F: ObjFn<T>, T> Nlopt<F, T> {
    /// Creates a new `Nlopt` struct.
    ///
    /// * `algorithm` - Which optimization algorithm to use. This cannot be changed after creation
    /// of the struct
    /// * `n_dims` - Dimension of the argument to the objective function
    /// * `objective_fn` - This function has the signature `(&[f64],
    /// Option<&mut [f64]>, T) -> f64`. The first argument is the vector `x` passed to the function.
    /// The second argument is `Some(&mut [f64])` if the calling optimization algorithm needs
    /// the gradient of the function. If the gradient is not needed, it is `None`. The last
    /// argument is the user data provided beforehand using the `user_data` argument to the
    /// constructor.
    /// * `target` - Whether to minimize or maximize the objective function
    /// * `user_data` - Optional data that is passed to the objective function
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

    /// Retrive chosen algorithm
    pub fn get_algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// Consume the struct and recover the user data that was passed into
    /// the constructor.
    ///
    /// Useful if you wish to collect information during
    /// the optimization process - for example, pass in some kind of `Statistics`
    /// object, mutate it inside the objective function, then recover it.
    pub fn recover_user_data(self) -> T {
        self.func_cfg.user_data
    }

    /// Most of the algorithms in NLopt are designed for minimization of functions with simple bound
    /// constraints on the inputs. That is, the input vectors `x` are constrainted to lie in a
    /// hyperrectangle `lower_bound[i] ≤ x[i] ≤ upper_bound[i] for 0 ≤ i < n`.
    /// NLopt guarantees that your objective
    /// function and any nonlinear constraints will never be evaluated outside of these bounds
    /// (unlike nonlinear constraints, which may be violated at intermediate steps).
    ///
    /// These bounds are specified by passing an array `bound` of length `n` (the dimension of the
    /// problem) to one or both of the functions:
    ///
    /// `set_lower_bounds(&[f64])`
    ///
    /// `set_upper_bounds(&[f64])`
    ///
    /// If a lower/upper bound is not set, the default is no bound (unconstrained, i.e. a bound of
    /// infinity); it is possible to have lower bounds but not upper bounds or vice versa.
    /// Alternatively, the user can call one of the above functions and explicitly pass a lower
    /// bound of `-INFINITY` and/or an upper bound of `+INFINITY` for some optimization parameters to
    /// make them have no lower/upper bound, respectively.
    ///
    /// It is permitted to set `lower_bound[i] == upper_bound[i]` in one or more dimensions;
    /// this is equivalent to fixing the corresponding `x[i]` parameter, eliminating it
    /// from the optimization.
    ///
    /// Note, however, that some of the algorithms in NLopt, in particular most of the
    /// global-optimization algorithms, do not support unconstrained optimization and will return an
    /// error in `optimize` if you do not supply finite lower and upper bounds.
    pub fn set_lower_bounds(&mut self, bound: &[f64]) -> OptResult {
        result_from_outcome(unsafe {
            sys::nlopt_set_lower_bounds(self.nloptc_obj.0, bound.as_ptr())
        })
    }

    /// See documentation for `set_lower_bounds`
    pub fn set_upper_bounds(&mut self, bound: &[f64]) -> OptResult {
        result_from_outcome(unsafe {
            sys::nlopt_set_upper_bounds(self.nloptc_obj.0, bound.as_ptr())
        })
    }

    /// For convenience, `set_lower_bound` is supplied in order to set the lower
    /// bounds for all optimization parameters to a single constant
    pub fn set_lower_bound(&mut self, bound: f64) -> OptResult {
        let v = vec![bound; self.n_dims];
        self.set_lower_bounds(&v)
    }

    /// For convenience, `set_upper_bound` is supplied in order to set the upper
    /// bounds for all optimization parameters to a single constant
    pub fn set_upper_bound(&mut self, bound: f64) -> OptResult {
        let v = vec![bound; self.n_dims];
        self.set_upper_bounds(&v)
    }

    /// Retrieve the current upper bonds on `x`
    pub fn get_upper_bounds(&self) -> Option<Vec<f64>> {
        let mut bound: Vec<f64> = vec![0.0_f64; self.n_dims];
        let b = bound.as_mut_ptr();
        let res = unsafe { sys::nlopt_get_upper_bounds(self.nloptc_obj.0, b as *mut f64) };
        result_from_outcome(res).ok().map(|_| bound)
    }

    /// Retrieve the current lower bonds on `x`
    pub fn get_lower_bounds(&self) -> Option<Vec<f64>> {
        let mut bound: Vec<f64> = vec![0.0_f64; self.n_dims];
        let b = bound.as_mut_ptr();
        let res = unsafe { sys::nlopt_get_lower_bounds(self.nloptc_obj.0, b as *mut f64) };
        result_from_outcome(res).ok().map(|_| bound)
    }

    /// Several of the algorithms in NLopt (MMA, COBYLA, and ORIG_DIRECT) also support arbitrary
    /// nonlinear inequality constraints, and some additionally allow nonlinear equality constraints
    /// (ISRES and AUGLAG). For these algorithms, you can specify as many nonlinear constraints as
    /// you wish.
    ///
    /// In particular, a nonlinear constraint of the form `fc(x) = 0`, where the function
    /// fc is has the same form as an objective function, can be specified by calling this function.
    ///
    /// * `tolerance` - This parameter is a tolerance
    /// that is used for the purpose of stopping criteria only: a point `x` is considered feasible for
    /// judging whether to stop the optimization if `fc(x) ≤ tol`. A tolerance of zero means that
    /// NLopt will try not to consider any `x` to be converged unless the constraint is strictly
    /// satisfied;
    /// generally, at least a small positive tolerance is advisable to reduce sensitivity to
    /// rounding errors.
    pub fn add_equality_constraint<G: ObjFn<U>, U>(
        &mut self,
        constraint: G,
        user_data: U,
        tolerance: f64,
    ) -> OptResult {
        self.add_constraint(constraint, user_data, tolerance, true)
    }

    /// Set a nonlinear constraint of the form `fc(x) ≤ 0`.
    /// For more information see the documentation for `add_equality_constraint`.
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

    /// In some applications with multiple constraints, it is more convenient to define a single
    /// function that returns the values (and gradients) of all constraints at once. For example,
    /// different constraint functions might share computations in some way. Or, if you have a large
    /// number of constraints, you may wish to compute them in parallel. This possibility is
    /// supported by this function, which defines multiple equality constraints at once, or
    /// equivalently a vector-valued constraint function `fc(x) | R^n --> R^m`:
    ///
    /// * `constraint` - A constraint function bundled with user defined parameters.
    /// * `tolerance` - An array slice of length `m` of the tolerances in each constraint dimension
    pub fn add_equality_mconstraint<G: MObjFn<U>, U>(
        &mut self,
        m: usize,
        constraint: G,
        user_data: U,
        tolerance: &[f64],
    ) -> OptResult {
        self.add_mconstraint(m, constraint, user_data, tolerance, true)
    }

    /// Set a nonlinear multivalue inequality constraint.
    /// For more information see the documentation for `add_equality_mconstraint`.
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

    // TODO untested
    /// Remove all of the inequality and equality constraints from a given problem.
    pub fn remove_constraints(&mut self) -> OptResult {
        result_from_outcome(unsafe {
            std::cmp::min(
                sys::nlopt_remove_inequality_constraints(self.nloptc_obj.0),
                sys::nlopt_remove_equality_constraints(self.nloptc_obj.0),
            )
        })
    }

    /// Multiple stopping criteria for the optimization are supported,
    /// as specified by the functions to modify a given optimization problem. The optimization
    /// halts whenever any one of these criteria is satisfied. In some cases, the precise
    /// interpretation of the stopping criterion depends on the optimization algorithm above
    /// (although we have tried to make them as consistent as reasonably possible), and some
    /// algorithms do not support all of the stopping criteria.
    ///
    /// Note: you do not need to use all of the stopping criteria! In most cases, you only need one
    /// or two, and can omit the remainder (all criteria are disabled by default).
    ///
    /// This functions specifies a stop when an objective value of at least `stopval` is found: stop minimizing when an objective
    /// `value ≤ stopval` is found, or stop maximizing a `value ≥ stopval` is found.
    pub fn set_stopval(&mut self, stopval: f64) -> OptResult {
        result_from_outcome(unsafe { sys::nlopt_set_stopval(self.nloptc_obj.0, stopval) })
    }

    pub fn get_stopval(&self) -> f64 {
        unsafe { sys::nlopt_get_stopval(self.nloptc_obj.0) }
    }

    /// Set relative tolerance on function value: stop when an optimization step (or an estimate of
    /// the optimum) changes the objective function value by less than `tolerance` multiplied by the
    /// absolute value of the function value. (If there is any chance that your optimum function
    /// value is close to zero, you might want to set an absolute tolerance with `set_ftol_abs`
    /// as well.) Criterion is disabled if `tolerance` is non-positive.
    pub fn set_ftol_rel(&mut self, tolerance: f64) -> OptResult {
        result_from_outcome(unsafe { sys::nlopt_set_ftol_rel(self.nloptc_obj.0, tolerance) })
    }

    pub fn get_ftol_rel(&self) -> Option<f64> {
        unsafe {
            match sys::nlopt_get_ftol_rel(self.nloptc_obj.0) {
                x if x < 0.0 => None,
                x => Some(x),
            }
        }
    }

    /// Set absolute tolerance on function value: stop when an optimization step (or an estimate of
    /// the optimum) changes the function value by less than `tolerance`.
    /// Criterion is disabled if `tolerance` is
    /// non-positive.
    pub fn set_ftol_abs(&mut self, tolerance: f64) -> OptResult {
        result_from_outcome(unsafe { sys::nlopt_set_ftol_abs(self.nloptc_obj.0, tolerance) })
    }

    pub fn get_ftol_abs(&self) -> Option<f64> {
        match unsafe { sys::nlopt_get_ftol_abs(self.nloptc_obj.0) } {
            x if x < 0.0 => None,
            x => Some(x),
        }
    }

    /// Set relative tolerance on optimization parameters: stop when an optimization step (or an
    /// estimate of the optimum) changes every parameter by less than `tolerance`
    /// multiplied by the absolute
    /// value of the parameter. (If there is any chance that an optimal parameter is close to zero,
    /// you might want to set an absolute tolerance with `set_xtol_abs` as well.) Criterion is
    /// disabled if `tolerance` is non-positive.
    pub fn set_xtol_rel(&mut self, tolerance: f64) -> OptResult {
        result_from_outcome(unsafe { sys::nlopt_set_xtol_rel(self.nloptc_obj.0, tolerance) })
    }

    pub fn get_xtol_rel(&self) -> Option<f64> {
        match unsafe { sys::nlopt_get_xtol_rel(self.nloptc_obj.0) } {
            x if x < 0.0 => None,
            x => Some(x),
        }
    }

    /// Set absolute tolerances on optimization parameters. `tolerance` is a an array slice of length `n`
    /// giving the tolerances: stop when an optimization step (or
    /// an estimate of the optimum) changes every parameter `x[i]` by less than `tolerance[i]`.
    pub fn set_xtol_abs(&mut self, tolerance: &[f64]) -> OptResult {
        result_from_outcome(unsafe {
            sys::nlopt_set_xtol_abs(self.nloptc_obj.0, tolerance.as_ptr())
        })
    }

    /// For convenience, this function may be used to set the absolute tolerances in all `n`
    /// optimization parameters to the same value.
    pub fn set_xtol_abs1(&mut self, tolerance: f64) -> OptResult {
        let tol: &[f64] = &vec![tolerance; self.n_dims];
        self.set_xtol_abs(tol)
    }

    pub fn get_xtol_abs(&mut self) -> Option<Vec<f64>> {
        let mut tol: Vec<f64> = vec![0.0_f64; self.n_dims];
        let b = tol.as_mut_ptr();
        let res = unsafe { sys::nlopt_get_xtol_abs(self.nloptc_obj.0, b as *mut f64) };
        result_from_outcome(res).ok().map(|_| tol)
    }

    /// Stop when the number of function evaluations exceeds `maxeval`. (This is not a strict maximum:
    /// the number of function evaluations may exceed `maxeval` slightly, depending upon the
    /// algorithm.) Criterion is disabled if `maxeval` is non-positive.
    pub fn set_maxeval(&mut self, maxeval: u32) -> OptResult {
        result_from_outcome(unsafe { sys::nlopt_set_maxeval(self.nloptc_obj.0, maxeval as i32) })
    }

    pub fn get_maxeval(&mut self) -> Option<u32> {
        match unsafe { sys::nlopt_get_maxeval(self.nloptc_obj.0) } {
            x if x < 0 => None,
            x => Some(x as u32),
        }
    }

    /// Stop when the optimization time (in seconds) exceeds `maxtime`. (This is not a strict maximum:
    /// the time may exceed `maxtime` slightly, depending upon the algorithm and on how slow your
    /// function evaluation is.) Criterion is disabled if `maxtime` is non-positive.
    pub fn set_maxtime(&mut self, timeout: f64) -> OptResult {
        result_from_outcome(unsafe { sys::nlopt_set_maxtime(self.nloptc_obj.0, timeout) })
    }

    pub fn get_maxtime(&self) -> Option<f64> {
        match unsafe { sys::nlopt_get_maxtime(self.nloptc_obj.0) } {
            x if x < 0.0 => None,
            x => Some(x),
        }
    }

    /// In certain cases, the caller may wish to force the optimization to halt, for some reason
    /// unknown to NLopt. For example, if the user presses Ctrl-C, or there is an error of some
    /// sort in the objective function. In this case, it is possible to tell NLopt to halt
    /// the optimization gracefully, returning the best point found so far, by calling this
    /// function from within your objective or constraint functions. This causes nlopt_optimize to
    /// halt, returning the NLOPT_FORCED_STOP error code. It has no effect if not called
    /// during nlopt_optimize.
    ///
    /// # Params
    /// stopval: If you want to provide a bit more information, set a forced-stop integer value
    /// ```val```, which can be later retrieved by calling: ```get_force_stop()```, which returns  the
    /// last force-stop value that was set since the last nlopt_optimize. The force-stop value is
    /// ```None``` at the beginning of nlopt_optimize. Passing ```stopval=0``` to
    /// ```force_stop()``` tells NLopt not to force a halt.
    pub fn force_stop(&mut self, stopval: Option<i32>) -> OptResult {
        result_from_outcome(unsafe {
            match stopval {
                Some(x) => sys::nlopt_set_force_stop(self.nloptc_obj.0, x),
                None => sys::nlopt_force_stop(self.nloptc_obj.0),
            }
        })
    }

    pub fn get_force_stop(&mut self) -> Option<i32> {
        match unsafe { sys::nlopt_get_force_stop(self.nloptc_obj.0) } {
            0 => None,
            x => Some(x),
        }
    }

    /// Some of the algorithms, especially MLSL and AUGLAG, use a different optimization algorithm
    /// as a subroutine, typically for local optimization. You can change the local search algorithm
    /// and its tolerances using this function.
    ///
    /// Here, local_opt is another `Nlopt<T>` whose parameters are used to determine the local
    /// search algorithm, its stopping criteria, and other algorithm parameters. (However, the
    /// objective function, bounds, and nonlinear-constraint parameters of `local_opt` are ignored.)
    /// The dimension `n` of `local_opt` must match that of the main optimization.
    ///
    /// A stubbed version of `local_opt` can be obtained with `get_local_optimizer`.
    pub fn set_local_optimizer(&mut self, local_opt: Nlopt<impl ObjFn<()>, ()>) -> OptResult {
        result_from_outcome(unsafe {
            sys::nlopt_set_local_optimizer(self.nloptc_obj.0, local_opt.nloptc_obj.0)
        })
    }

    pub fn get_local_optimizer(&mut self, algorithm: Algorithm) -> Nlopt<impl ObjFn<()>, ()> {
        fn stub_opt(_: &[f64], _: Option<&mut [f64]>, _: &mut ()) -> f64 {
            unreachable!()
        }
        // create a new object based on former one
        Nlopt::new(algorithm, self.n_dims, stub_opt, self.target, ())
    }

    /// For derivative-free local-optimization algorithms, the optimizer must somehow decide on some
    /// initial step size to perturb x by when it begins the optimization. This step size should be
    /// big enough that the value of the objective changes significantly, but not too big if you
    /// want to find the local optimum nearest to x. By default, NLopt chooses this initial step
    /// size heuristically from the bounds, tolerances, and other information, but this may not
    /// always be the best choice. You can use this function to modify the initial step size.
    ///
    /// Here, `dx` is an array of length `n` containing
    /// the (nonzero) initial step size for each component of the optimization parameters `x`. For
    /// convenience, if you want to set the step sizes in every direction to be the same value, you
    /// can instead call `set_initial_step1`.
    pub fn set_initial_step(&mut self, dx: &[f64]) -> OptResult {
        result_from_outcome(unsafe { sys::nlopt_set_initial_step(self.nloptc_obj.0, dx.as_ptr()) })
    }

    pub fn set_initial_step1(&mut self, dx: f64) -> OptResult {
        let d: &[f64] = &vec![dx; self.n_dims];
        self.set_initial_step(d)
    }

    /// Here, `x` is the same as the initial guess that you plan to pass to `optimize` – if you
    /// have not set the initial step and NLopt is using its heuristics, its heuristic step size may
    /// depend on the initial `x`, which is why you must pass it here. Both `x`
    /// and the return value are arrays of
    /// length `n`.
    pub fn get_initial_step(&mut self, x: &[f64]) -> Option<Vec<f64>> {
        let mut dx: Vec<f64> = vec![0.0_f64; self.n_dims];
        let b = dx.as_mut_ptr();
        let res =
            unsafe { sys::nlopt_get_initial_step(self.nloptc_obj.0, x.as_ptr(), b as *mut f64) };
        result_from_outcome(res).ok().map(|_| dx)
    }

    // Stochastic Population
    /// Several of the stochastic search algorithms (e.g., CRS, MLSL, and ISRES) start by generating
    /// some initial "population" of random points `x.` By default, this initial population size is
    /// chosen heuristically in some algorithm-specific way, but the initial population can by
    /// changed by calling this function. A `population` of zero implies
    /// that the heuristic default will be used.
    pub fn set_population(&mut self, population: usize) -> OptResult {
        result_from_outcome(unsafe {
            sys::nlopt_set_population(self.nloptc_obj.0, population as u32)
        })
    }

    pub fn get_population(&mut self) -> usize {
        unsafe { sys::nlopt_get_population(self.nloptc_obj.0) as usize }
    }

    // Pseudorandom Numbers
    /// For stochastic optimization algorithms, we use pseudorandom numbers generated by the
    /// Mersenne Twister algorithm, based on code from Makoto Matsumoto. By default, the seed for
    /// the random numbers is generated from the system time, so that you will get a different
    /// sequence of pseudorandom numbers each time you run your program. If you want to use a
    /// "deterministic" sequence of pseudorandom numbers, i.e. the same sequence from run to run,
    /// you can set the seed with this function. To reset the seed based on the system time, you can
    /// call this function with `seed = None`.
    pub fn srand_seed(seed: Option<u64>) {
        unsafe {
            match seed {
                None => sys::nlopt_srand_time(),
                Some(x) => sys::nlopt_srand(x as c_ulong),
            }
        }
    }

    // Vector storage for limited-memory quasi-Newton algorithms
    /// Some of the NLopt algorithms are limited-memory "quasi-Newton" algorithms, which "remember"
    /// the gradients from a finite number M of the previous optimization steps in order to
    /// construct an approximate 2nd derivative matrix. The bigger M is, the more storage the
    /// algorithms require, but on the other hand they may converge faster for larger M. By default,
    /// NLopt chooses a heuristic value of M, but this can be changed by calling this function.
    /// Passing M=0 (the default) tells NLopt to use a heuristic value. By default, NLopt currently
    /// sets M to 10 or at most 10 MiB worth of vectors, whichever is larger.
    pub fn set_vector_storage(&mut self, m: Option<usize>) -> OptResult {
        let outcome = match m {
            None => unsafe { sys::nlopt_set_vector_storage(self.nloptc_obj.0, 0_u32) },
            Some(x) => unsafe { sys::nlopt_set_vector_storage(self.nloptc_obj.0, x as u32) },
        };
        result_from_outcome(outcome)
    }

    pub fn get_vector_storage(&mut self) -> usize {
        unsafe { sys::nlopt_get_vector_storage(self.nloptc_obj.0) as usize }
    }

    // Preconditioning TODO --> this is somewhat complex but not overly so.
    // Just did not get around to it yet.

    /// To determine the version number of NLopt at runtime, you can call this function. For
    /// example, NLopt version 3.1.4 would return `(3, 1, 4)`.
    pub fn version() -> (i32, i32, i32) {
        unsafe {
            let mut i: i32 = 0;
            let mut j: i32 = 0;
            let mut k: i32 = 0;
            sys::nlopt_version(&mut i, &mut j, &mut k);
            (i, j, k)
        }
    }

    /// Once all of the desired optimization parameters have been specified in a given
    /// `NLoptOptimzer`, you can perform the optimization by calling this function. On input,
    /// `x_init` is an array of length `n` giving an initial
    /// guess for the optimization parameters. On successful return, `x_init`
    /// contains the optimized values
    /// of the parameters, and the function returns the corresponding value of the objective function.
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
/// Can be useful when a gradient must be provided to the optimization
/// algorithm and a closed-form derivative cannot be obtained
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
