#![allow(dead_code)]

//! # Rust-NLopt
//! 
//! This is a wrapper for the NLopt library (http://ab-initio.mit.edu/wiki/index.php/NLopt).
//! Study first the documentation for the `NLoptOptimizer` `struct` to get started.

enum NLoptOpt {}

///Defines constants to specify whether the objective function should be minimized or maximized.
pub enum NLoptTarget {
    MAXIMIZE,
    MINIMIZE,
}

///Optimization algorithms available in NLopt. For a description, check
///http://ab-initio.mit.edu/wiki/index.php/NLopt_Algorithms
#[repr(C)]
#[allow(non_camel_case_types)]
pub enum NLoptAlgorithm {
    GN_DIRECT = 0,
    GN_DIRECT_L,
    GN_DIRECT_L_RAND,
    GN_DIRECT_NOSCAL,
    GN_DIRECT_L_NOSCAL,
    GN_DIRECT_L_RAND_NOSCAL,

    GN_ORIG_DIRECT,
    GN_ORIG_DIRECT_L,

    GD_STOGO,
    GD_STOGO_RAND,

    LD_LBFGS_NOCEDAL,

    LD_LBFGS,

    LN_PRAXIS,

    LD_VAR1,
    LD_VAR2,

    LD_TNEWTON,
    LD_TNEWTON_RESTART,
    LD_TNEWTON_PRECOND,
    LD_TNEWTON_PRECOND_RESTART,

    GN_CRS2_LM,

    GN_MLSL,
    GD_MLSL,
    GN_MLSL_LDS,
    GD_MLSL_LDS,

    LD_MMA,

    LN_COBYLA,

    LN_NEWUOA,
    LN_NEWUOA_BOUND,

    LN_NELDERMEAD,
    LN_SBPLX,

    LN_AUGLAG,
    LD_AUGLAG,
    LN_AUGLAG_EQ,
    LD_AUGLAG_EQ,

    LN_BOBYQA,

    GN_ISRES,

    /* new variants that require local_optimizer to be set,
     * not with older constants for backwards compatibility
     **/
    AUGLAG,
    AUGLAG_EQ,
    G_MLSL,
    G_MLSL_LDS,

    LD_SLSQP,

    LD_CCSAQ,

    GN_ESCH,

    NUM_ALGORITHMS /* not an algorithm, just the number of them */
}

extern crate libc;
use std::slice;
use libc::*;

#[link(name = "nlopt",
       vers = "0.1.0")]
extern "C" {
    fn nlopt_create(algorithm: i32, n_dims: u32) -> *mut NLoptOpt;
    fn nlopt_destroy(opt: *mut NLoptOpt);
    fn nlopt_set_min_objective(opt: *mut NLoptOpt, nlopt_fdf: extern "C" fn(n:u32,x:*const f64,g:*mut f64,d:*mut c_void) -> f64, d:*const c_void) -> i32;
    fn nlopt_set_max_objective(opt: *mut NLoptOpt, nlopt_fdf: extern "C" fn(n:u32,x:*const f64,g:*mut f64,d:*mut c_void) -> f64, d:*const c_void) -> i32;
    fn nlopt_optimize(opt: *mut NLoptOpt, x_init:*mut f64, opt_value: *mut f64) -> i32;
    fn nlopt_set_lower_bounds(opt: *mut NLoptOpt, lb: *const f64) -> i32;
    fn nlopt_set_upper_bounds(opt: *mut NLoptOpt, ub: *const f64) -> i32;
    fn nlopt_get_lower_bounds(opt: *const NLoptOpt, lb: *mut f64) -> i32;
    fn nlopt_get_upper_bounds(opt: *const NLoptOpt, ub: *mut f64) -> i32;
    fn nlopt_add_inequality_constraint(opt: *mut NLoptOpt, fc: extern "C" fn(n:u32,x:*const f64,g:*mut f64,d:*mut c_void) -> f64, d: *const c_void, tol: f64) -> i32;
    fn nlopt_add_equality_constraint(opt: *mut NLoptOpt, fc: extern "C" fn(n:u32,x:*const f64,g:*mut f64,d:*mut c_void) -> f64, d: *const c_void, tol: f64) -> i32;
    fn nlopt_add_inequality_mconstraint(opt: *mut NLoptOpt, m:u32, fc: extern "C" fn(m:u32, r:*mut f64, n:u32,x:*const f64,g:*mut f64,d:*mut c_void) -> i32, d: *const c_void, tol:*const f64) -> i32;
    fn nlopt_add_equality_mconstraint(opt: *mut NLoptOpt, m:u32, fc: extern "C" fn(m:u32, r:*mut f64, n:u32,x:*const f64,g:*mut f64,d:*mut c_void) -> i32, d: *const c_void, tol:*const f64) -> i32;
    fn nlopt_remove_inequality_constraints(opt: *mut NLoptOpt) -> i32;
    fn nlopt_remove_equality_constraints(opt: *mut NLoptOpt) -> i32;
    fn nlopt_set_stopval(opt: *mut NLoptOpt, stopval:f64) -> i32;
    fn nlopt_get_stopval(opt: *const NLoptOpt) -> f64;
    fn nlopt_set_ftol_rel(opt: *mut NLoptOpt, tol: f64) -> i32;
    fn nlopt_get_ftol_rel(opt: *const NLoptOpt) -> f64;
    fn nlopt_set_ftol_abs(opt: *mut NLoptOpt, tol: f64) -> i32;
    fn nlopt_get_ftol_abs(opt: *const NLoptOpt) -> f64;
    fn nlopt_set_xtol_rel(opt: *mut NLoptOpt, tol: f64) -> i32;
    fn nlopt_get_xtol_rel(opt: *const NLoptOpt) -> f64;
    fn nlopt_set_xtol_abs(opt: *mut NLoptOpt, tol: *const f64) -> i32;
    fn nlopt_get_xtol_abs(opt: *const NLoptOpt, tol: *mut f64) -> i32;
    fn nlopt_set_maxeval(opt: *mut NLoptOpt, maxeval: i32) -> i32;
    fn nlopt_get_maxeval(opt: *const NLoptOpt) -> i32;
    fn nlopt_set_maxtime(opt: *mut NLoptOpt, maxtime: f64) -> i32;
    fn nlopt_get_maxtime(opt: *const NLoptOpt) -> f64;
    fn nlopt_force_stop(opt: *mut NLoptOpt) -> i32;
    fn nlopt_set_force_stop(opt: *mut NLoptOpt, val: i32) -> i32;
    fn nlopt_get_force_stop(opt: *const NLoptOpt) -> i32;
    fn nlopt_set_local_optimizer(opt: *mut NLoptOpt, local_opt: *mut NLoptOpt) -> i32;
    fn nlopt_set_population(opt: *mut NLoptOpt, pop: u32) -> i32;
    fn nlopt_get_population(opt: *const NLoptOpt) -> u32;
    fn nlopt_set_initial_step(opt: *mut NLoptOpt, dx: *const f64) -> i32;
    fn nlopt_get_initial_step(opt: *const NLoptOpt, x: *const f64, dx: *mut f64) -> i32;
    fn nlopt_srand(seed: c_ulong);
    fn nlopt_srand_time();
    fn nlopt_set_vector_storage(opt: *mut NLoptOpt, m: u32) -> i32;
    fn nlopt_get_vector_storage(opt: *const NLoptOpt) -> u32;
    fn nlopt_version(major: *mut i32, minor: *mut i32, bugfix: *mut i32);
}

/// This is the central ```struct``` of this library. It represents an optimization of a given
/// function, called the objective function. The argument `x` to this function is an
/// `n`-dimensional double-precision vector. The dimensions are set at creation of the struct and
/// cannot be changed afterwards. NLopt offers different optimization algorithms. One must be
/// chosen at struct creation and cannot be changed afterwards. Always use ```NLoptOptimizer::<T>::new()``` to create an `NLoptOptimizer` struct.
pub struct NLoptOptimizer<T> {
    opt: *mut NLoptOpt,
    n_dims: usize,
    params: T,
    function: NLoptFn<T>,
}

/// A function `f(x) | R^n --> R` with additional user specified parameters `params` of type `T`.
///
/// * `argument` - `n`-dimensional array `x`
/// * `gradient` - `n`-dimensional array to store the gradient `grad f(x)`. If `gradient` matches
/// `Some(x)`, the user is required to provide a gradient, otherwise the optimization will
/// probabely fail.
/// * `params` - user defined data
///
/// # Returns
/// `f(x)`
pub type NLoptFn<T> = fn(argument: &[f64],
                     gradient: Option<&mut [f64]>,
                     params: T) -> f64;

type StrResult = Result<&'static str,&'static str>;

/// Packs a function of type `NLoptFn<T>` with a user defined parameter set of type `T`.
pub struct Function<T> {
    function: NLoptFn<T>,
    params: T,
}

/// Defines constants for equality constraints (of the form `f(x) = 0`) and inequality constraints
/// (of the form `f(x) <= 0`).
pub enum ConstraintType {
    EQUALITY,
    INEQUALITY,
}

struct Constraint<F> {
    function: F,
    ctype: ConstraintType,
}

/// A function `f(x) | R^n --> R^m` with additional user specified parameters `params` of type `T`.
///
/// * `result` - `m`-dimensional array to store the value `f(x)`
/// * `argument` - `n`-dimensional array `x`
/// * `gradient` - `n×m`-dimensional array to store the gradient `grad f(x)`. The n dimension of
/// gradient is stored contiguously, so that `df_i / dx_j` is stored in `gradient[i*n + j]`. If
/// `gradient` is `Some(x)`, the user is required to return a valid gradient, otherwise the
/// optimization will most likely fail.
/// * `params` - user defined data
///
/// # Returns
/// If an error occurs, the function should return an `Err(x)` where `x` is a string literal
/// specifying the error. On success, just return `Ok(())`.
pub type NLoptMFn<T> = fn(result: &mut [f64],
                      argument: &[f64],
                      gradient: Option<&mut [f64]>,
                      params: T) -> Result<(),&'static str>;

/// Packs an `m`-dimensional function of type `NLoptMFn<T>` with a user defined parameter set of type `T`.
pub struct MFunction<T> {
    m: usize,
    function: NLoptMFn<T>,
    params: T,
}

impl <T> NLoptOptimizer<T> where T: Copy {
    ///Creates a new `NLoptOptimizer` struct.
    ///
    /// * `algorithm` - Which optimization algorithm to use. This cannot be changed after creation
    /// of the struct
    /// * `n_dims` - Dimension of the argument to the objective function
    /// * `obj` - The objective function. This function has the signature `(&[f64],
    /// Option<&mut [f64]>, T) -> f64`. The first argument is the vector `x` passed to the function.
    /// The second argument is `Some(&mut [f64])` if the calling optimization algorithm needs
    /// the gradient of the function. If the gradient is not needed, it is `None`. The last
    /// argument is the user data provided beforehand using the `user_data` argument to the
    /// constructor.
    /// * `target` - Whether to minimize or maximize the objectiv function
    /// * `user_data` - Optional data that is passed to the objective function
    pub fn new(algorithm: NLoptAlgorithm, n_dims: usize, obj: NLoptFn<T>, target : NLoptTarget, user_data: T) -> NLoptOptimizer<T> {
        unsafe{
            let fb = Box::new(Function{ function: obj, params: user_data });
            let opt = NLoptOptimizer {
                opt: nlopt_create(algorithm as i32,n_dims as u32),
                n_dims: n_dims,
                params: user_data,
                function: obj,
            };
            let u_data = Box::into_raw(fb) as *const c_void;
            match target {
                NLoptTarget::MINIMIZE => nlopt_set_min_objective(opt.opt, NLoptOptimizer::<T>::function_raw_callback, u_data),
                NLoptTarget::MAXIMIZE => nlopt_set_max_objective(opt.opt, NLoptOptimizer::<T>::function_raw_callback, u_data),
            };
            opt
        }
    }

    //Static Bounds
    ///Most of the algorithms in NLopt are designed for minimization of functions with simple bound
    ///constraints on the inputs. That is, the input vectors `x` are constrainted to lie in a
    ///hyperrectangle `lower_bound[i] ≤ x[i] ≤ upper_bound[i] for 0 ≤ i < n`. NLopt guarantees that your objective
    ///function and any nonlinear constraints will never be evaluated outside of these bounds
    ///(unlike nonlinear constraints, which may be violated at intermediate steps).
    ///
    ///These bounds are specified by passing an array `bound` of length `n` (the dimension of the
    ///problem) to one or both of the functions:
    ///
    ///`set_lower_bounds(&[f64])`
    ///
    ///`set_upper_bounds(&[f64])`
    ///
    ///If a lower/upper bound is not set, the default is no bound (unconstrained, i.e. a bound of
    ///infinity); it is possible to have lower bounds but not upper bounds or vice versa.
    ///Alternatively, the user can call one of the above functions and explicitly pass a lower
    ///bound of `-INFINITY` and/or an upper bound of `+INFINITY` for some optimization parameters to
    ///make them have no lower/upper bound, respectively.
    ///
    ///It is permitted to set `lower_bound[i] == upper_bound[i]` in one or more dimensions; this is equivalent to
    ///fixing the corresponding `x[i]` parameter, eliminating it from the optimization.
    ///
    ///Note, however, that some of the algorithms in NLopt, in particular most of the
    ///global-optimization algorithms, do not support unconstrained optimization and will return an
    ///error in `optimize` if you do not supply finite lower and upper bounds.
    pub fn set_lower_bounds(&mut self, bound: &[f64]) -> Result<i32,i32> {
        let ret;
        unsafe{
            ret = nlopt_set_lower_bounds(self.opt, bound.as_ptr());
        }
        match ret {
            x if x < 0 => Err(x),
            x => Ok(x),
        }
    }

    ///See documentation for `set_lower_bounds`
    pub fn set_upper_bounds(&mut self, bound: &[f64]) -> Result<i32,i32> {
        let ret;
        unsafe{
            ret = nlopt_set_upper_bounds(self.opt, bound.as_ptr());
        }
        match ret {
            x if x < 0 => Err(x),
            x => Ok(x),
        }
    }

    ///For convenience, `set_lower_bound` is supplied in order to set the lower
    ///bounds for all optimization parameters to a single constant
    pub fn set_lower_bound(&mut self, bound: f64) -> Result<i32,i32>{
        let v = vec![bound;self.n_dims];
        self.set_lower_bounds(&v)
    }

    ///For convenience, `set_upper_bound` is supplied in order to set the upper
    ///bounds for all optimization parameters to a single constant
    pub fn set_upper_bound(&mut self, bound: f64) -> Result<i32,i32>{
        let v = vec![bound;self.n_dims];
        self.set_upper_bounds(&v)
    }

    ///Retrieve the current upper bonds on `x`
    pub fn get_upper_bounds(&self) -> Option<&[f64]>{
        let mut bound : Vec<f64> = vec![0.0 as f64;self.n_dims];
        unsafe {
            let b = bound.as_mut_ptr();
            let ret = nlopt_get_upper_bounds(self.opt, b as *mut f64);
            match ret {
                x if x < 0 => None,
                _ => Some(slice::from_raw_parts(b as *mut f64,self.n_dims))
            }
        }
    }

    ///Retrieve the current lower bonds on `x`
    pub fn get_lower_bounds(&self) -> Option<&[f64]>{
        let mut bound : Vec<f64> = vec![0.0 as f64;self.n_dims];
        unsafe {
            let b = bound.as_mut_ptr();
            let ret = nlopt_get_lower_bounds(self.opt, b as *mut f64);
            match ret {
                x if x < 0 => None,
                _ => Some(slice::from_raw_parts(b as *mut f64,self.n_dims))
            }
        }
    }

    //Nonlinear Constraints
    ///Several of the algorithms in NLopt (MMA, COBYLA, and ORIG_DIRECT) also support arbitrary
    ///nonlinear inequality constraints, and some additionally allow nonlinear equality constraints
    ///(ISRES and AUGLAG). For these algorithms, you can specify as many nonlinear constraints as
    ///you wish.
    ///
    ///In particular, a nonlinear constraint of the form `fc(x) ≤ 0` or `fc(x) = 0`, where the function
    ///fc is an `NLoptFn<T>`, can be specified by calling this function.
    ///
    ///* `t` - Specify whether the constraint is an equality (`fc(x) = 0`) or inequality (`fc(x) ≤ 0`) constraint.
    ///* `tolerance` - This parameter is a tolerance
    ///that is used for the purpose of stopping criteria only: a point `x` is considered feasible for
    ///judging whether to stop the optimization if `fc(x) ≤ tol`. A tolerance of zero means that
    ///NLopt will try not to consider any `x` to be converged unless the constraint is strictly
    ///satisfied;
    ///generally, at least a small positive tolerance is advisable to reduce sensitivity to
    ///rounding errors.
    pub fn add_constraint(&mut self, constraint: Box<Function<T>>, t: ConstraintType, tolerance: f64) -> StrResult {
        match t {
            ConstraintType::INEQUALITY => unsafe { 
                NLoptOptimizer::<T>::nlopt_res_to_result(
                    nlopt_add_inequality_constraint(self.opt, NLoptOptimizer::<T>::function_raw_callback, Box::into_raw(constraint) as *const c_void, tolerance)
                    ) 
            },
            ConstraintType::EQUALITY => unsafe { 
                NLoptOptimizer::<T>::nlopt_res_to_result(
                    nlopt_add_equality_constraint(self.opt, NLoptOptimizer::<T>::function_raw_callback, Box::into_raw(constraint) as *const c_void, tolerance)
                    ) 
            },
        }
    }
    
    //UNTESTED
    ///In some applications with multiple constraints, it is more convenient to define a single
    ///function that returns the values (and gradients) of all constraints at once. For example,
    ///different constraint functions might share computations in some way. Or, if you have a large
    ///number of constraints, you may wish to compute them in parallel. This possibility is
    ///supported by this function, which defines multiple constraints at once, or
    ///equivalently a vector-valued constraint function `fc(x) | R^n --> R^m`:
    ///
    /// * `constraint` - A `Box` containing the constraint function bundled with user defined
    /// parameters.
    /// * `t` - Specify whether the constraint is an equality or inequality constraint
    /// * `tolerance` - An array slice of length `m` of the tolerances in each constraint dimension
    pub fn add_mconstraint(&mut self, constraint: Box<MFunction<T>>, t: ConstraintType, tolerance: &[f64]) -> StrResult {
        let m: u32 = (*constraint).m as u32;
        match t {
            ConstraintType::INEQUALITY => unsafe { 
                NLoptOptimizer::<T>::nlopt_res_to_result(
                    nlopt_add_inequality_mconstraint(self.opt, m, NLoptOptimizer::<T>::mfunction_raw_callback, Box::into_raw(constraint) as *const c_void, tolerance.as_ptr())
                    ) 
            },
            ConstraintType::EQUALITY => unsafe { 
                NLoptOptimizer::<T>::nlopt_res_to_result(
                    nlopt_add_equality_mconstraint(self.opt, m, NLoptOptimizer::<T>::mfunction_raw_callback, Box::into_raw(constraint) as *const c_void, tolerance.as_ptr())
                    ) 
            },
        }
    }
    
    //UNTESTED
    ///Remove all of the inequality and equality constraints from a given problem.
    pub fn remove_constraints(&mut self) -> StrResult{
        unsafe {
            NLoptOptimizer::<T>::nlopt_res_to_result(
                std::cmp::min(
                    nlopt_remove_inequality_constraints(self.opt),
                    nlopt_remove_equality_constraints(self.opt)
                )
            )
        }
    }

    //Stopping Criteria
    ///Multiple stopping criteria for the optimization are supported,
    ///as specified by the functions to modify a given optimization problem. The optimization
    ///halts whenever any one of these criteria is satisfied. In some cases, the precise
    ///interpretation of the stopping criterion depends on the optimization algorithm above
    ///(although we have tried to make them as consistent as reasonably possible), and some
    ///algorithms do not support all of the stopping criteria.
    ///
    ///Note: you do not need to use all of the stopping criteria! In most cases, you only need one
    ///or two, and can omit the remainder (all criteria are disabled by default).
    ///
    ///This functions specifies a stop when an objective value of at least `stopval` is found: stop minimizing when an objective
    ///`value ≤ stopval` is found, or stop maximizing a `value ≥ stopval` is found.
    pub fn set_stopval(&mut self, stopval: f64) -> StrResult {
        unsafe {
            NLoptOptimizer::<T>::nlopt_res_to_result(
                nlopt_set_stopval(self.opt, stopval)
                )
        }
    }
    
    pub fn get_stopval(& self) -> f64 {
        unsafe{
            nlopt_get_stopval(self.opt)
        }
    }

    ///Set relative tolerance on function value: stop when an optimization step (or an estimate of
    ///the optimum) changes the objective function value by less than `tolerance` multiplied by the
    ///absolute value of the function value. (If there is any chance that your optimum function
    ///value is close to zero, you might want to set an absolute tolerance with `set_ftol_abs`
    ///as well.) Criterion is disabled if `tolerance` is non-positive.
    pub fn set_ftol_rel(&mut self, tolerance: f64) -> StrResult {
        unsafe {
            NLoptOptimizer::<T>::nlopt_res_to_result(
                nlopt_set_ftol_rel(self.opt, tolerance)
                )
        }
    }
    
    pub fn get_ftol_rel(& self) -> Option<f64> {
        unsafe{
            match nlopt_get_ftol_rel(self.opt) {
                x if x < 0.0 => None,
                x => Some(x),
            }
        }
    }

    ///Set absolute tolerance on function value: stop when an optimization step (or an estimate of
    ///the optimum) changes the function value by less than `tolerance`. Criterion is disabled if `tolerance` is
    ///non-positive.
    pub fn set_ftol_abs(&mut self, tolerance: f64) -> StrResult {
        unsafe {
            NLoptOptimizer::<T>::nlopt_res_to_result(
                nlopt_set_ftol_abs(self.opt, tolerance)
                )
        }
    }
    
    pub fn get_ftol_abs(& self) -> Option<f64> {
        unsafe{
            match nlopt_get_ftol_abs(self.opt) {
                x if x < 0.0 => None,
                x => Some(x),
            }
        }
    }

    ///Set relative tolerance on optimization parameters: stop when an optimization step (or an
    ///estimate of the optimum) changes every parameter by less than `tolerance` multiplied by the absolute
    ///value of the parameter. (If there is any chance that an optimal parameter is close to zero,
    ///you might want to set an absolute tolerance with `set_xtol_abs` as well.) Criterion is
    ///disabled if `tolerance` is non-positive.
    pub fn set_xtol_rel(&mut self, tolerance: f64) -> StrResult {
        unsafe {
            NLoptOptimizer::<T>::nlopt_res_to_result(
                nlopt_set_xtol_rel(self.opt, tolerance)
                )
        }
    }
    
    pub fn get_xtol_rel(& self) -> Option<f64> {
        unsafe{
            match nlopt_get_xtol_rel(self.opt) {
                x if x < 0.0 => None,
                x => Some(x),
            }
        }
    }

    ///Set absolute tolerances on optimization parameters. `tolerance` is a an array slice of length `n`
    ///giving the tolerances: stop when an optimization step (or
    ///an estimate of the optimum) changes every parameter `x[i]` by less than `tolerance[i]`.
    pub fn set_xtol_abs(&mut self, tolerance: &[f64]) -> StrResult{
        unsafe {
            NLoptOptimizer::<T>::nlopt_res_to_result(nlopt_set_xtol_abs(self.opt, tolerance.as_ptr()))
        }
    }

    ///For convenience, this function may be used to set the absolute tolerances in all `n`
    ///optimization parameters to the same value.
    pub fn set_xtol_abs1(&mut self, tolerance: f64) -> StrResult{
        let tol : &[f64] = &vec![tolerance;self.n_dims];
        self.set_xtol_abs(tol)
    }
    
    pub fn get_xtol_abs(&mut self) -> Option<&[f64]> {
        let mut tol : Vec<f64> = vec![0.0 as f64;self.n_dims];
        unsafe {
            let b = tol.as_mut_ptr();
            let ret = nlopt_get_xtol_abs(self.opt, b as *mut f64);
            match ret {
                x if x < 0 => None,
                _ => Some(slice::from_raw_parts(b as *mut f64,self.n_dims))
            }
        }
    }

    ///Stop when the number of function evaluations exceeds `maxeval`. (This is not a strict maximum:
    ///the number of function evaluations may exceed `maxeval` slightly, depending upon the
    ///algorithm.) Criterion is disabled if `maxeval` is non-positive.
    pub fn set_maxeval(&mut self, maxeval: u32) -> StrResult {
        unsafe{
            let ret = nlopt_set_maxeval(self.opt, maxeval as i32);
            NLoptOptimizer::<T>::nlopt_res_to_result(ret)
        }
    }

    pub fn get_maxeval(&mut self) -> Option<u32> {
        unsafe {
            match nlopt_get_maxeval(self.opt){
                x if x < 0 => None,
                x => Some(x as u32),
            }
        }
    }

    ///Stop when the optimization time (in seconds) exceeds `maxtime`. (This is not a strict maximum:
    ///the time may exceed `maxtime` slightly, depending upon the algorithm and on how slow your
    ///function evaluation is.) Criterion is disabled if `maxtime` is non-positive.
    pub fn set_maxtime(&mut self, timeout: f64) -> StrResult {
        unsafe {
            NLoptOptimizer::<T>::nlopt_res_to_result(
                nlopt_set_maxtime(self.opt, timeout)
                )
        }
    }
    
    pub fn get_maxtime(& self) -> Option<f64> {
        unsafe{
            match nlopt_get_maxtime(self.opt) {
                x if x < 0.0 => None,
                x => Some(x),
            }
        }
    }

    //Forced Termination

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
    pub fn force_stop(&mut self, stopval: Option<i32>) -> StrResult {
        unsafe {
            match stopval {
                Some(x) => NLoptOptimizer::<T>::nlopt_res_to_result(
                    nlopt_set_force_stop(self.opt, x)
                    ),
                None => NLoptOptimizer::<T>::nlopt_res_to_result(
                    nlopt_force_stop(self.opt)
                    ),
            }
        }
    }
    
    pub fn get_force_stop(&mut self) -> Option<i32> {
        unsafe {
            match nlopt_get_force_stop(self.opt) {
                0 => None,
                x => Some(x),
            }
        }
    }
 
    //Local Optimization
    ///Some of the algorithms, especially MLSL and AUGLAG, use a different optimization algorithm
    ///as a subroutine, typically for local optimization. You can change the local search algorithm
    ///and its tolerances using this function.
    ///
    ///Here, local_opt is another `NLoptOptimizer<T>` whose parameters are used to determine the local
    ///search algorithm, its stopping criteria, and other algorithm parameters. (However, the
    ///objective function, bounds, and nonlinear-constraint parameters of `local_opt` are ignored.)
    ///The dimension `n` of `local_opt` must match that of the main optimization.
    pub fn set_local_optimizer(&mut self, local_opt: NLoptOptimizer<T>) -> StrResult{
        unsafe {
            NLoptOptimizer::<T>::nlopt_res_to_result( nlopt_set_local_optimizer(self.opt, local_opt.opt) )
        }
    }
    
    //Initial Step Size
    ///For derivative-free local-optimization algorithms, the optimizer must somehow decide on some
    ///initial step size to perturb x by when it begins the optimization. This step size should be
    ///big enough that the value of the objective changes significantly, but not too big if you
    ///want to find the local optimum nearest to x. By default, NLopt chooses this initial step
    ///size heuristically from the bounds, tolerances, and other information, but this may not
    ///always be the best choice. You can use this function to modify the initial step size.
    ///
    ///Here, `dx` is an array of length `n` containing
    ///the (nonzero) initial step size for each component of the optimization parameters `x`. For
    ///convenience, if you want to set the step sizes in every direction to be the same value, you
    ///can instead call `set_initial_step1`.
    pub fn set_initial_step(&mut self, dx: &[f64]) -> StrResult{
        unsafe {
            NLoptOptimizer::<T>::nlopt_res_to_result(nlopt_set_initial_step(self.opt, dx.as_ptr()))
        }
    }

    pub fn set_initial_step1(&mut self, dx: f64) -> StrResult{
        let d : &[f64] = &vec![dx;self.n_dims];
        self.set_initial_step(d)
    }
    
    ///Here, `x` is the same as the initial guess that you plan to pass to `optimize` – if you
    ///have not set the initial step and NLopt is using its heuristics, its heuristic step size may
    ///depend on the initial `x`, which is why you must pass it here. Both `x` and the return value are arrays of
    ///length `n`.
    pub fn get_initial_step(&mut self, x: &[f64]) -> Option<&[f64]> {
        let mut dx : Vec<f64> = vec![0.0 as f64;self.n_dims];
        unsafe {
            let b = dx.as_mut_ptr();
            let ret = nlopt_get_initial_step(self.opt, x.as_ptr(), b as *mut f64);
            match ret {
                x if x < 0 => None,
                _ => Some(slice::from_raw_parts(b as *mut f64,self.n_dims))
            }
        }
    }

    //Stochastic Population
    ///Several of the stochastic search algorithms (e.g., CRS, MLSL, and ISRES) start by generating
    ///some initial "population" of random points `x.` By default, this initial population size is
    ///chosen heuristically in some algorithm-specific way, but the initial population can by
    ///changed by calling this function. A `population` of zero implies that the heuristic default will be
    ///used.
    pub fn set_population(&mut self, population: usize) -> StrResult {
        unsafe {
            NLoptOptimizer::<T>::nlopt_res_to_result( nlopt_set_population(self.opt, population as u32) )
        }
    }

    pub fn get_population(&mut self) -> usize {
        unsafe {
            nlopt_get_population(self.opt) as usize
        }
    }

    //Pseudorandom Numbers
    ///For stochastic optimization algorithms, we use pseudorandom numbers generated by the
    ///Mersenne Twister algorithm, based on code from Makoto Matsumoto. By default, the seed for
    ///the random numbers is generated from the system time, so that you will get a different
    ///sequence of pseudorandom numbers each time you run your program. If you want to use a
    ///"deterministic" sequence of pseudorandom numbers, i.e. the same sequence from run to run,
    ///you can set the seed with this function. To reset the seed based on the system time, you can
    ///call this function with `seed = None`.
    pub fn srand_seed(seed: Option<u64>){
        unsafe{
            match seed {
                None => nlopt_srand_time(),
                Some(x) => nlopt_srand(x as c_ulong),
            }
        }
    }

    //Vector storage for limited-memory quasi-Newton algorithms
    ///Some of the NLopt algorithms are limited-memory "quasi-Newton" algorithms, which "remember"
    ///the gradients from a finite number M of the previous optimization steps in order to
    ///construct an approximate 2nd derivative matrix. The bigger M is, the more storage the
    ///algorithms require, but on the other hand they may converge faster for larger M. By default,
    ///NLopt chooses a heuristic value of M, but this can be changed by calling this function.
    ///Passing M=0 (the default) tells NLopt to use a heuristic value. By default, NLopt currently
    ///sets M to 10 or at most 10 MiB worth of vectors, whichever is larger.
    pub fn set_vector_storage(&mut self, m: Option<usize>) -> StrResult {
        unsafe {
            match m {
                None => NLoptOptimizer::<T>::nlopt_res_to_result(nlopt_set_vector_storage(self.opt, 0 as u32)),
                Some(x) => NLoptOptimizer::<T>::nlopt_res_to_result(nlopt_set_vector_storage(self.opt, x as u32)),
            }
        }
    }

    pub fn get_vector_storage(&mut self) -> usize {
        unsafe {
            nlopt_get_vector_storage(self.opt) as usize
        }
    }

    //Preconditioning TODO --> this is somewhat complex but not overly so. Just did not get around
    //to it yet.


    //Version Number
    ///To determine the version number of NLopt at runtime, you can call this function. For
    ///example, NLopt version 3.1.4 would return `(3, 1, 4)`.
    pub fn version() -> (i32,i32,i32) {
        unsafe {
            let mut i: i32 = 0;
            let mut j: i32 = 0;
            let mut k: i32 = 0;
            nlopt_version(&mut i, &mut j, &mut k);
            (i,j,k)
        }
    }

    //NLopt Refernce: http://ab-initio.mit.edu/wiki/index.php/NLopt_Reference

    //helpers
    fn nlopt_res_to_result(ret: i32) -> StrResult {
        match ret {
            x if x < 0 => Err(NLoptOptimizer::<T>::nlopt_result_to_string(x)),
            x => Ok(NLoptOptimizer::<T>::nlopt_result_to_string(x)),
        }
    }

    fn nlopt_result_to_string(n: i32) -> &'static str{
        match n {
           1 => "NLOPT_SUCCESS",
           2 => "NLOPT_STOPVAL_REACHED",
           3 => "NLOPT_FTOL_REACHED",
           4 => "NLOPT_XTOL_REACHED",
           5 => "NLOPT_MAXEVAL_REACHED",
           6 => "NLOPT_MAXTIME_REACHED",
           -1 => "NLOPT_FAILURE",
           -2 => "NLOPT_INVALID_ARGS",
           -3 => "NLOPT_OUT_OF_MEMORY",
           -4 => "NLOPT_ROUNDOFF_LIMITED",
           -5 => "NLOPT_FORCED_STOP",
           _ => "Unknown return value",
        }
    }


    #[no_mangle]
    extern "C" fn function_raw_callback(n:u32,x:*const f64,g:*mut f64,d:*mut c_void) -> f64 {
        let f : &Function<T> = unsafe { &*(d as *const Function<T>) };
        let argument = unsafe { slice::from_raw_parts(x,n as usize) };
        let gradient : Option<&mut [f64]> = unsafe { if g.is_null() { None } else { Some(slice::from_raw_parts_mut(g,n as usize)) } };
        let ret : f64 = ((*f).function)(argument, gradient, (*f).params);
        ret
    }

    #[no_mangle]
    extern "C" fn mfunction_raw_callback(m:u32, re:*mut f64, n:u32, x:*const f64, g:*mut f64, d:*mut c_void) -> i32 {
        let f : &MFunction<T> = unsafe { &*(d as *const MFunction<T>) };
        let argument = unsafe { slice::from_raw_parts(x,n as usize) };
        let gradient : Option<&mut [f64]> = unsafe { if g.is_null() { None } else { Some(slice::from_raw_parts_mut(g,(n as usize)*(m as usize))) } };
        match unsafe { ((*f).function)(slice::from_raw_parts_mut(re,m as usize), argument, gradient, (*f).params) }{
            Ok(_) => 1,
            Err(x) => { println!("Error in mfunction: {}",x); -1 }
        }
    }

    ///Once all of the desired optimization parameters have been specified in a given
    ///`NLoptOptimzer`, you can perform the optimization by calling this function. On input,
    ///`x_init` is an array of length `n` giving an initial
    ///guess for the optimization parameters. On successful return, `x_init` contains the optimized values
    ///of the parameters, and the function returns the corresponding value of the objective function.
    pub fn optimize(&self, x_init:&mut[f64]) -> (StrResult,f64) {
        unsafe {
            let mut min_value : f64 = 0.0;
            (NLoptOptimizer::<T>::nlopt_res_to_result(nlopt_optimize(self.opt, x_init.as_mut_ptr(), &mut min_value)),min_value)
        }
    }
}

impl <T> Drop for NLoptOptimizer<T> {
    fn drop(&mut self) {
        unsafe {
            nlopt_destroy(self.opt);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        println!("Initializing optimizer");
        //initialize the optimizer, choose algorithm, dimensions, target function, user parameters
        let mut opt = NLoptOptimizer::<f64>::new(NLoptAlgorithm::LD_AUGLAG,10,test_objective,NLoptTarget::MINIMIZE,10.0);

        println!("Setting bounds");
        //set lower bounds for the search
        match opt.set_lower_bound(-15.0) {
            Err(_) => panic!("Could not set lower bounds"),
            _ => (),
        };

        println!("Adding inequality constraint");
        match opt.add_constraint(Box::new(Function::<f64>{ function: test_inequality_constraint, params: 120.0, }), ConstraintType::INEQUALITY, 1e-6) {
            Err(x) => panic!("Could not add inequality constraint (Err {})",x),
            _ => (),
        };

        println!("Adding equality constraint");
        match opt.add_constraint(Box::new(Function::<f64>{ function: test_equality_constraint, params: 20.0, }), ConstraintType::EQUALITY, 1e-6) {
            Err(x) => println!("Could not add equality constraint (Err {})",x),
            _ => (),
        };

        match opt.get_lower_bounds() {
            None => panic!("Could not read lower bounds"),
            Some(x) => println!("Lower bounds set to {:?}",x),
        };

        println!("got stopval = {}",opt.get_stopval());

        match opt.set_maxeval(100) {
            Err(_) => panic!("Could not set maximum evaluations"),
            _ => (),
        };

        let (x,y,z) = NLoptOptimizer::<f64>::version();
        println!("Using nlopt version {}.{}.{}",x,y,z);

        println!("Start optimization...");
        //do the actual optimization
        let mut b : Vec<f64> = vec![100.0;opt.n_dims];
        let (ret,min) = opt.optimize(&mut b);
        match ret {
            Ok(x) => println!("Optimization succeeded. ret = {}, min = {} @ {:?}",x,min,b),
            Err(x) => println!("Optimization failed. ret = {}, min = {} @ {:?}",x,min,b),
        }
    }

    fn test_objective(a:&[f64], g:Option<&mut [f64]>, param:f64) -> f64 {
        match g {
            Some(x) => for (target,value) in (*x).iter_mut().zip(a.iter().map(|f| { (f-param)*2.0 })) {
                *target = value;
            },
            None => (),
        }
        a.iter().map(|x| { (x-param)*(x-param) }).sum()
    }

    fn test_inequality_constraint(a:&[f64], g:Option<&mut [f64]>, param:f64) -> f64 {
        match g {
            Some(x) => {    for (index, mut value) in x.iter_mut().enumerate() {
                                *value = match index { 5 => -1.0, _ => 0.0 };
                            }},
            None => (),
        };
        param-a[5]
    }

    fn test_equality_constraint(a:&[f64], g:Option<&mut [f64]>, param:f64) -> f64 {
        match g {
            Some(x) => {    for (index, mut value) in x.iter_mut().enumerate() {
                                *value = match index { 4 => -1.0, _ => 0.0 };
                            }},
            None => (),
        };
        param-a[4]
    }

}
