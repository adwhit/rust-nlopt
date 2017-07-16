#![allow(dead_code)]

//! # Rust-NLopt
//! 
//! This is a wrapper for the NLopt library (http://ab-initio.mit.edu/wiki/index.php/NLopt).

pub enum NLoptOpt {}

pub enum NLoptTarget {
    MAXIMIZE,
    MINIMIZE,
}

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
    fn nlopt_add_inequality_mconstraint(opt: *mut NLoptOpt, m:u32, fc: extern "C" fn(m:u32, r:*mut f64, n:u32,x:*const f64,g:*mut f64,d:*mut c_void) -> f64, d: *const c_void, tol:*const f64) -> i32;
    fn nlopt_add_equality_mconstraint(opt: *mut NLoptOpt, m:u32, fc: extern "C" fn(m:u32, r:*mut f64, n:u32,x:*const f64,g:*mut f64,d:*mut c_void) -> f64, d: *const c_void, tol:*const f64) -> i32;
    fn nlopt_remove_inequality_constraints(opt: *mut NLoptOpt) -> i32;
    fn nlopt_remove_equality_constraints(opt: *mut NLoptOpt) -> i32;
    fn nlopt_set_stopval(opt: *mut NLoptOpt, stopval:f64) -> i32;
    fn nlopt_get_stopval(opt: *mut NLoptOpt) -> f64;
    fn nlopt_set_ftol_rel(opt: *mut NLoptOpt, tol: f64) -> i32;
    fn nlopt_get_ftol_rel(opt: *mut NLoptOpt) -> f64;
    fn nlopt_set_ftol_abs(opt: *mut NLoptOpt, tol: f64) -> i32;
    fn nlopt_get_ftol_abs(opt: *mut NLoptOpt) -> f64;
    fn nlopt_set_xtol_rel(opt: *mut NLoptOpt, tol: f64) -> i32;
    fn nlopt_get_xtol_rel(opt: *mut NLoptOpt) -> f64;
    fn nlopt_set_xtol_abs(opt: *mut NLoptOpt, tol: *const f64) -> i32;
    fn nlopt_get_xtol_abs(opt: *mut NLoptOpt, tol: *mut f64) -> i32;
    fn nlopt_set_maxeval(opt: *mut NLoptOpt, maxeval: i32) -> i32;
    fn nlopt_get_maxeval(opt: *mut NLoptOpt) -> i32;
    fn nlopt_set_maxtime(opt: *mut NLoptOpt, maxtime: f64) -> i32;
    fn nlopt_get_maxtime(opt: *mut NLoptOpt) -> f64;
    fn nlopt_force_stop(opt: *mut NLoptOpt) -> i32;
    fn nlopt_set_force_stop(opt: *mut NLoptOpt, val: i32) -> i32;
    fn nlopt_get_force_stop(opt: *mut NLoptOpt) -> i32;
    fn nlopt_set_local_optimizer(opt: *mut NLoptOpt, local_opt: *mut NLoptOpt) -> i32;
    fn nlopt_set_population(opt: *mut NLoptOpt, pop: u32) -> i32;
    fn nlopt_get_population(opt: *mut NLoptOpt) -> u32;
    fn nlopt_set_initial_step(opt: *mut NLoptOpt, dx: *const f64) -> i32;
    fn nlopt_get_initial_step(opt: *mut NLoptOpt, x: *const f64, dx: *mut f64) -> i32;
    fn nlopt_srand(seed: c_ulong);
    fn nlopt_srand_time();
}

pub struct NLoptOptimizer<T> {
    opt: *mut NLoptOpt,
    n_dims: usize,
    params: T,
    function: NLoptFn<T>,
}

pub type NLoptFn<T> = fn(argument: &[f64],
                     gradient: Option<&mut [f64]>,
                     params: T) -> f64;

type StrResult = Result<&'static str,&'static str>;

pub struct Function<T> {
    function: NLoptFn<T>,
    params: T,
}

pub enum ConstraintType {
    EQUALITY,
    INEQUALITY,
}

struct Constraint<F> {
    function: F,
    ctype: ConstraintType,
}

pub type NLoptMFn<T> = fn(result: &mut [f64],
                      argument: &[f64],
                      gradient: Option<&mut [f64]>,
                      params: T) -> f64;

pub struct MFunction<T> {
    m: usize,
    function: NLoptMFn<T>,
    params: T,
}

impl <T> NLoptOptimizer<T> where T: Copy {
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

    pub fn set_lower_bound(&mut self, bound: f64) -> Result<i32,i32>{
        let v = vec![bound;self.n_dims];
        self.set_lower_bounds(&v)
    }

    pub fn set_upper_bound(&mut self, bound: f64) -> Result<i32,i32>{
        let v = vec![bound;self.n_dims];
        self.set_upper_bounds(&v)
    }

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

    pub fn set_xtol_abs(&mut self, tolerance: &[f64]) -> StrResult{
        unsafe {
            NLoptOptimizer::<T>::nlopt_res_to_result(nlopt_set_xtol_abs(self.opt, tolerance.as_ptr()))
        }
    }

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
    pub fn set_local_optimizer(&mut self, local_opt: NLoptOptimizer<T>) -> StrResult{
        unsafe {
            NLoptOptimizer::<T>::nlopt_res_to_result( nlopt_set_local_optimizer(self.opt, local_opt.opt) )
        }
    }
    
    //Initial Step Size
    pub fn set_initial_step(&mut self, dx: &[f64]) -> StrResult{
        unsafe {
            NLoptOptimizer::<T>::nlopt_res_to_result(nlopt_set_initial_step(self.opt, dx.as_ptr()))
        }
    }

    pub fn set_initial_step1(&mut self, dx: f64) -> StrResult{
        let d : &[f64] = &vec![dx;self.n_dims];
        self.set_initial_step(d)
    }
    
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
    pub fn srand_seed(seed: Option<u64>){
        unsafe{
            match seed {
                None => nlopt_srand_time(),
                Some(x) => nlopt_srand(x as c_ulong),
            }
        }
    }

    //Vector storage for limited-memory quasi-Newton algorithms TODO
    //Preconditioning TODO
    //Version Number TODO
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
    extern "C" fn mfunction_raw_callback(m:u32, re:*mut f64, n:u32, x:*const f64, g:*mut f64, d:*mut c_void) -> f64 {
        let f : &MFunction<T> = unsafe { &*(d as *const MFunction<T>) };
        let argument = unsafe { slice::from_raw_parts(x,n as usize) };
        let gradient : Option<&mut [f64]> = unsafe { if g.is_null() { None } else { Some(slice::from_raw_parts_mut(g,(n as usize)*(m as usize))) } };
        let ret : f64 = unsafe { ((*f).function)(slice::from_raw_parts_mut(re,m as usize), argument, gradient, (*f).params) };
        ret
    }

    pub fn minimize(&self, x_init:&mut[f64]) -> (StrResult,f64) {
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
        let mut opt = NLoptOptimizer::<f64>::new(NLoptAlgorithm::LD_MMA,10,test_objective,NLoptTarget::MINIMIZE,10.0);

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

        println!("Start optimization...");
        //do the actual optimization
        let mut b : Vec<f64> = vec![100.0;opt.n_dims];
        let (ret,min) = opt.minimize(&mut b);
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
