#![allow(dead_code)]

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
     *         not with older constants for backwards compatibility
     *         */
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
    fn nlopt_set_maxeval(opt: *mut NLoptOpt, maxeval: i32) -> i32;
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
        let bound : Box<[f64]> = vec![0.0;self.n_dims].into();
        unsafe {
            let b = Box::into_raw(bound);
            let ret = nlopt_get_upper_bounds(self.opt, b as *mut f64);
            match ret {
                x if x < 0 => None,
                _ => Some(slice::from_raw_parts(b as *mut f64,self.n_dims))
            }
        }
    }

    pub fn get_lower_bounds(&self) -> Option<&[f64]>{
        let bound : Box<[f64]> = vec![0.0;self.n_dims].into();
        unsafe {
            let b = Box::into_raw(bound);
            let ret = nlopt_get_lower_bounds(self.opt, b as *mut f64);
            match ret {
                x if x < 0 => None,
                _ => Some(slice::from_raw_parts(b as *mut f64,self.n_dims))
            }
        }
    }

    //Nonlinear Constraints
    //UNTESTED
    pub fn add_inequality_constraint(&mut self, constraint: Box<Function<T>>, t: ConstraintType, tolerance: f64) -> Result<i32,i32> {
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
    pub fn add_inequality_mconstraint(&mut self, constraint: Box<MFunction<T>>, t: ConstraintType, tolerance: &[f64]) -> Result<i32,i32> {
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
    pub fn remove_constraints(&mut self) -> Result<i32,i32>{
        unsafe {
            NLoptOptimizer::<T>::nlopt_res_to_result(
                std::cmp::min(
                    nlopt_remove_inequality_constraints(self.opt),
                    nlopt_remove_equality_constraints(self.opt)
                )
            )
        }
    }

    //Stopping Criteria TODO
    pub fn set_maxeval(&mut self, maxeval: u32) -> Result<i32,i32> {
        unsafe{
            let ret = nlopt_set_maxeval(self.opt, maxeval as i32);
            NLoptOptimizer::<T>::nlopt_res_to_result(ret)
        }
    }

    fn nlopt_res_to_result(ret: i32) -> Result<i32,i32> {
        match ret {
            x if x < 0 => Err(x),
            x => Ok(x),
        }
    }


    //Forced Termination TODO
    //Local Optimization TODO
    //Stochastic Population TODO
    //Pseudorandom Numbers TODO
    //Vector storage for limited-memory quasi-Newton algorithms TODO
    //Preconditioning TODO
    //Version Number TODO
    //NLopt Refernce: http://ab-initio.mit.edu/wiki/index.php/NLopt_Reference

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

    pub fn minimize(&self, x_init:&mut[f64]) -> (Result<i32,i32>,f64) {
        unsafe {
            let mut min_value : f64 = 0.0;
            let ret = nlopt_optimize(self.opt, x_init.as_mut_ptr(), &mut min_value); 
            match ret < 0 {
                true => (Err(ret),min_value),
                false => (Ok(ret),min_value),
            }
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
        let mut opt = NLoptOptimizer::<f64>::new(NLoptAlgorithm::LD_LBFGS,10,test_objective,NLoptTarget::MINIMIZE,10.0);

        println!("Setting bounds");
        //set lower bounds for the search
        match opt.set_lower_bound(-15.0) {
            Err(_) => panic!("Could not set lower bounds"),
            _ => (),
        };

        match opt.get_lower_bounds() {
            None => panic!("Could not read lower bounds"),
            Some(x) => println!("Bounds set to {:?}",x),
        };

        match opt.set_maxeval(100) {
            Err(_) => panic!("Could not set maximum evaluations"),
            _ => (),
        };

        println!("Start optimization...");
        //do the actual optimization
        let mut b : Vec<f64> = vec![100.0;opt.n_dims];
        let (ret,min) = opt.minimize(&mut b);
        match ret {
            Ok(x) => println!("Optimization succeeded. ret = {} ({}), min = {} @ {:?}",x,nlopt_result_to_string(x),min,b),
            Err(x) => println!("Optimization failed. ret = {} ({}), min = {} @ {:?}",x,nlopt_result_to_string(x),min,b),
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

    fn nlopt_result_to_string(n: i32) -> String{
        match n {
           1 => String::from("NLOPT_SUCCESS"),
           2 => String::from("NLOPT_STOPVAL_REACHED"),
           3 => String::from("NLOPT_FTOL_REACHED"),
           4 => String::from("NLOPT_XTOL_REACHED"),
           5 => String::from("NLOPT_MAXEVAL_REACHED"),
           6 => String::from("NLOPT_MAXTIME_REACHED"),
           -1 => String::from("NLOPT_FAILURE"),
           -2 => String::from("NLOPT_INVALID_ARGS"),
           -3 => String::from("NLOPT_OUT_OF_MEMORY"),
           -4 => String::from("NLOPT_ROUNDOFF_LIMITED"),
           -5 => String::from("NLOPT_FORCED_STOP"),
           _ => String::from("Unknown return value"),
        }
    }
}
