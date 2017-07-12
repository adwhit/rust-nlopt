#![allow(dead_code)]

pub enum NLoptOpt {}

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
    fn nlopt_optimize(opt: *mut NLoptOpt, x_init:*mut f64, min_value: *mut f64) -> i32;
    fn nlopt_set_lower_bounds(opt: *mut NLoptOpt, lb: *const f64) -> i32;
    fn nlopt_set_maxeval(opt: *mut NLoptOpt, maxeval: i32) -> i32;
}

pub struct NLoptMinimizer<T> {
    opt: *mut NLoptOpt,
    n_dims: usize,
    params: T,
    function: fn(n_dims: usize,
                 argument: &[f64],
                 gradient: Option<&mut [f64]>,
                 params: T) -> f64,
    lower_bound: Option<Box<[f64]>>,
    upper_bound: Option<Box<[f64]>>,
    maxeval: Option<u32>,
}

struct Function<T> {
    function: fn(n_dims: usize,
                 argument: &[f64],
                 gradient: Option<&mut [f64]>,
                 params: T) -> f64,
    params: T,
}

impl <T> NLoptMinimizer<T> where T: Copy {
    pub fn new(algorithm: NLoptAlgorithm, n_dims: usize, obj: fn(n:usize,a:&[f64],g:Option<&mut [f64]>,ud:T) -> f64, user_data: T) -> NLoptMinimizer<T> {
        unsafe{
            let fb = Box::new(Function{ function: obj, params: user_data });
            let min = NLoptMinimizer {
                opt: nlopt_create(algorithm as i32,n_dims as u32),
                n_dims: n_dims,
                params: user_data,
                function: obj,
                lower_bound: None,
                upper_bound: None,
                maxeval: None,
            };
            let u_data = Box::into_raw(fb) as *const c_void;
            nlopt_set_min_objective(min.opt, NLoptMinimizer::<T>::objective_raw_callback, u_data);
            min
        }
    }

    pub fn set_lower_bound(&mut self, bound: Box<[f64]>) {
        unsafe{
            nlopt_set_lower_bounds(self.opt, (&*bound).as_ptr());
        }
        self.lower_bound = Some(bound);
    }

    pub fn set_maxeval(&mut self, maxeval: u32) {
        self.maxeval = Some(maxeval);
        unsafe{
            let ret = nlopt_set_maxeval(self.opt, maxeval as i32);
            if ret < 0 {
                panic!("Could not set maximum evaluation");
            }
        }
    }

    #[no_mangle]
    extern "C" fn objective_raw_callback(n:u32,x:*const f64,g:*mut f64,d:*mut c_void) -> f64 {
        let f : &Function<T> = unsafe { &*(d as *const Function<T>) };
        let argument = unsafe { slice::from_raw_parts(x,n as usize) };
        let gradient : Option<&mut [f64]> = unsafe { if g.is_null() { None } else { Some(slice::from_raw_parts_mut(g,n as usize)) } };
        let ret : f64 = ((*f).function)(n as usize, argument, gradient, (*f).params);
        ret
    }

    pub fn minimize(&self, x_init:&mut[f64]) -> (i32,f64) {
        unsafe {
            let mut min_value : f64 = 0.0;
            let ret = nlopt_optimize(self.opt, x_init.as_mut_ptr(), &mut min_value); 
            (ret,min_value)
        }
    }
}

impl <T> Drop for NLoptMinimizer<T> {
    fn drop(&mut self) {
        println!("I am being dropped");
        unsafe {
            nlopt_destroy(self.opt);
        }
        println!("still good");
    }
}

#[cfg(test)]
mod tests {

    use NLoptMinimizer;
    use NLoptAlgorithm;

    #[test]
    fn it_works() {
        println!("Initializing optimizer");
        //initialize the optimizer, choose algorithm, dimensions, target function, user parameters
        let mut opt = NLoptMinimizer::<f64>::new(NLoptAlgorithm::LD_LBFGS,10,test_objective,10.0);

        println!("Setting bounds");
        //set lower bounds for the search
        let lb = Box::new([-10.0;10]);
        opt.set_lower_bound(lb);

        opt.set_maxeval(100);

        println!("Start optimization...");
        //do the actual optimization
        let mut a = [100.0;10];
        let (ret,min) = opt.minimize(&mut a);

        //read the results
        println!("ret = {} ({}), min = {}",ret,nlopt_result_to_string(ret),min);
        println!("a = {:?}", a);
    }

    fn test_objective(_:usize, a:&[f64], g:Option<&mut [f64]>, param:f64) -> f64 {
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
