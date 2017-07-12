#![allow(dead_code)]

pub enum NLoptOpt {}

#[repr(C)]
pub enum NLoptAlgorithm {
    NLOPT_GN_DIRECT = 0,
    NLOPT_GN_DIRECT_L,
    NLOPT_GN_DIRECT_L_RAND,
    NLOPT_GN_DIRECT_NOSCAL,
    NLOPT_GN_DIRECT_L_NOSCAL,
    NLOPT_GN_DIRECT_L_RAND_NOSCAL,

    NLOPT_GN_ORIG_DIRECT,
    NLOPT_GN_ORIG_DIRECT_L,

    NLOPT_GD_STOGO,
    NLOPT_GD_STOGO_RAND,

    NLOPT_LD_LBFGS_NOCEDAL,

    NLOPT_LD_LBFGS,

    NLOPT_LN_PRAXIS,

    NLOPT_LD_VAR1,
    NLOPT_LD_VAR2,

    NLOPT_LD_TNEWTON,
    NLOPT_LD_TNEWTON_RESTART,
    NLOPT_LD_TNEWTON_PRECOND,
    NLOPT_LD_TNEWTON_PRECOND_RESTART,

    NLOPT_GN_CRS2_LM,

    NLOPT_GN_MLSL,
    NLOPT_GD_MLSL,
    NLOPT_GN_MLSL_LDS,
    NLOPT_GD_MLSL_LDS,

    NLOPT_LD_MMA,

    NLOPT_LN_COBYLA,

    NLOPT_LN_NEWUOA,
    NLOPT_LN_NEWUOA_BOUND,

    NLOPT_LN_NELDERMEAD,
    NLOPT_LN_SBPLX,

    NLOPT_LN_AUGLAG,
    NLOPT_LD_AUGLAG,
    NLOPT_LN_AUGLAG_EQ,
    NLOPT_LD_AUGLAG_EQ,

    NLOPT_LN_BOBYQA,

    NLOPT_GN_ISRES,

    /* new variants that require local_optimizer to be set,
     *         not with older constants for backwards compatibility
     *         */
    NLOPT_AUGLAG,
    NLOPT_AUGLAG_EQ,
    NLOPT_G_MLSL,
    NLOPT_G_MLSL_LDS,

    NLOPT_LD_SLSQP,

    NLOPT_LD_CCSAQ,

    NLOPT_GN_ESCH,

    NLOPT_NUM_ALGORITHMS /* not an algorithm, just the number of them */
}

extern crate libc;
use std::slice;

#[link(name = "nlopt",
       vers = "0.1.0")]
extern "C" {
    fn nlopt_create(algorithm: i32,
                    n_dims: u32) -> *mut NLoptOpt;
    fn nlopt_destroy(opt: *mut NLoptOpt);
    fn nlopt_set_min_objective(opt: *mut NLoptOpt, nlopt_fdf: extern "C" fn(n:u32,x:*const f64,g:*mut f64,d:*mut libc::c_void) -> f64, d:*const libc::c_void);
    fn nlopt_optimize(opt: *mut NLoptOpt, x_init:*mut f64, min_value: *mut f64) -> i32;
}

pub struct NLoptMinimizer<T> {
    opt: *mut NLoptOpt,
    n_dims: usize,
    objective: fn(n_dims: usize,
                  argument: &[f64],
                  gradient: Option<&mut [f64]>,
                  user_data: T) -> f64,
    user_data: T,
}

impl <T> NLoptMinimizer <T> where T: Clone {
    pub fn new(algorithm: NLoptAlgorithm, n_dims: usize, obj: fn(n:usize,a:&[f64],g:Option<&mut [f64]>,ud:T) -> f64, user_data: T) -> NLoptMinimizer<T> {
        unsafe{
            let min = NLoptMinimizer {
                opt: nlopt_create(algorithm as i32,n_dims as u32),
                n_dims: n_dims,
                objective: obj,
                user_data: user_data,
            };
            //testing
            let ob = min.objective;
            let a = [1.0;10];
            let ret = ob(n_dims,&a,None,min.user_data.clone());
            println!("{}",ret);
            //end
            let u_data:*const libc::c_void = & min as *const _ as *const libc::c_void;
            nlopt_set_min_objective(min.opt, NLoptMinimizer::<T>::objective_raw_callback, u_data);
            min
        }
    }

    #[no_mangle]
    extern "C" fn objective_raw_callback(n:u32,x:*const f64,g:*mut f64,d:*mut libc::c_void) -> f64 {
        let min : & NLoptMinimizer<T> = unsafe { & *(d as *const NLoptMinimizer<T>) };
        let argument = unsafe { slice::from_raw_parts(x,n as usize) };
        let gradient : Option<&mut [f64]> = unsafe { if g.is_null() { None } else { Some(slice::from_raw_parts_mut(g,n as usize)) } };
        let ob = min.objective;
        let ret : f64 = ob(n as usize, argument, gradient, min.user_data.clone());
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
        unsafe {
            nlopt_destroy(self.opt);
        }
    }
}

#[cfg(test)]
mod tests {

    use NLoptMinimizer;
    use NLoptAlgorithm;

    #[test]
    fn it_works() {
        let opt = NLoptMinimizer::<i32>::new(NLoptAlgorithm::NLOPT_LD_LBFGS,10,test_objective,1);
        let mut a = [1.0;10];
        let (ret,min) = opt.minimize(&mut a);
        println!("ret = {}, min = {}",ret,min);
        //println!("a = {:?}", a);
    }

    pub fn test_objective(n:usize, a:&[f64], g:Option<&mut [f64]>, _:i32) -> f64 {
        match g {
            Some(x) => for i in 0..n {
                x[i] = a[i]*2.0;
            },
                None => (),
        }
        a.iter().map(|x| { x*x }).sum()
    }
}
