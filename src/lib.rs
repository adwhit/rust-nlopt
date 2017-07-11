#![allow(dead_code)]

pub enum NLoptOpt {}

extern crate libc;
use std::slice;

#[link(name = "nlopt",
       vers = "0.1.0")]
extern "C" {
	fn nlopt_create(algorithm: i32,
			n_dims: u32) -> *mut NLoptOpt;
	fn nlopt_destroy(opt: *mut NLoptOpt);
    fn nlopt_set_min_objective(opt: *mut NLoptOpt, nlopt_fdf: extern "C" fn(n:u32,x:*const f64,g:*mut f64,d:*mut libc::c_void) -> f64, d:*mut libc::c_void);
}

struct NLoptMinimizer<T> {
	opt: *mut NLoptOpt,
	n_dims: u32,
	objective: fn(n_dims: u32,
		      argument: &[f64],
		      gradient: &mut [f64],
		      user_data: T) -> f64,
	user_data: T,
}

impl <T> NLoptMinimizer <T> where T: Clone {
	pub fn new(algorithm: i32, n_dims: u32, obj: fn(n:u32,a:&[f64],g:&mut [f64],ud:T) -> f64, user_data: T) -> NLoptMinimizer<T> {
		unsafe{
			let mut min = NLoptMinimizer {
				opt: nlopt_create(algorithm,n_dims),
				n_dims: n_dims,
				objective: obj,
				user_data: user_data,
			};

            extern "C" fn objective_raw_callback<Ti:Clone>(n:u32,x:*const f64,g:*mut f64,d:*mut libc::c_void) -> f64 {
                let min : &mut NLoptMinimizer<Ti> = unsafe { &mut *(d as *mut NLoptMinimizer<Ti>) };
                let argument = unsafe { slice::from_raw_parts(x,n as usize) };
                let mut gradient = unsafe { slice::from_raw_parts_mut(g,n as usize) };
                let ret : f64 = (min.objective)(n, argument, gradient, min.user_data.clone());
                ret
            }

            let u_data:*mut libc::c_void = &mut min as *mut _ as *mut libc::c_void;
            nlopt_set_min_objective(min.opt, objective_raw_callback::<T>, u_data);
            min
		}
	}

    pub fn minimize() -> i32 {
        1
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
#[test]
	fn it_works() {
		fn test_objective(_:u32, a:&[f64], g:&mut [f64], ud:i32) -> f64 {
			for mut gi in g{
                *gi = 0.0;
            }
            a.iter().map(|x| { x*x*(ud as f64) }).sum()
		}

		let opt = NLoptMinimizer::<i32>::new(2,1,test_objective,1);
        let a = [1.0,2.0];
        let mut g = [1000.0,1.0];
        let ret = (opt.objective)(1,&a,&mut g,1);
        println!("ret = {}",ret);
        println!("g = {:?}", g);
        println!("a = {:?}", a);
	}
}
