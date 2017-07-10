#![allow(dead_code)]

pub enum NLoptOpt {}

#[link(name = "nlopt",
       vers = "0.1.0")]
extern "C" {
	fn nlopt_create(algorithm: i32,
			n_dims: u32) -> *mut NLoptOpt;
	fn nlopt_destroy(opt: *mut NLoptOpt);
}

struct NLoptMinimizer<T> {
	opt: *mut NLoptOpt,
	n_dims: u32,
	objective: fn(n_dims: u32,
		      argument: Vec<f64>,
		      gradient: Vec<f64>,
		      user_data: T) -> f64,
	user_data: T,
}

impl <T> NLoptMinimizer <T> {
	pub fn new(algorithm: i32, n_dims: u32, obj: fn(n:u32,a:Vec<f64>,g:Vec<f64>,ud:T) -> f64, user_data: T) -> NLoptMinimizer<T> {
		unsafe{
			NLoptMinimizer {
				opt: nlopt_create(algorithm,n_dims),
				n_dims: n_dims,
				objective: obj,
				user_data: user_data,
			}
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
#[test]
	fn it_works() {
		fn test_objective(n:u32, a:Vec<f64>, g:Vec<f64>, ud:i32) -> f64 {
			0.0
		}

		let opt = NLoptMinimizer::<i32>::new(1,1,test_objective,1);
	}
}
