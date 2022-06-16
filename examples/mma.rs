// this example is adapted from the NLopt Tutorial
// at https://nlopt.readthedocs.io/en/latest/NLopt_Tutorial/

use nlopt::*;
use std::f64;

fn main() -> Result<(), FailState> {
    mma_example()?;
    Ok(())
}

// the objective function.
fn myfunc(x: &[f64], grad: Option<&mut [f64]>, _param: &mut ()) -> f64 {
    if let Some(_grad) = grad {
        _grad[0] = 0.0;
        _grad[1] = 0.5 / x[1].sqrt();
    }
    x[1].sqrt()
}

// the constraint function.
fn myconstraint(x: &[f64], grad: Option<&mut [f64]>, data: &mut [f64; 2]) -> f64 {
    let a = data[0];
    let b = data[1];

    if let Some(grad) = grad {
        grad[0] = 3.0 * a * (a * x[0] + b) * (a * x[0] + b);
        grad[1] = -1.0;
    }
    (a * x[0] + b) * (a * x[0] + b) * (a * x[0] + b) - x[1]
}

fn mma_example() -> Result<(), FailState> {
    let mut opt = Nlopt::new(Algorithm::Mma, 2, myfunc, Target::Minimize, ());

    let lb = [f64::NEG_INFINITY, 0.0]; // lower bounds
    opt.set_lower_bounds(&lb)?;

    opt.add_inequality_constraint(myconstraint, [2.0, 0.0], 1.0e-8)?;
    opt.add_inequality_constraint(myconstraint, [-1.0, 1.0], 1.0e-8)?;

    opt.set_xtol_rel(1.0e-4)?;

    let mut x = [1.234, 5.678];
    let res = opt.optimize(&mut x);

    println!("Result: {:?}", res);
    println!("X vals: {:?}", &x[..]);

    Ok(())
}
