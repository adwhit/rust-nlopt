extern crate nlopt;
use nlopt::*;

fn main() {
    auglag_example()
}

fn auglag_example() {
    println!("Initializing optimizer");
    //initialize the optimizer, choose algorithm, dimensions, target function, user parameters
    let mut opt = Nlopt::<f64>::new(
        Algorithm::Auglag,
        10,
        example_objective,
        Target::Minimize,
        10.0,
    );

    println!("Setting bounds");
    //set lower bounds for the search
    match opt.set_lower_bound(-15.0) {
        Err(_) => panic!("Could not set lower bounds"),
        _ => (),
    };

    println!("Adding inequality constraint");
    match opt.add_constraint(
        Box::new(Function::<f64> {
            function: example_inequality_constraint,
            params: 120.0,
        }),
        ConstraintType::Inequality,
        1e-6,
    ) {
        Err(x) => panic!("Could not add inequality constraint (Err {:?})", x),
        _ => (),
    };

    println!("Adding equality constraint");
    match opt.add_constraint(
        Box::new(Function::<f64> {
            function: example_equality_constraint,
            params: 20.0,
        }),
        ConstraintType::Equality,
        1e-6,
    ) {
        Err(x) => println!("Could not add equality constraint (Err {:?})", x),
        _ => (),
    };

    match opt.get_lower_bounds() {
        None => panic!("Could not read lower bounds"),
        Some(x) => println!("Lower bounds set to {:?}", x),
    };

    println!("got stopval = {}", opt.get_stopval());

    match opt.set_maxeval(100) {
        Err(_) => panic!("Could not set maximum evaluations"),
        _ => (),
    };

    let (x, y, z) = Nlopt::<f64>::version();
    println!("Using nlopt version {}.{}.{}", x, y, z);

    println!("Start optimization...");

    //do the actual optimization
    let mut b: Vec<f64> = vec![100.0; opt.n_dims];
    let ret = opt.optimize(&mut b);
    match ret {
        Ok((s, min)) => println!(
            "Optimization succeeded. ret = {:?}, min = {} @ {:?}",
            s, min, b
        ),
        Err((e, min)) => panic!(
                "Optimization failed. ret = {:?}, min = {} @ {:?}",
                e, min, b)
    }
}

fn example_objective(a: &[f64], g: Option<&mut [f64]>, param: f64) -> f64 {
    match g {
        Some(x) => {
            for (target, value) in (*x).iter_mut().zip(a.iter().map(|f| (f - param) * 2.0)) {
                *target = value;
            }
        }
        None => (),
    }
    a.iter().map(|x| (x - param) * (x - param)).sum()
}

fn example_inequality_constraint(a: &[f64], g: Option<&mut [f64]>, param: f64) -> f64 {
    match g {
        Some(x) => {
            for (index, mut value) in x.iter_mut().enumerate() {
                *value = match index {
                    5 => -1.0,
                    _ => 0.0,
                };
            }
        }
        None => (),
    };
    param - a[5]
}

fn example_equality_constraint(a: &[f64], g: Option<&mut [f64]>, param: f64) -> f64 {
    match g {
        Some(x) => {
            for (index, mut value) in x.iter_mut().enumerate() {
                *value = match index {
                    4 => -1.0,
                    _ => 0.0,
                };
            }
        }
        None => (),
    };
    param - a[4]
}
