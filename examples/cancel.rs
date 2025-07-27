use nlopt::*;

fn main() {
    let mut opt = Nlopt::new(
        Algorithm::Cobyla,
        1,
        |x, _, _| {
            if x[0] > 32. {
                // `Option<f64>` implemenents termination, so when we return `None` the
                // optimization gets cancelled
                return None;
            }
            Some(x[0])
        },
        // Target::Maximize, // using this will cancel in the cost function
        Target::Minimize, // using this will cancel in the constraint
        (),
    );

    opt.add_inequality_constraint(
        |x, _, _| {
            // Constraints function can have different termination type than the cost function. In
            // this case we use a `Result<f64, impl Into<i32>>`
            if x[0] < -64. {
                return Err(Reason::TooSmall);
            }
            if x[0] > 64. {
                return Err(Reason::TooBig);
            }
            Ok(-1.)
        },
        (),
        1e-9,
    )
    .unwrap();
    opt.set_xtol_rel(1.0e-4).unwrap();

    let mut x = [0.];
    let result = opt.optimize(&mut x);
    let reason = opt.get_force_stop();

    println!("Result={result:?}, reason={reason:?}");
}

// Custom Cancellation error type, which needs to convertible to i32
enum Reason {
    TooSmall = 1,
    TooBig = 2,
}

impl From<Reason> for i32 {
    fn from(value: Reason) -> Self {
        value as i32
    }
}
