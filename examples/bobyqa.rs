// this example is adapted from the original bobyqa source
// at http://mat.uc.pt/~zhang/software.html

use nlopt::*;

use std::f64::consts::PI;

fn main() {
    bobyqa_example()
}

fn objfn(x: &[f64], _gradient: Option<&mut [f64]>, _params: &mut ()) -> f64 {
    let mut f = 0.0;
    for i in (4..=x.len()).step_by(2) {
        let j1 = i - 2;
        for j in (2..=j1).step_by(2) {
            let tmpa = x[i - 2] - x[j - 2];
            let tmpb = x[i - 1] - x[j - 1];
            let tmp = tmpa * tmpa + tmpb * tmpb;
            f += 1.0 / tmp.max(1e-6).sqrt()
        }
    }
    f
}

fn bobyqa_example() {
    let twopi = 2.0 * PI;

    let mut x = [0.0; 100];
    let mut xl = [0.0; 100];
    let mut xu = [0.0; 100];

    let bdl = -1.0;
    let bdu = 1.0;

    for &m in &[5, 10] {
        let q = twopi / m as f64;
        let n = 2 * m;
        for i in 1..=n {
            xl[i - 1] = bdl;
            xu[i - 1] = bdu;
        }
        for &jcase in &[1, 2] {
            let npt = if jcase == 2 { 2 * n + 1 } else { n + 6 };
            println!("2D output with M={}, N={} and NPT={}", m, n, npt);
            for j in 1..=m {
                let temp = (j as f64) * q;
                x[2 * j - 2] = temp.cos();
                x[2 * j - 1] = temp.sin();
            }
            let mut opt = Nlopt::new(Algorithm::Bobyqa, n, objfn, Target::Minimize, ());
            opt.set_lower_bounds(&xl).unwrap();
            opt.set_upper_bounds(&xu).unwrap();
            opt.set_xtol_rel(1e-6).unwrap();

            let res = opt.optimize(&mut x);
            println!("Result: {:?}", res);
            println!("X vals: {:?}\n", &x[..n]);
        }
    }
}
