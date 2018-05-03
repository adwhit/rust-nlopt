extern crate nlopt;
extern crate num_iter;

use num_iter::*;
use nlopt::*;

use std::f64::consts::PI;

fn main() {
    bobyqa_test()
}

fn objfn(x: &[f64], _gradient: Option<&mut [f64]>, params: ()) -> f64 {
    let mut f = 0.0;
    for i in range_step_inclusive(4, x.len(), 2) {
        let j1 = i - 2;
        for j in range_step_inclusive(2, j1, 2) {
            let tmpa = x[i-2] - x[j - 2];
            let tmpb = x[i-1] - x[j - 1];
            let tmp = tmpa * tmpa + tmpb * tmpb;
            f += 1.0/tmp.max(1e-6).sqrt()
        }
    }
    f
}

fn bobyqa_test() {
    /* Constants. */
    let twopi = 2.0 * PI;

    /* Local variables. */
    // REAL bdl, bdu, rhobeg, rhoend, temp;
    // REAL w[500000], x[100], xl[100], xu[100];
    // INTEGER i, iprint, j, jcase, m, maxfun;

    let mut x = [0.0; 100];
    let mut xl = [0.0; 100];
    let mut xu = [0.0; 100];

    let bdl = -1.0;
    let bdu =  1.0;
    let iprint = 2;
    let maxfun = 500000;
    let rhobeg = 0.1;
    let rhoend = 1e-6;
    for &m in &[5, 10] {

    // for (m = 5; m <= 10; m += m) {

        let q = twopi / m as f64;
        let n = 2 * m;

        // REAL q = twopi/(REAL)m;
        // INTEGER n = 2*m;

        for i in 1..=n {
            xl[i - 1] = bdl;
            xu[i - 1] = bdu;
        }

        for &jcase in &[1, 2] {
            let npt = if jcase == 2 {
                2*n + 1
            } else {
                n + 6
            };
            println!("\n\n     2D output with M ={},  N ={}  and  NPT ={}\n", m, n, npt);
            for j in 1..=m {
                let temp = (j as f64)*q;
                x[2*j - 2] = temp.cos();
                x[2*j - 1] = temp.sin();
            }
            let opt = Nlopt::new(Algorithm::LnBobyqa, n, objfn, Target::Maximize, ());
            let (res, score) = opt.optimize(&mut x);
            // bobyqa(n, npt, objfun_test, NULL, x, xl, xu, rhobeg, rhoend,
            //        iprint, maxfun, w);
        }
    }
}
