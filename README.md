# rust-nlopt

[![crates.io](https://img.shields.io/crates/v/nlopt.svg)](https://crates.io/crates/nlopt)
![Build Status](https://github.com/adwhit/rust-nlopt/workflows/CI/badge.svg)
[![Documentation](https://docs.rs/nlopt/badge.svg)](https://docs.rs/nlopt)

A safe and idiomatic Rust wrapper for the [NLopt](https://nlopt.readthedocs.io/) nonlinear optimization library.

## Overview

NLopt is a powerful and versatile library for nonlinear optimization, providing a collection of algorithms for solving optimization problems across various domains. This Rust wrapper offers the following features:

- **Safe Access to NLopt**: Memory-safe wrappers around the C library
- **Complete API Coverage**: All core NLopt functionality exposed through idiomatic Rust interfaces
- **Type Safety**: Strong Rust types for algorithms, constraints, and result states
- **Proper Error Handling**: Error conditions represented as Rust `Result` types
- **Efficient Memory Management**: Automatic cleanup of NLopt resources via Rust's RAII
- **Callback Support**: Seamless integration with Rust closures and user data

## Installation

Add `nlopt` to your Cargo.toml:

```toml
[dependencies]
nlopt = "0.8.0"
```

### Build Requirements

This crate requires [CMake](https://cmake.org/) to build the bundled NLopt C library. Install it using your system's package manager:

- Ubuntu/Debian: `sudo apt install cmake`
- Fedora/RHEL: `sudo dnf install cmake`
- macOS: `brew install cmake`
- Windows: Install CMake from the [official website](https://cmake.org/download/)

## Basic Usage

Here's a simple example that minimizes a quadratic function:

```rust
use nlopt::{Algorithm, Nlopt, Target};

fn main() {
    // Define the objective function - a simple quadratic: f(x) = (x - 3)² + 7
    let objective = |x: &[f64], _gradient: Option<&mut [f64]>, _: &mut ()| {
        let x = x[0];
        (x - 3.0) * (x - 3.0) + 7.0
    };

    // Create a new optimizer using COBYLA algorithm for 1-dimensional problem
    let mut opt = Nlopt::new(Algorithm::Cobyla, 1, objective, Target::Minimize, ());
    
    // Set stopping criteria
    opt.set_xtol_rel(1e-6).unwrap();
    
    // Run the optimization (starting from x=0.0)
    let mut x = [0.0];
    match opt.optimize(&mut x) {
        Ok((status, value)) => {
            println!("Found minimum: x = {}, f(x) = {}", x[0], value);
            println!("Optimization status: {:?}", status);
        },
        Err((error, _)) => {
            println!("Optimization failed: {:?}", error);
        }
    }
}
```

## Comprehensive Examples

For more detailed examples, see the `/examples` directory in the repository:

- **bobyqa.rs**: Demonstrates the gradient-free BOBYQA optimizer
- **mma.rs**: Shows constrained optimization with the MMA algorithm

Run the examples with:

```bash
cargo run --example bobyqa
cargo run --example mma
```

## Features

### Supported Optimization Algorithms

The library supports all NLopt algorithms (43 in total), broadly categorized into:

- **Global Optimization**: DIRECT, CRS, MLSL, ISRES, and more
- **Local Optimization**: BOBYQA, COBYLA, LBFGS, MMA, Nelder-Mead, and more
- **Gradient-Based**: Methods that use derivatives for faster convergence
- **Derivative-Free**: Methods that only require function evaluations

### Constraint Handling

The library supports various types of constraints:

- **Bounds**: Simple upper/lower bounds on variables
- **Nonlinear Inequality Constraints**: Constraints of the form g(x) ≤ 0
- **Nonlinear Equality Constraints**: Constraints of the form h(x) = 0
- **Vector Constraints**: Efficiently handle multiple constraints at once

### Stopping Criteria

Control termination with various stopping criteria:

- Function value targets
- Relative/absolute tolerances on function values
- Relative/absolute tolerances on optimization parameters
- Maximum evaluations/time limits

## Documentation

For detailed API documentation, visit [docs.rs/nlopt](https://docs.rs/nlopt).

For information about the underlying NLopt algorithms, visit the [NLopt documentation](https://nlopt.readthedocs.io/).

## License

This Rust wrapper is licensed under the MIT License.

The bundled NLopt C library (version 2.9.1) is included for convenience and is linked statically. Please refer to the [NLopt license](nlopt-2.9.1/COPYING) for its licensing terms.

## Attribution

This library was originally forked from https://github.com/mithodin/rust-nlopt.