[![crates.io](https://img.shields.io/crates/v/nlopt.svg)](https://crates.io/crates/nlopt)
![Build Status](https://github.com/adwhit/rust-nlopt/workflows/CI/badge.svg)
[![Documentation](https://docs.rs/nlopt/badge.svg)](https://docs.rs/nlopt)

# nlopt

Thin wrapper around the C [`nlopt`](https://nlopt.readthedocs.io/en/latest/) library.

## Building

You will need [`cmake`](https://cmake.org/) to build successfully. It should be easy to find
on your favourite package manager.


## Examples

For a basic usage examples, see the `/examples` directory. Run with
```
cargo run --example bobyqa
cargo run --example mma
```
See also the tests in `src/lib.rs`. Run them with
```
cargo test
```

## Attribution

This library was originally forked from <https://github.com/mithodin/rust-nlopt>.

## License
The Rust code is licensed under MIT.

For convenience, this crate bundles `nlopt` and links it statically. This may have
licensing implications so I refer the user to the [bundled license](nlopt-2.5.0/COPYING)
for more information.
