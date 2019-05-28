# nlopt

Thin wrapper around the C [`nlopt`](https://nlopt.readthedocs.io/en/latest/) library.

## [Docs](https://docs.rs/nlopt)

## Building

You will need [`cmake`](https://cmake.org/) to build successfully. It should be easy to find
on your favourite package manager.


## Examples

For a basic usage example, see `examples/bobyqa.rs`. Run with
```
cargo run --example bobyqa
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
