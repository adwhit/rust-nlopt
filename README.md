# nlopt

Thin wrapper around the C [`nlopt`](https://nlopt.readthedocs.io/en/latest/) library.

Note: Most functionality has been implemented, but not all has been tested working.

## [Docs](https://docs.rs/nlopt)

## Building

This crate depends upon `nlopt` and will fail if it cannot find a library to link against. It has been tested against `nlopt v2.5.0` - it may or may not work against other versions.

The source can be downloaded from the [official site](https://nlopt.readthedocs.io/en/latest/), which also has provides build instructions.

Note you may find it more convenient the build `nlopt` as a static library, by passing `-DBUILD_SHARED_LIBS=OFF` to `cmake`.

The resulting C-lib must be on the search path at link-time. This can be set with
environment variables, the `rustc` command or a `build.rs` script.

## Tests

```
cargo test
```
(This is a quick way to check the C-lib can be found).

## Examples

For a basic usage example, see `examples/bobyqa.rs`. Run with
```
cargo run --example bobyqa
```

See also the tests in `src/lib.rs`

## Attribution

This library was originally forked from <https://github.com/mithodin/rust-nlopt>.
