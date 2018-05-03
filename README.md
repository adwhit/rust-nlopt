# nlopt

This Rust crate is a thin wrapper around the C `nlopt` library.

Before running, make sure that libnlopt.so is on your `LD_LIBRARY_PATH`.


## Tests

This is a bit tricky because it's necessary to pass in linker arguments, so
`cargo test` won't work. One quick workaround is:
```
cargo rustc --tests -- -L /home/alex/repo/nlopt/build
target/debug/nlopt-<SOME-HASH>
```
