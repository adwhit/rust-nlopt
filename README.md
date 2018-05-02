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

## Static linking

The recommended way to use this library is to link it statically.

To do this, build `nlopt` normally (with `cmake`), then run
```
NLOPT_DIR=/path/to/nlopt/build
ar -r $NLOPT_DIR/libnlopt.a $NLOPT_DIR/libnlopt.so
```
to create an archive. Then
