# nlopt

Thin wrapper around the C [`nlopt`](https://nlopt.readthedocs.io/en/latest/) library.

Note: Not all functionality has been tested.

## [Docs](https://docs.rs/nlopt)

## Building

This library depends upon [nlopt](http://nlopt.readthedocs.io) and will fail if it cannot find a library to link against.

For Linux, it is recommended to clone `nlopt` from [github](http://github.com/stevengj/nlopt) (the official release is
many years behind master) and follow the [installation docs](https://nlopt.readthedocs.io/en/latest/NLopt_Installation/).
You may find it more convenient the build `nlopt` as a static library, by passing `-DBUILD_SHARED_LIBS=OFF` to `cmake`.

Windows is a bit more tricky. One way is to [download](http://ab-initio.mit.edu/nlopt/nlopt-2.4.2-dll64.zip) the precompiled
binary and turn it into `nlopt.lib` by executing
```
lib.exe /def:libnlopt-0.def /out:nlopt.lib /MACHINE:x64`
```
(This requires the Visual Studio development tools to be installed).

For either platform, the resulting object must be on the search path at link-time. This can be set with
environment variables, the `rustc` command or a `build.rs` script.

## Tests

```
cargo test
```

## Examples

For a basic usage example, see `examples/bobyqa.rs`. Run with
```
cargo run --example bobyqa
```

See also the tests in `src/lib.rs`
