# nlopt

Thin wrapper around the C [`nlopt`](https://nlopt.readthedocs.io/en/latest/) library.

Note: Has not been extensively tested.

## Building

This library depends upon [nlopt](http://nlopt.readthedocs.io) and will fail if it cannot find a library to link against.

For Linux, it is recommended to clone `nlopt` from [github](http://github.com/stevengj/nlopt) (the official release is
many years behind master) and follow the installation instructions.

Windows is a bit more tricky. One way is to [download](http://ab-initio.mit.edu/nlopt/nlopt-2.4.2-dll64.zip) the precompiled
binary and turn it into `nlopt.lib` by executing
```
lib.exe /def:libnlopt-0.def /out:nlopt.lib /MACHINE:x64`
```
(This requires the Visual Studio development tools to be installed).

For either platform, the resulting object must be on the search path at link-time. This can be set with
environment variables or a `build.rs` script.

An example build script which would do the job is

```
fn main() {
    println!(r"cargo:rustc-link-search=C:\path\to\nlopt");
}
```

## Tests

This can be tricky because it's necessary to pass in linker arguments, and without `build.rs` this may
not work with a simple `cargo test` command. One quick workaround is:
```
cargo rustc --tests -- -L /path/to/nlopt
target/debug/nlopt-<SOME-HASH>
```
