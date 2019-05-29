fn main() {
    let dst = cmake::Config::new("./nlopt-2.5.0")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_CXX_COMPILER", "c++") // Not used
        // The default C++ flags mess things up - override them
        .define("CMAKE_CXX_FLAGS", "")
        .build();
    // Lib could be in either of two locations
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());
    // println!("cargo:rustc-link-lib=static=nlopt");
}
