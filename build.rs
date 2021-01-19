fn main() {
    let dst = cmake::Config::new("./nlopt-2.5.0")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_CXX_COMPILER", "c++")
        .define("NLOPT_CXX", "OFF")
        .define("NLOPT_PYTHON", "OFF")
        .define("NLOPT_OCTAVE", "OFF")
        .define("NLOPT_MATLAB", "OFF")
        .define("NLOPT_GUILE", "OFF")
        .define("NLOPT_SWIG", "OFF")
        .define("NLOPT_LINK_PYTHON", "OFF")
        .build();
    // Lib could be in either of two locations
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());
    // println!("cargo:rustc-link-lib=static=nlopt");
}
