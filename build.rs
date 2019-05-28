fn main() {
    let dst = cmake::Config::new("./nlopt-2.5.0")
        .define("BUILD_SHARED_LIBS", "OFF")
        .build();
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=nlopt");
}
