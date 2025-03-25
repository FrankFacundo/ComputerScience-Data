#!/usr/bin/env run-cargo-script
//! ```cargo
//! [dependencies]
//! rayon = "1.7"
//! ```

extern crate rayon;

use std::time::Instant;
use rayon::prelude::*;

fn main() {
    let iterations = 100_000_000_000;

    // Sequential benchmark
    let start_seq = Instant::now();
    let mut sum_seq = 0;
    for i in 0..iterations {
        sum_seq += i;
    }
    let duration_seq = start_seq.elapsed();
    println!("Sequential loop: sum = {}, time elapsed = {:?}", sum_seq, duration_seq);

    // Parallel benchmark using Rayon
    let start_par = Instant::now();
    let sum_par: usize = (0..iterations).into_iter().sum();
    let duration_par = start_par.elapsed();
    println!("Parallel loop: sum = {}, time elapsed = {:?}", sum_par, duration_par);
}
