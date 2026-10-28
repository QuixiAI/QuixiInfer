//! QuixiInfer's CPU kernels: the expert pool's quantized dot products and its worker threads.
#![allow(unsafe_op_in_unsafe_fn)]
pub mod act;
pub mod dot;
pub mod math;
pub mod pool;
