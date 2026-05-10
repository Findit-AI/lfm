//! Bench the tile-grid selection algorithm — pure CPU, no model files needed.
//!
//! Run: `cargo bench --bench bench_tile_grid`

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

fn bench_tile_grid(c: &mut Criterion) {
  let budget = lfm::ImageBudget::default();

  c.bench_function("pick_tile_grid_1920x1080_landscape", |b| {
    b.iter(|| {
      let _ = black_box(lfm::preproc::tile_grid::pick_tile_grid(
        black_box(1920),
        black_box(1080),
        &budget,
      ));
    });
  });

  c.bench_function("pick_tile_grid_1024x1024_square", |b| {
    b.iter(|| {
      let _ = black_box(lfm::preproc::tile_grid::pick_tile_grid(
        black_box(1024),
        black_box(1024),
        &budget,
      ));
    });
  });

  c.bench_function("pick_tile_grid_256x256_single_tile", |b| {
    b.iter(|| {
      let _ = black_box(lfm::preproc::tile_grid::pick_tile_grid(
        black_box(256),
        black_box(256),
        &budget,
      ));
    });
  });
}

criterion_group!(benches, bench_tile_grid);
criterion_main!(benches);
