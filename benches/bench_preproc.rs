//! Bench preprocessing throughput with synthetic images at fixed sizes.
//!
//! Run: `cargo bench --bench bench_preproc`

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use image::{DynamicImage, RgbImage};

fn bench_preproc(c: &mut Criterion) {
  let budget = lfm::ImageBudget::default();
  let preproc = lfm::Preprocessor::new(budget);

  // 1024×1024 routes through the multi-tile path (4 tiles + thumbnail
  // at default budget), giving a realistic workload.
  let img_1024 = DynamicImage::ImageRgb8(RgbImage::new(1024, 1024));

  // 256×256 routes through the single-tile path.
  let img_256 = DynamicImage::ImageRgb8(RgbImage::new(256, 256));

  c.bench_function("preprocess_1024x1024_multi_tile", |b| {
    b.iter(|| {
      let _ = black_box(preproc.preprocess(black_box(&img_1024)));
    });
  });

  c.bench_function("preprocess_256x256_single_tile", |b| {
    b.iter(|| {
      let _ = black_box(preproc.preprocess(black_box(&img_256)));
    });
  });
}

criterion_group!(benches, bench_preproc);
criterion_main!(benches);
