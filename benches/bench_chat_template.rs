//! Bench chat-template rendering — pure CPU, no model files needed.
//!
//! Run: `cargo bench --bench bench_chat_template --features inference`
//!
//! When `inference` is disabled the bench function is a no-op so that
//! `cargo check --benches` still compiles cleanly without that feature.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(feature = "inference")]
use lfm::chat_template::{ContentItem, Message, UserContent, apply_chat_template};

#[cfg(feature = "inference")]
fn bench_chat_template(c: &mut Criterion) {
  // Short user message with one image placeholder — the common
  // production path through the minijinja renderer.
  let messages_image = vec![Message::User {
    content: UserContent::Multimodal(vec![
      ContentItem::Image,
      ContentItem::Text {
        text: "Describe this image.",
      },
    ]),
  }];

  // Text-only user message — exercises the simpler template branch.
  let messages_text = vec![Message::User {
    content: UserContent::Text("What is 2+2?"),
  }];

  // System + user — exercises the system-prompt branch.
  let messages_sys_user = vec![
    Message::System {
      content: "You are a helpful assistant.",
    },
    Message::User {
      content: UserContent::Text("Hello."),
    },
  ];

  c.bench_function("chat_template_image_user", |b| {
    b.iter(|| {
      let _ = black_box(apply_chat_template(black_box(&messages_image), None, true));
    });
  });

  c.bench_function("chat_template_text_only", |b| {
    b.iter(|| {
      let _ = black_box(apply_chat_template(black_box(&messages_text), None, true));
    });
  });

  c.bench_function("chat_template_system_user", |b| {
    b.iter(|| {
      let _ = black_box(apply_chat_template(
        black_box(&messages_sys_user),
        None,
        true,
      ));
    });
  });
}

#[cfg(not(feature = "inference"))]
fn bench_chat_template(_c: &mut Criterion) {
  // No-op when the inference feature is disabled.
  // TODO Task 16: expose apply_chat_template without the inference gate so
  // this bench can run with just --features decoders.
  eprintln!("bench_chat_template: inference feature not enabled; no benchmarks to run.");
}

criterion_group!(benches, bench_chat_template);
criterion_main!(benches);
