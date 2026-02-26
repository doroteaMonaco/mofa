//! Performance benchmarks for RAG pipeline
//!
//! These benchmarks measure the performance characteristics of the RAG system
//! under various loads and configurations.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mofa_foundation::rag::{
    ChunkConfig, DocumentChunk, InMemoryVectorStore, SimilarityMetric, TextChunker, VectorStore,
};
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Generate test embedding
fn bench_embedding(text: &str, dimensions: usize) -> Vec<f32> {
    let mut embedding = vec![0.0_f32; dimensions];
    for (i, byte) in text.bytes().enumerate() {
        embedding[i % dimensions] += byte as f32 / 255.0;
    }
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut embedding {
            *x /= norm;
        }
    }
    embedding
}

/// Benchmark document chunking performance
fn bench_chunking(c: &mut Criterion) {
    let large_text = include_str!("../../README.md").repeat(10);
    let chunker = TextChunker::new(ChunkConfig {
        chunk_size: 500,
        chunk_overlap: 50,
    });

    c.bench_function("chunk_by_chars_5kb", |b| {
        b.iter(|| {
            let chunks = chunker.chunk_by_chars(black_box(&large_text));
            black_box(chunks);
        })
    });
}

/// Benchmark embedding generation (simulated)
fn bench_embedding_generation(c: &mut Criterion) {
    let texts = vec![
        "Short text",
        "This is a medium length text for testing embedding performance",
        include_str!("../../README.md"),
    ];

    c.bench_function("generate_embedding_64d", |b| {
        b.iter(|| {
            for text in &texts {
                let embedding = bench_embedding(black_box(text), 64);
                black_box(embedding);
            }
        })
    });
}

/// Benchmark vector store operations
fn bench_vector_store_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    // Setup: Create store with test data
    let store = Arc::new(InMemoryVectorStore::cosine());
    let chunks = rt.block_on(async {
        let mut chunks = Vec::new();
        for i in 0..100 {
            let text = format!("Test document number {} with some content", i);
            let embedding = bench_embedding(&text, 64);
            let chunk = DocumentChunk::new(&format!("doc-{}", i), &text, embedding);
            chunks.push(chunk);
        }
        chunks
    });

    rt.block_on(async {
        store.upsert_batch(chunks).await.unwrap();
    });

    c.bench_function("vector_store_search_100_docs", |b| {
        b.iter(|| {
            rt.block_on(async {
                let query = bench_embedding("test query", 64);
                let results = store.search(black_box(&query), 5, None).await.unwrap();
                black_box(results);
            });
        })
    });

    c.bench_function("vector_store_upsert_single", |b| {
        b.iter(|| {
            rt.block_on(async {
                let text = "New document content";
                let embedding = bench_embedding(text, 64);
                let chunk = DocumentChunk::new("new-doc", text, embedding);
                store.upsert(black_box(chunk)).await.unwrap();
            });
        })
    });
}

/// Benchmark concurrent operations
fn bench_concurrent_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    // Setup larger dataset
    let store = Arc::new(InMemoryVectorStore::cosine());
    let chunks = rt.block_on(async {
        let mut chunks = Vec::new();
        for i in 0..500 {
            let text = format!("Document {}: This is a test document with some content for benchmarking purposes. It contains multiple sentences and should provide a realistic test case for the RAG system performance evaluation.", i);
            let embedding = bench_embedding(&text, 128);
            let chunk = DocumentChunk::new(&format!("doc-{}", i), &text, embedding);
            chunks.push(chunk);
        }
        chunks
    });

    rt.block_on(async {
        store.upsert_batch(chunks).await.unwrap();
    });

    c.bench_function("concurrent_search_10_queries", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut handles = vec![];
                for i in 0..10 {
                    let store_clone = Arc::clone(&store);
                    let handle = tokio::spawn(async move {
                        let query_text = format!("query number {}", i);
                        let query = bench_embedding(&query_text, 128);
                        let results = store_clone.search(&query, 3, None).await.unwrap();
                        black_box(results);
                    });
                    handles.push(handle);
                }

                for handle in handles {
                    handle.await.unwrap();
                }
            });
        })
    });
}

/// Benchmark different similarity metrics
fn bench_similarity_metrics(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let metrics = vec![
        ("cosine", SimilarityMetric::Cosine),
        ("euclidean", SimilarityMetric::Euclidean),
        ("dot_product", SimilarityMetric::DotProduct),
    ];

    for (name, metric) in metrics {
        let store = Arc::new(InMemoryVectorStore::new(metric));
        let chunks = rt.block_on(async {
            let mut chunks = Vec::new();
            for i in 0..200 {
                let text = format!("Test document {}", i);
                let embedding = bench_embedding(&text, 64);
                let chunk = DocumentChunk::new(&format!("doc-{}", i), &text, embedding);
                chunks.push(chunk);
            }
            chunks
        });

        rt.block_on(async {
            store.upsert_batch(chunks).await.unwrap();
        });

        c.bench_function(&format!("search_{}_200_docs", name), |b| {
            b.iter(|| {
                rt.block_on(async {
                    let query = bench_embedding("test query", 64);
                    let results = store.search(&query, 5, None).await.unwrap();
                    black_box(results);
                });
            });
        });
    }
}

criterion_group!(
    benches,
    bench_chunking,
    bench_embedding_generation,
    bench_vector_store_operations,
    bench_concurrent_operations,
    bench_similarity_metrics
);
criterion_main!(benches);