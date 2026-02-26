//! Load testing for RAG pipeline
//!
//! This module provides tools for testing the RAG system under high load
//! and measuring performance characteristics.

use mofa_foundation::rag::{DocumentChunk, InMemoryVectorStore, VectorStore};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tokio::task;

/// Generate test embedding
fn load_test_embedding(text: &str, dimensions: usize) -> Vec<f32> {
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

/// Load test configuration
#[derive(Debug, Clone)]
pub struct LoadTestConfig {
    pub num_documents: usize,
    pub document_size: usize, // Approximate characters per document
    pub num_concurrent_users: usize,
    pub num_queries_per_user: usize,
    pub dimensions: usize,
    pub max_results: usize,
}

impl Default for LoadTestConfig {
    fn default() -> Self {
        Self {
            num_documents: 1000,
            document_size: 1000,
            num_concurrent_users: 50,
            num_queries_per_user: 10,
            dimensions: 128,
            max_results: 5,
        }
    }
}

/// Load test results
#[derive(Debug)]
pub struct LoadTestResults {
    pub total_queries: usize,
    pub successful_queries: usize,
    pub failed_queries: usize,
    pub total_duration: Duration,
    pub avg_query_time: Duration,
    pub qps: f64, // Queries per second
    pub p95_query_time: Duration,
    pub p99_query_time: Duration,
}

/// Run a comprehensive load test
pub async fn run_load_test(config: LoadTestConfig) -> Result<LoadTestResults, Box<dyn std::error::Error>> {
    println!("Starting RAG load test with config: {:?}", config);

    // Setup: Create and populate vector store
    let store = Arc::new(InMemoryVectorStore::cosine());
    let setup_start = Instant::now();

    let mut documents = Vec::new();
    for i in 0..config.num_documents {
        let content = format!("Document {}: {}", i, "This is test content. ".repeat(config.document_size / 20));
        let embedding = load_test_embedding(&content, config.dimensions);
        let chunk = DocumentChunk::new(&format!("doc-{}", i), &content, embedding)
            .with_metadata("index", &i.to_string());
        documents.push(chunk);
    }

    store.upsert_batch(documents).await?;
    let setup_duration = setup_start.elapsed();
    println!("Setup completed in {:?}", setup_duration);

    // Load test: Execute concurrent queries
    let test_start = Instant::now();
    let semaphore = Arc::new(Semaphore::new(config.num_concurrent_users));
    let mut handles = vec![];

    for user_id in 0..config.num_concurrent_users {
        let store_clone = Arc::clone(&store);
        let semaphore_clone = Arc::clone(&semaphore);
        let config_clone = config.clone();

        let handle = task::spawn(async move {
            let _permit = semaphore_clone.acquire().await.unwrap();
            let mut user_results = Vec::new();

            for query_id in 0..config_clone.num_queries_per_user {
                let query_start = Instant::now();
                let query_text = format!("user {} query {}", user_id, query_id);
                let query_embedding = load_test_embedding(&query_text, config_clone.dimensions);

                let result = store_clone.search(&query_embedding, config_clone.max_results, None).await;
                let query_duration = query_start.elapsed();

                match result {
                    Ok(results) => {
                        user_results.push((true, query_duration, results.len()));
                    }
                    Err(_) => {
                        user_results.push((false, query_duration, 0));
                    }
                }
            }

            user_results
        });

        handles.push(handle);
    }

    // Collect results
    let mut all_results = Vec::new();
    for handle in handles {
        let user_results = handle.await?;
        all_results.extend(user_results);
    }

    let test_duration = test_start.elapsed();

    // Analyze results
    let total_queries = all_results.len();
    let successful_queries = all_results.iter().filter(|(success, _, _)| *success).count();
    let failed_queries = total_queries - successful_queries;

    let query_times: Vec<Duration> = all_results.iter().map(|(_, duration, _)| *duration).collect();
    let avg_query_time = query_times.iter().sum::<Duration>() / total_queries as u32;

    let mut sorted_times = query_times.clone();
    sorted_times.sort();

    let p95_index = (total_queries as f64 * 0.95) as usize;
    let p99_index = (total_queries as f64 * 0.99) as usize;

    let p95_query_time = sorted_times[p95_index];
    let p99_query_time = sorted_times[p99_index];

    let qps = total_queries as f64 / test_duration.as_secs_f64();

    let results = LoadTestResults {
        total_queries,
        successful_queries,
        failed_queries,
        total_duration: test_duration,
        avg_query_time,
        qps,
        p95_query_time,
        p99_query_time,
    };

    println!("Load test completed:");
    println!("  Total queries: {}", results.total_queries);
    println!("  Successful: {} ({:.1}%)", results.successful_queries,
             results.successful_queries as f64 / results.total_queries as f64 * 100.0);
    println!("  Failed: {}", results.failed_queries);
    println!("  Total duration: {:?}", results.total_duration);
    println!("  Average query time: {:?}", results.avg_query_time);
    println!("  QPS: {:.1}", results.qps);
    println!("  P95 query time: {:?}", results.p95_query_time);
    println!("  P99 query time: {:?}", results.p99_query_time);

    Ok(results)
}

/// Stress test with increasing load
pub async fn stress_test() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running RAG stress test with increasing load...");

    let base_config = LoadTestConfig {
        num_documents: 500,
        document_size: 500,
        dimensions: 64,
        max_results: 3,
        ..Default::default()
    };

    let concurrency_levels = vec![10, 25, 50, 100];

    for concurrency in concurrency_levels {
        println!("\n--- Testing with {} concurrent users ---", concurrency);
        let config = LoadTestConfig {
            num_concurrent_users: concurrency,
            ..base_config.clone()
        };

        let results = run_load_test(config).await?;

        // Basic assertions for stress test
        assert!(results.successful_queries > 0, "Should have successful queries");
        assert!(results.qps > 0.0, "Should have positive QPS");

        // Performance thresholds (adjust based on system capabilities)
        if results.avg_query_time > Duration::from_millis(100) {
            println!("WARNING: Average query time exceeds 100ms threshold");
        }
    }

    println!("\nStress test completed successfully!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_load_test() {
        let config = LoadTestConfig {
            num_documents: 100,
            document_size: 100,
            num_concurrent_users: 5,
            num_queries_per_user: 2,
            dimensions: 32,
            max_results: 3,
        };

        let results = run_load_test(config).await.unwrap();

        assert_eq!(results.total_queries, 10); // 5 users * 2 queries
        assert!(results.successful_queries > 0);
        assert!(results.qps > 0.0);
    }

    #[tokio::test]
    async fn test_stress_test() {
        stress_test().await.unwrap();
    }
}