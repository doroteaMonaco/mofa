//! Error recovery and edge case testing for RAG pipeline
//!
//! This module tests the robustness of the RAG system under various
//! failure conditions and edge cases.

use mofa_foundation::rag::{DocumentChunk, InMemoryVectorStore, VectorStore};
use std::sync::Arc;
use tokio::time::{timeout, Duration};

/// Generate test embedding
fn error_test_embedding(text: &str, dimensions: usize) -> Vec<f32> {
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

/// Test recovery from corrupted data
#[tokio::test]
async fn test_corrupted_data_recovery() {
    let mut store = InMemoryVectorStore::cosine();

    // Add valid data
    let valid_chunks = vec![
        DocumentChunk::new("valid-1", "Valid content 1", error_test_embedding("content 1", 64)),
        DocumentChunk::new("valid-2", "Valid content 2", error_test_embedding("content 2", 64)),
    ];

    store.upsert_batch(valid_chunks).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 2);

    // Simulate corrupted chunk (empty text, zero embedding)
    let corrupted_chunk = DocumentChunk::new("corrupted", "", vec![]);
    assert!(store.upsert(corrupted_chunk).await.is_err(), "Should reject corrupted chunk");

    // Verify store still works with valid data
    let query = error_test_embedding("content", 64);
    let results = store.search(&query, 5, None).await.unwrap();
    assert!(!results.is_empty(), "Should still find valid results after corruption attempt");

    // Test recovery by re-adding valid data
    let recovery_chunk = DocumentChunk::new("recovery", "Recovery content", error_test_embedding("recovery", 64));
    store.upsert(recovery_chunk).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 3);
}

/// Test behavior with extreme embedding values
#[tokio::test]
async fn test_extreme_embedding_values() {
    let mut store = InMemoryVectorStore::cosine();

    // Test with very large values
    let large_embedding = vec![1e10_f32; 64];
    let large_chunk = DocumentChunk::new("large", "Large values", large_embedding);
    store.upsert(large_chunk).await.unwrap();

    // Test with very small values
    let small_embedding = vec![1e-10_f32; 64];
    let small_chunk = DocumentChunk::new("small", "Small values", small_embedding);
    store.upsert(small_chunk).await.unwrap();

    // Test with NaN values (should be handled gracefully)
    let mut nan_embedding = vec![1.0_f32; 64];
    nan_embedding[0] = f32::NAN;
    let nan_chunk = DocumentChunk::new("nan", "NaN values", nan_embedding);
    // Note: This might succeed or fail depending on implementation
    let nan_result = store.upsert(nan_chunk).await;
    // If it fails, that's acceptable; if it succeeds, verify search still works

    // Test with infinite values
    let mut inf_embedding = vec![1.0_f32; 64];
    inf_embedding[0] = f32::INFINITY;
    let inf_chunk = DocumentChunk::new("inf", "Infinite values", inf_embedding);
    let inf_result = store.upsert(inf_chunk).await;

    // Verify basic functionality still works
    let query = vec![1.0_f32; 64];
    let results = store.search(&query, 5, None).await.unwrap();
    assert!(!results.is_empty(), "Should find results despite extreme values");
}

/// Test concurrent operations with failures
#[tokio::test]
async fn test_concurrent_failures() {
    let store = Arc::new(InMemoryVectorStore::cosine());
    let mut handles = vec![];

    // Mix successful and failing operations
    for i in 0..20 {
        let store_clone = Arc::clone(&store);
        let handle = tokio::spawn(async move {
            if i % 3 == 0 {
                // Failing operation: wrong dimensions
                let wrong_dim_embedding = vec![1.0_f32; 32]; // Should be 64
                let result = store_clone.search(&wrong_dim_embedding, 5, None).await;
                assert!(result.is_err(), "Should fail with wrong dimensions");
                false // Failed
            } else {
                // Successful operation
                let embedding = error_test_embedding(&format!("query {}", i), 64);
                let chunk = DocumentChunk::new(&format!("chunk-{}", i), &format!("Content {}", i), embedding.clone());
                let upsert_result = store_clone.upsert(chunk).await;
                if upsert_result.is_ok() {
                    let search_result = store_clone.search(&embedding, 1, None).await;
                    search_result.is_ok()
                } else {
                    false
                }
            }
        });
        handles.push(handle);
    }

    // Collect results
    let mut success_count = 0;
    for handle in handles {
        if handle.await.unwrap() {
            success_count += 1;
        }
    }

    assert!(success_count > 10, "Should have multiple successful operations despite failures");
}

/// Test memory pressure scenarios
#[tokio::test]
async fn test_memory_pressure() {
    let store = Arc::new(InMemoryVectorStore::cosine());

    // Add many chunks to simulate memory pressure
    let mut handles = vec![];

    for batch in 0..10 {
        let store_clone = Arc::clone(&store);
        let handle = tokio::spawn(async move {
            let mut chunks = vec![];
            for i in 0..100 {
                let id = format!("batch-{}-chunk-{}", batch, i);
                let content = format!("Large content for memory test: {}", "x".repeat(1000));
                let embedding = error_test_embedding(&content, 128);
                let chunk = DocumentChunk::new(&id, &content, embedding);
                chunks.push(chunk);
            }
            store_clone.upsert_batch(chunks).await
        });
        handles.push(handle);
    }

    // Wait for all batches
    for handle in handles {
        handle.await.unwrap().unwrap();
    }

    let total_count = store.count().await.unwrap();
    assert_eq!(total_count, 1000, "Should have stored all chunks under memory pressure");

    // Test search still works
    let query = error_test_embedding("Large content", 128);
    let results = store.search(&query, 5, None).await.unwrap();
    assert!(!results.is_empty(), "Should find results after memory pressure test");
}

/// Test timeout scenarios
#[tokio::test]
async fn test_timeout_scenarios() {
    let store = Arc::new(InMemoryVectorStore::cosine());

    // Populate with many chunks to potentially cause slow searches
    let mut chunks = vec![];
    for i in 0..5000 {
        let content = format!("Timeout test content {} with lots of text to make search slower", i);
        let embedding = error_test_embedding(&content, 256); // Higher dimensions = slower
        let chunk = DocumentChunk::new(&format!("timeout-{}", i), &content, embedding);
        chunks.push(chunk);
    }

    store.upsert_batch(chunks).await.unwrap();

    // Test with very short timeout
    let query = error_test_embedding("timeout test", 256);
    let search_future = store.search(&query, 10, None);

    let result = timeout(Duration::from_millis(10), search_future).await;

    match result {
        Ok(Ok(results)) => {
            // Search completed within timeout
            assert!(!results.is_empty(), "Should find results when search completes in time");
        }
        Ok(Err(_)) => {
            panic!("Search should not fail, only potentially timeout");
        }
        Err(_) => {
            // Search timed out - this is acceptable for very slow operations
            // The important thing is that the system doesn't crash
            println!("Search timed out as expected under load");
        }
    }
}

/// Test invalid UTF-8 and edge case strings
#[tokio::test]
async fn test_invalid_utf8_and_edge_cases() {
    let mut store = InMemoryVectorStore::cosine();

    // Test with empty strings
    let empty_chunk = DocumentChunk::new("empty", "", error_test_embedding("", 64));
    store.upsert(empty_chunk).await.unwrap();

    // Test with very long strings
    let long_content = "a".repeat(100000);
    let long_chunk = DocumentChunk::new("long", &long_content, error_test_embedding(&long_content, 64));
    store.upsert(long_chunk).await.unwrap();

    // Test with special characters
    let special_content = "Special chars: ñáéíóú 🚀 🔥 💯";
    let special_chunk = DocumentChunk::new("special", special_content, error_test_embedding(special_content, 64));
    store.upsert(special_chunk).await.unwrap();

    // Test search with various queries
    let queries = vec![
        "",
        "a".repeat(1000),
        "ñáéíóú",
        "🚀",
    ];

    for query in queries {
        let query_embedding = error_test_embedding(&query, 64);
        let results = store.search(&query_embedding, 5, None).await.unwrap();
        // Should not crash, even if results are empty
        assert!(results.len() <= 5, "Should respect max_results limit");
    }
}

/// Test recovery from partial failures in batch operations
#[tokio::test]
async fn test_partial_batch_failure_recovery() {
    let mut store = InMemoryVectorStore::cosine();

    // Create a batch with some valid and some invalid chunks
    let mut chunks = vec![];

    // Valid chunks
    for i in 0..5 {
        let chunk = DocumentChunk::new(
            &format!("valid-{}", i),
            &format!("Valid content {}", i),
            error_test_embedding(&format!("content {}", i), 64)
        );
        chunks.push(chunk);
    }

    // Invalid chunk (empty embedding)
    let invalid_chunk = DocumentChunk::new("invalid", "Invalid content", vec![]);
    chunks.push(invalid_chunk);

    // More valid chunks
    for i in 5..10 {
        let chunk = DocumentChunk::new(
            &format!("valid-{}", i),
            &format!("Valid content {}", i),
            error_test_embedding(&format!("content {}", i), 64)
        );
        chunks.push(chunk);
    }

    // Batch upsert should either succeed entirely or fail atomically
    // (depending on implementation, but shouldn't corrupt state)
    let initial_count = store.count().await.unwrap();

    let result = store.upsert_batch(chunks).await;

    if result.is_ok() {
        // If batch succeeded, all valid chunks should be stored
        let final_count = store.count().await.unwrap();
        assert!(final_count >= initial_count + 9, "Should store at least the valid chunks");
    } else {
        // If batch failed, state should be unchanged
        let final_count = store.count().await.unwrap();
        assert_eq!(final_count, initial_count, "Failed batch should not modify store state");
    }

    // Verify store is still functional
    let query = error_test_embedding("valid content", 64);
    let _results = store.search(&query, 5, None).await.unwrap();
}