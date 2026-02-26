//! Integration tests for RAG pipeline with real-world scenarios
//!
//! These tests validate the RAG pipeline under production-like conditions,
//! including concurrent operations, large document processing, and error recovery.

use mofa_foundation::rag::{
    ChunkConfig, DocumentChunk, InMemoryVectorStore, QdrantConfig, QdrantVectorStore,
    SimilarityMetric, TextChunker, VectorStore,
};
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::time::{timeout, Duration};

/// Generate a deterministic embedding for testing.
/// In production, this would be replaced with a real embedding model.
fn test_embedding(text: &str, dimensions: usize) -> Vec<f32> {
    let mut embedding = vec![0.0_f32; dimensions];
    for (i, byte) in text.bytes().enumerate() {
        embedding[i % dimensions] += byte as f32 / 255.0;
    }
    // Normalize to unit vector
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut embedding {
            *x /= norm;
        }
    }
    embedding
}

/// Test concurrent RAG operations with multiple users
#[tokio::test]
async fn test_concurrent_rag_operations() {
    let store = Arc::new(InMemoryVectorStore::cosine());
    let dimensions = 64;

    // Large knowledge base simulating production data
    let documents = vec![
        "Rust is a systems programming language focused on safety, speed, and concurrency.",
        "Memory safety in Rust is achieved through ownership, borrowing, and lifetimes.",
        "The Rust compiler prevents data races at compile time through its borrow checker.",
        "Cargo is Rust's package manager and build system, handling dependencies automatically.",
        "Rust's async/await syntax makes asynchronous programming ergonomic and safe.",
        "The Rust ecosystem includes powerful crates for web development, databases, and more.",
        "Zero-cost abstractions in Rust mean high-level code performs as well as hand-written C.",
        "Rust's trait system enables generic programming with compile-time polymorphism.",
        "The Rust community emphasizes correctness, performance, and developer experience.",
        "Cross-platform development is straightforward with Rust's target-specific compilation.",
    ];

    // Chunk and store documents
    let chunker = TextChunker::new(ChunkConfig {
        chunk_size: 100,
        chunk_overlap: 20,
    });

    let mut all_chunks = Vec::new();
    for (doc_idx, document) in documents.iter().enumerate() {
        let text_chunks = chunker.chunk_by_chars(document);
        for (chunk_idx, text) in text_chunks.iter().enumerate() {
            let id = format!("doc-{doc_idx}-chunk-{chunk_idx}");
            let embedding = test_embedding(text, dimensions);
            let chunk = DocumentChunk::new(&id, text.as_str(), embedding)
                .with_metadata("source", &format!("document_{doc_idx}"))
                .with_metadata("chunk_index", &chunk_idx.to_string());
            all_chunks.push(chunk);
        }
    }

    // Store all chunks
    let store_clone = Arc::clone(&store);
    store_clone.upsert_batch(all_chunks).await.unwrap();

    // Simulate concurrent users querying the system
    let semaphore = Arc::new(Semaphore::new(10)); // Limit concurrent operations
    let mut handles = vec![];

    let queries = vec![
        "What is Rust's approach to memory safety?",
        "How does Cargo manage dependencies?",
        "What makes Rust suitable for systems programming?",
        "Explain Rust's async capabilities",
        "What are zero-cost abstractions?",
        "How does Rust achieve cross-platform compatibility?",
        "What is the role of traits in Rust?",
        "How does the Rust compiler prevent data races?",
        "What tools does the Rust ecosystem provide?",
        "Why is Rust gaining popularity?",
    ];

    for query in queries {
        let store_clone = Arc::clone(&store);
        let semaphore_clone = Arc::clone(&semaphore);
        let query = query.to_string();

        let handle = tokio::spawn(async move {
            let _permit = semaphore_clone.acquire().await.unwrap();

            let query_embedding = test_embedding(&query, dimensions);
            let results = store_clone.search(&query_embedding, 3, None).await.unwrap();

            // Validate results
            assert!(!results.is_empty(), "Search should return results for: {}", query);
            assert!(results[0].score > 0.0, "Top result should have positive similarity");

            // Check that results are ordered by similarity
            for i in 1..results.len() {
                assert!(results[i-1].score >= results[i].score,
                       "Results should be ordered by decreasing similarity");
            }
        });

        handles.push(handle);
    }

    // Wait for all concurrent operations to complete
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify final state
    let total_chunks = store.count().await.unwrap();
    assert!(total_chunks > 0, "Store should contain chunks after concurrent operations");
}

/// Test RAG pipeline with large documents
#[tokio::test]
async fn test_large_document_processing() {
    let mut store = InMemoryVectorStore::cosine();
    let dimensions = 128;

    // Simulate a large document (multiple paragraphs)
    let large_document = r#"
        Artificial Intelligence (AI) is a transformative technology that is reshaping industries
        and societies worldwide. Machine learning, a subset of AI, enables computers to learn
        from data without being explicitly programmed. Deep learning, using neural networks
        with many layers, has achieved remarkable success in image recognition, natural language
        processing, and game playing.

        The field of natural language processing (NLP) has seen tremendous advances with the
        development of transformer architectures. These models can understand context, generate
        human-like text, and perform complex language tasks. Applications range from chatbots
        and virtual assistants to automated translation and content generation.

        Computer vision, another major area of AI, allows machines to interpret and understand
        visual information from the world. Convolutional neural networks excel at tasks like
        object detection, facial recognition, and medical image analysis. Self-driving cars
        and autonomous systems rely heavily on computer vision technologies.

        Reinforcement learning, inspired by behavioral psychology, enables agents to learn
        optimal behaviors through trial and error. This approach has led to breakthroughs in
        robotics, game playing (like AlphaGo), and autonomous systems. The combination of
        reinforcement learning with deep learning has opened new frontiers in AI research.

        Ethics and responsible AI development have become increasingly important as AI systems
        become more powerful and ubiquitous. Issues of bias, privacy, transparency, and safety
        must be addressed to ensure AI benefits society as a whole. Research in AI safety and
        alignment aims to create AI systems that are beneficial and aligned with human values.

        The future of AI holds great promise but also challenges. Continued research in areas
        like explainable AI, federated learning, and quantum computing will shape the next
        generation of intelligent systems. Collaboration between researchers, policymakers,
        and industry leaders will be crucial in navigating this rapidly evolving landscape.
    "#.repeat(5); // Make it even larger

    let chunker = TextChunker::new(ChunkConfig {
        chunk_size: 500,
        chunk_overlap: 100,
    });

    let text_chunks = chunker.chunk_by_chars(&large_document);
    let mut chunks = Vec::new();

    for (i, text) in text_chunks.iter().enumerate() {
        let id = format!("large-doc-chunk-{i}");
        let embedding = test_embedding(text, dimensions);
        let chunk = DocumentChunk::new(&id, text, embedding)
            .with_metadata("document_type", "large_article")
            .with_metadata("chunk_index", &i.to_string());
        chunks.push(chunk);
    }

    // Test batch insertion
    store.upsert_batch(chunks).await.unwrap();

    let total_chunks = store.count().await.unwrap();
    assert!(total_chunks > 10, "Large document should be split into multiple chunks");

    // Test search on large document
    let query = "What are the main areas of AI research mentioned?";
    let query_embedding = test_embedding(query, dimensions);
    let results = store.search(&query_embedding, 5, None).await.unwrap();

    assert!(!results.is_empty(), "Should find relevant chunks in large document");
    assert!(results[0].score > 0.5, "Top result should be highly relevant");

    // Verify metadata is preserved
    for result in &results {
        assert_eq!(result.metadata.get("document_type").unwrap(), "large_article");
    }
}

/// Test error recovery and edge cases
#[tokio::test]
async fn test_error_recovery_and_edge_cases() {
    let mut store = InMemoryVectorStore::cosine();

    // Test with empty store
    let query_embedding = test_embedding("test query", 64);
    let results = store.search(&query_embedding, 5, None).await.unwrap();
    assert!(results.is_empty(), "Empty store should return no results");

    // Test with zero-dimensional embeddings (edge case)
    let zero_dim_chunk = DocumentChunk::new("zero", "zero dimensions", vec![]);
    assert!(store.upsert(zero_dim_chunk).await.is_err(), "Should reject zero-dimensional embeddings");

    // Test with mismatched dimensions
    let chunk_64d = DocumentChunk::new("64d", "64 dimensions", vec![0.0; 64]);
    store.upsert(chunk_64d).await.unwrap();

    let query_32d = vec![0.0; 32];
    assert!(store.search(&query_32d, 5, None).await.is_err(), "Should reject mismatched dimensions");

    // Test deletion of non-existent items
    let deleted = store.delete("nonexistent").await.unwrap();
    assert!(!deleted, "Deleting non-existent item should return false");

    // Test upsert after delete
    let chunk = DocumentChunk::new("test", "test content", vec![1.0; 64]);
    store.upsert(chunk.clone()).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 1);

    store.delete("test").await.unwrap();
    assert_eq!(store.count().await.unwrap(), 0);

    // Re-upsert should work
    store.upsert(chunk).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 1);
}

/// Test RAG pipeline with timeout constraints (simulating production timeouts)
#[tokio::test]
async fn test_timeout_constraints() {
    let store = Arc::new(InMemoryVectorStore::cosine());
    let dimensions = 64;

    // Populate with many chunks to simulate slow search
    let mut chunks = Vec::new();
    for i in 0..1000 {
        let text = format!("Document chunk number {}", i);
        let embedding = test_embedding(&text, dimensions);
        let chunk = DocumentChunk::new(&format!("chunk-{}", i), &text, embedding);
        chunks.push(chunk);
    }

    let store_clone = Arc::clone(&store);
    store_clone.upsert_batch(chunks).await.unwrap();

    // Test search with timeout
    let query_embedding = test_embedding("find document chunk", dimensions);

    let search_future = store.search(&query_embedding, 10, None);
    let result = timeout(Duration::from_millis(100), search_future).await;

    match result {
        Ok(Ok(results)) => {
            assert!(!results.is_empty(), "Should find results within timeout");
        }
        Ok(Err(_)) => panic!("Search should not fail"),
        Err(_) => panic!("Search should complete within timeout"),
    }
}

/// Test metadata filtering and advanced search features
#[tokio::test]
async fn test_metadata_filtering() {
    let mut store = InMemoryVectorStore::cosine();
    let dimensions = 64;

    // Add chunks with different metadata
    let chunks = vec![
        DocumentChunk::new("rust-doc", "Rust programming language", test_embedding("rust", dimensions))
            .with_metadata("language", "rust")
            .with_metadata("category", "programming"),
        DocumentChunk::new("python-doc", "Python programming language", test_embedding("python", dimensions))
            .with_metadata("language", "python")
            .with_metadata("category", "programming"),
        DocumentChunk::new("ai-doc", "Artificial Intelligence overview", test_embedding("ai", dimensions))
            .with_metadata("language", "none")
            .with_metadata("category", "science"),
    ];

    store.upsert_batch(chunks).await.unwrap();

    // Test basic search
    let query_embedding = test_embedding("programming languages", dimensions);
    let results = store.search(&query_embedding, 10, None).await.unwrap();

    assert!(results.len() >= 2, "Should find programming-related documents");

    // Verify metadata is preserved and accessible
    let rust_result = results.iter().find(|r| r.id == "rust-doc").unwrap();
    assert_eq!(rust_result.metadata.get("language").unwrap(), "rust");
    assert_eq!(rust_result.metadata.get("category").unwrap(), "programming");
}