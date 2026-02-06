#!/usr/bin/env python3
"""
Test Chroma MCP Server Connection

Verifies Chroma vector database is running and accessible.

Usage:
    python scripts/test_chroma_connection.py
"""

import sys
import os

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("❌ ERROR: chromadb not installed!")
    print("Run: pip install chromadb==0.5.0")
    sys.exit(1)

# Chroma connection settings
CHROMA_HOST = os.getenv('CHROMA_HOST', 'localhost')
CHROMA_PORT = os.getenv('CHROMA_PORT', '8000')
CHROMA_PERSIST_DIR = os.getenv('CHROMA_PERSIST_DIR', f'{os.path.expanduser("~")}/LeverageBot/mcp_data/chroma_db')

def test_rest_api_connection():
    """Test connection to Chroma REST API (MCP server)"""
    print("Testing Chroma REST API connection...")

    try:
        client = chromadb.Client(Settings(
            chroma_api_impl="rest",
            chroma_server_host=CHROMA_HOST,
            chroma_server_http_port=CHROMA_PORT
        ))

        # Get heartbeat
        heartbeat = client.heartbeat()
        print(f"  ✓ Chroma server heartbeat: {heartbeat}")

        # List collections
        collections = client.list_collections()
        print(f"  ✓ Existing collections: {len(collections)}")
        for coll in collections:
            print(f"    - {coll.name} ({coll.count()} items)")

        return True

    except Exception as e:
        print(f"  ❌ REST API connection failed: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check if Chroma server is running:")
        print(f"     ps aux | grep chroma-mcp")
        print(f"  2. Start Chroma server:")
        print(f"     chroma-mcp start --persist-directory {CHROMA_PERSIST_DIR} --port {CHROMA_PORT} &")
        print(f"  3. Verify server is listening:")
        print(f"     curl http://{CHROMA_HOST}:{CHROMA_PORT}/api/v1/heartbeat")
        return False

def test_persistent_client():
    """Test connection to local persistent Chroma database"""
    print("\nTesting Chroma persistent client...")

    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        # List collections
        collections = client.list_collections()
        print(f"  ✓ Persistent database path: {CHROMA_PERSIST_DIR}")
        print(f"  ✓ Collections: {len(collections)}")

        # Create test collection
        test_collection = client.get_or_create_collection("test_patterns")
        print(f"  ✓ Created test collection 'test_patterns'")

        # Add a test pattern
        test_collection.add(
            embeddings=[[0.1] * 400],  # 400-dim vector (20×20 GAF)
            metadatas=[{"test": True, "timestamp": "2026-02-05T00:00:00Z"}],
            ids=["test_pattern_1"]
        )
        print(f"  ✓ Added test pattern (400-dim vector)")

        # Query
        results = test_collection.query(
            query_embeddings=[[0.1] * 400],
            n_results=1
        )
        print(f"  ✓ Query successful: {len(results['ids'][0])} results")

        # Clean up
        client.delete_collection("test_patterns")
        print(f"  ✓ Deleted test collection")

        return True

    except Exception as e:
        print(f"  ❌ Persistent client connection failed: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check if directory exists and is writable:")
        print(f"     ls -la {CHROMA_PERSIST_DIR}")
        print(f"  2. Create directory if missing:")
        print(f"     mkdir -p {CHROMA_PERSIST_DIR}")
        return False

def main():
    print("=" * 60)
    print("LeverageBot ML Pipeline - Chroma Connection Test")
    print("=" * 60)

    rest_ok = test_rest_api_connection()
    persistent_ok = test_persistent_client()

    print("\n" + "=" * 60)
    if rest_ok and persistent_ok:
        print("✅ Chroma connection test PASSED!")
        print("=" * 60)
        print("\nChroma is ready for ML pattern storage.")
        print("\nNext steps:")
        print("  1. Proceed with Phase 1 training (see ML_IMPLEMENTATION_GUIDE.md)")
        print("  2. Store GAF-encoded patterns in Chroma during Phase 2")
    else:
        print("❌ Chroma connection test FAILED!")
        print("=" * 60)
        print("\nFix the issues above before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()
