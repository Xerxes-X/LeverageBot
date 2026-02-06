#!/usr/bin/env python3
"""
Import MCP Data for Multi-Computer Sync

Imports:
1. PostgreSQL labeled trades + features (SQL dump)
2. Chroma vector database collections (JSON)

Usage:
    python scripts/import_mcp_data.py
"""

import sys
import os
import json
import subprocess
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
    from sqlalchemy import create_engine, text
except ImportError:
    print("❌ ERROR: Required packages not installed!")
    print("Run: pip install chromadb==0.5.0 sqlalchemy==2.0.36 psycopg2-binary==2.9.10")
    sys.exit(1)

# Configuration
CHROMA_HOST = os.getenv('CHROMA_HOST', 'localhost')
CHROMA_PORT = os.getenv('CHROMA_PORT', '8000')

DB_USER = os.getenv('POSTGRES_USER', 'postgres')
DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
DB_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_NAME = os.getenv('POSTGRES_DB', 'leverage_bot')

EXPORT_DIR = Path.home() / "LeverageBot" / "mcp_data" / "exports"

def import_postgres():
    """Import PostgreSQL data from SQL dump"""
    print("Importing PostgreSQL data...")

    # Find latest export
    postgres_file = EXPORT_DIR / "postgres_latest.sql"

    if not postgres_file.exists():
        print(f"  ⚠ No PostgreSQL export found at {postgres_file}")
        print(f"  Run export_mcp_data.py on the main computer first.")
        return False

    try:
        # Import SQL dump
        cmd = [
            'psql',
            '-U', DB_USER,
            '-h', DB_HOST,
            '-p', DB_PORT,
            '-d', DB_NAME,
            '-f', str(postgres_file)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  ❌ psql import failed: {result.stderr}")
            return False

        file_size = postgres_file.stat().st_size
        print(f"  ✓ Imported PostgreSQL data from {postgres_file.name} ({file_size:,} bytes)")

        # Verify import
        engine = create_engine(f'postgresql://{DB_USER}:@{DB_HOST}:{DB_PORT}/{DB_NAME}')
        with engine.connect() as conn:
            # Count rows in key tables
            for table in ['ohlcv', 'labeled_trades', 'ml_features']:
                try:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table};"))
                    count = result.fetchone()[0]
                    print(f"    ✓ {table}: {count:,} rows")
                except Exception:
                    print(f"    ℹ {table}: not found (may not have been exported)")

        return True

    except Exception as e:
        print(f"  ❌ PostgreSQL import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def import_chroma():
    """Import Chroma collections from JSON"""
    print("\nImporting Chroma vector database...")

    # Find latest export
    chroma_file = EXPORT_DIR / "chroma_latest.json"

    if not chroma_file.exists():
        print(f"  ⚠ No Chroma export found at {chroma_file}")
        print(f"  Run export_mcp_data.py on the main computer first.")
        return False

    try:
        # Load JSON
        with open(chroma_file, 'r') as f:
            export_data = json.load(f)

        print(f"  Loaded export from {export_data['timestamp']}")
        print(f"  Found {len(export_data['collections'])} collections")

        # Connect to Chroma
        client = chromadb.Client(Settings(
            chroma_api_impl="rest",
            chroma_server_host=CHROMA_HOST,
            chroma_server_http_port=CHROMA_PORT
        ))

        # Import each collection
        for coll_data in export_data['collections']:
            coll_name = coll_data['name']
            print(f"  Importing collection '{coll_name}'...")

            # Get or create collection
            collection = client.get_or_create_collection(coll_name)

            # Clear existing data (full replace)
            try:
                client.delete_collection(coll_name)
                collection = client.create_collection(coll_name)
            except:
                pass  # Collection might not exist

            # Add data in batches (max 100 items per batch)
            ids = coll_data['ids']
            embeddings = coll_data['embeddings']
            metadatas = coll_data['metadatas']
            documents = coll_data.get('documents', [None] * len(ids))

            batch_size = 100
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]
                batch_documents = documents[i:i+batch_size]

                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_documents if batch_documents[0] is not None else None
                )

            print(f"    ✓ {len(ids)} items imported")

        file_size = chroma_file.stat().st_size
        print(f"  ✓ Imported Chroma data from {chroma_file.name} ({file_size:,} bytes)")

        return True

    except Exception as e:
        print(f"  ❌ Chroma import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_import():
    """Verify imported data"""
    print("\nVerifying import...")

    try:
        # Check PostgreSQL
        engine = create_engine(f'postgresql://{DB_USER}:@localhost:{DB_PORT}/{DB_NAME}')
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM labeled_trades;"))
            trades_count = result.fetchone()[0]

            result = conn.execute(text("SELECT COUNT(*) FROM ohlcv;"))
            ohlcv_count = result.fetchone()[0]

            print(f"  ✓ PostgreSQL: {trades_count:,} labeled trades, {ohlcv_count:,} OHLCV candles")

        # Check Chroma
        client = chromadb.Client(Settings(
            chroma_api_impl="rest",
            chroma_server_host=CHROMA_HOST,
            chroma_server_http_port=CHROMA_PORT
        ))

        collections = client.list_collections()
        total_items = sum(coll.count() for coll in collections)

        print(f"  ✓ Chroma: {len(collections)} collections, {total_items:,} total items")

        return True

    except Exception as e:
        print(f"  ⚠ Verification failed: {e}")
        return False

def main():
    print("=" * 60)
    print("LeverageBot ML Pipeline - Import MCP Data")
    print("=" * 60)
    print(f"\nImport directory: {EXPORT_DIR}\n")

    # Check if export directory exists
    if not EXPORT_DIR.exists():
        print(f"❌ Export directory not found: {EXPORT_DIR}")
        print("\nEnsure you've pulled the latest changes from Git:")
        print("  git pull origin master")
        sys.exit(1)

    # Check for manifest
    manifest_file = EXPORT_DIR / "sync_manifest.json"
    if manifest_file.exists():
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        print(f"Last export: {manifest.get('last_export', 'Unknown')}\n")

    postgres_ok = import_postgres()
    chroma_ok = import_chroma()

    if postgres_ok or chroma_ok:
        verify_import()

    print("\n" + "=" * 60)
    if postgres_ok and chroma_ok:
        print("✅ MCP data import complete!")
        print("=" * 60)
        print("\nThis computer is now synchronized with the main computer.")
        print("\nNext steps:")
        print("  1. Verify data: python scripts/test_postgres_connection.py")
        print("  2. Verify Chroma: python scripts/test_chroma_connection.py")
        print("  3. Continue ML development (see ML_IMPLEMENTATION_GUIDE.md)")
    elif postgres_ok or chroma_ok:
        print("⚠ Partial import (some data sources failed)")
        print("=" * 60)
        print("\nCheck error messages above and retry.")
    else:
        print("❌ MCP data import failed!")
        print("=" * 60)
        print("\nEnsure export files exist in mcp_data/exports/")
        sys.exit(1)

if __name__ == "__main__":
    main()
