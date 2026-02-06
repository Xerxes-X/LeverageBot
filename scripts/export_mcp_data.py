#!/usr/bin/env python3
"""
Export MCP Data for Multi-Computer Sync

Exports:
1. PostgreSQL labeled trades + features (SQL dump)
2. Chroma vector database collections (JSON)

Usage:
    python scripts/export_mcp_data.py
"""

import sys
import os
import json
import subprocess
from datetime import datetime
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
CHROMA_PERSIST_DIR = os.getenv('CHROMA_PERSIST_DIR', f'{os.path.expanduser("~")}/LeverageBot/mcp_data/chroma_db')

DB_USER = os.getenv('POSTGRES_USER', 'postgres')
DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
DB_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_NAME = os.getenv('POSTGRES_DB', 'leverage_bot')

EXPORT_DIR = Path.home() / "LeverageBot" / "mcp_data" / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

def export_postgres():
    """Export PostgreSQL tables to SQL dump"""
    print("Exporting PostgreSQL data...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_file = EXPORT_DIR / f"postgres_export_{timestamp}.sql"

    try:
        # Export specific tables (not entire database)
        tables = ['ohlcv', 'ml_features', 'labeled_trades', 'model_performance']

        cmd = [
            'pg_dump',
            '-U', DB_USER,
            '-h', DB_HOST,
            '-p', DB_PORT,
            '-d', DB_NAME,
            '--data-only',  # Data only, no schema
            '--no-owner',
            '--no-privileges',
        ]

        # Add table-specific flags
        for table in tables:
            cmd.extend(['-t', table])

        # Add output file
        cmd.extend(['-f', str(export_file)])

        # Run pg_dump
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  ❌ pg_dump failed: {result.stderr}")
            return None

        file_size = export_file.stat().st_size
        print(f"  ✓ Exported PostgreSQL data: {export_file.name} ({file_size:,} bytes)")

        # Create symlink to latest
        latest_link = EXPORT_DIR / "postgres_latest.sql"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(export_file.name)

        return export_file

    except Exception as e:
        print(f"  ❌ PostgreSQL export failed: {e}")
        return None

def export_chroma():
    """Export Chroma collections to JSON"""
    print("\nExporting Chroma vector database...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_file = EXPORT_DIR / f"chroma_export_{timestamp}.json"

    try:
        # Connect to Chroma
        client = chromadb.Client(Settings(
            chroma_api_impl="rest",
            chroma_server_host=CHROMA_HOST,
            chroma_server_http_port=CHROMA_PORT
        ))

        # Get all collections
        collections = client.list_collections()
        print(f"  Found {len(collections)} collections")

        export_data = {
            "timestamp": timestamp,
            "collections": []
        }

        for coll in collections:
            print(f"  Exporting collection '{coll.name}'...")

            # Get all data from collection
            data = coll.get(include=['embeddings', 'metadatas', 'documents'])

            export_data["collections"].append({
                "name": coll.name,
                "count": len(data['ids']),
                "ids": data['ids'],
                "embeddings": data['embeddings'],
                "metadatas": data['metadatas'],
                "documents": data.get('documents', [])
            })

            print(f"    ✓ {len(data['ids'])} items")

        # Write JSON
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        file_size = export_file.stat().st_size
        print(f"  ✓ Exported Chroma data: {export_file.name} ({file_size:,} bytes)")

        # Create symlink to latest
        latest_link = EXPORT_DIR / "chroma_latest.json"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(export_file.name)

        return export_file

    except Exception as e:
        print(f"  ❌ Chroma export failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_sync_manifest():
    """Create manifest file with export metadata"""
    manifest_file = EXPORT_DIR / "sync_manifest.json"

    manifest = {
        "last_export": datetime.now().isoformat(),
        "postgres_latest": (EXPORT_DIR / "postgres_latest.sql").resolve().name if (EXPORT_DIR / "postgres_latest.sql").exists() else None,
        "chroma_latest": (EXPORT_DIR / "chroma_latest.json").resolve().name if (EXPORT_DIR / "chroma_latest.json").exists() else None,
    }

    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  ✓ Updated sync manifest: {manifest_file.name}")

def main():
    print("=" * 60)
    print("LeverageBot ML Pipeline - Export MCP Data")
    print("=" * 60)
    print(f"\nExport directory: {EXPORT_DIR}\n")

    postgres_file = export_postgres()
    chroma_file = export_chroma()

    if postgres_file or chroma_file:
        create_sync_manifest()

    print("\n" + "=" * 60)
    if postgres_file and chroma_file:
        print("✅ MCP data export complete!")
        print("=" * 60)
        print("\nExported files:")
        if postgres_file:
            print(f"  - {postgres_file}")
        if chroma_file:
            print(f"  - {chroma_file}")
        print("\nNext steps:")
        print("  1. Commit to Git:")
        print(f"     git add {EXPORT_DIR.relative_to(Path.home() / 'LeverageBot')}/*")
        print(f"     git commit -m 'MCP data export - {datetime.now().strftime('%Y-%m-%d %H:%M')}'")
        print(f"     git push origin master")
        print("  2. On second computer:")
        print("     git pull origin master")
        print("     python scripts/import_mcp_data.py")
    else:
        print("❌ MCP data export failed!")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
