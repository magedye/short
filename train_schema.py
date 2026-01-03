"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              Oracle DDL Training Script for Tier-2                           ║
║                                                                              ║
║  One-time setup script to train Vanna agent on database schema.             ║
║                                                                              ║
║  Features:                                                                   ║
║  - Retrieves DDL from Oracle using safe raw connection                      ║
║  - Direct ChromaDB injection (bypasses Vanna API layers)                    ║
║  - Proper metadata tagging for RAG retrieval                                ║
║  - Comprehensive error handling and logging                                 ║
║                                                                              ║
║  Usage:                                                                      ║
║    python train_schema.py                                                   ║
║                                                                              ║
║  Prerequisites:                                                              ║
║  - Backend (main.py) should be running                                      ║
║  - .env file configured with ORACLE_* and CHROMA_*                         ║
║  - VANNA_ALLOW_DDL=true (optional safety flag)                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import logging
import oracledb
import chromadb
from typing import List, Tuple, Optional
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# ==================================================================================
# 1. LOGGING SETUP
# ==================================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"train_schema_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("TrainingScript")

# Load environment from .env if present
load_dotenv()

# ==================================================================================
# 2. ENVIRONMENT VALIDATION
# ==================================================================================

def validate_environment() -> Tuple[bool, str]:
    """Validate that all required environment variables are set."""
    
    logger.info("🔍 Validating environment...")
    
    required_vars = {
        "ORACLE_USER": "Oracle username",
        "ORACLE_PASSWORD": "Oracle password",
        "ORACLE_DSN": "Oracle connection string",
        "CHROMA_PATH": "ChromaDB persistence directory",
        "CHROMA_COLLECTION": "ChromaDB collection name",
    }
    
    missing = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing.append(f"{var} ({description})")
    
    if missing:
        error_msg = f"Missing environment variables:\n" + "\n".join(f"  - {m}" for m in missing)
        logger.error(error_msg)
        return False, error_msg
    
    logger.info("✓ All required environment variables present")
    return True, "OK"

# ==================================================================================
# 3. ORACLE CONNECTION & DDL RETRIEVAL
# ==================================================================================

class OracleSchemaReader:
    """Safe Oracle schema discovery and DDL retrieval."""
    
    def __init__(self):
        self.user = os.getenv("ORACLE_USER")
        self.password = os.getenv("ORACLE_PASSWORD")
        self.dsn = os.getenv("ORACLE_DSN")
        self.connection = None
    
    def connect(self) -> bool:
        """Establish Oracle connection."""
        try:
            logger.info(f"🔌 Connecting to Oracle: {self.dsn}")
            self.connection = oracledb.connect(
                user=self.user,
                password=self.password,
                dsn=self.dsn
            )
            logger.info("✓ Oracle connection established")
            return True
        except Exception as e:
            logger.error(f"✗ Connection failed: {e}")
            return False
    
    def discover_tables(self) -> List[str]:
        """Get list of all user tables."""
        if not self.connection:
            logger.error("No connection available")
            return []
        
        try:
            logger.info("🔍 Discovering tables in schema...")
            cursor = self.connection.cursor()
            cursor.execute("SELECT table_name FROM user_tables ORDER BY table_name")
            tables = [row[0] for row in cursor.fetchall()]
            cursor.close()
            
            logger.info(f"✓ Discovered {len(tables)} tables")
            return tables
        
        except Exception as e:
            logger.error(f"✗ Table discovery failed: {e}")
            return []
    
    def get_table_ddl(self, table_name: str) -> Optional[str]:
        """Retrieve DDL for a specific table."""
        if not self.connection:
            return None
        
        try:
            cursor = self.connection.cursor()
            
            # Use DBMS_METADATA.GET_DDL to retrieve DDL
            cursor.execute(
                f"SELECT DBMS_METADATA.GET_DDL('TABLE', '{table_name}') FROM DUAL"
            )
            row = cursor.fetchone()
            cursor.close()
            
            if row and row[0]:
                # Force LOB read immediately to avoid stale state
                ddl_text = str(row[0])
                logger.debug(f"✓ DDL retrieved for {table_name} ({len(ddl_text)} chars)")
                return ddl_text
            else:
                logger.warning(f"⚠ No DDL found for {table_name}")
                return None
        
        except Exception as e:
            logger.error(f"✗ DDL retrieval failed for {table_name}: {e}")
            return None
    
    def close(self):
        """Close Oracle connection."""
        if self.connection:
            try:
                self.connection.close()
                logger.info("🔌 Oracle connection closed")
            except Exception:
                pass

# ==================================================================================
# 4. CHROMADB INJECTION
# ==================================================================================

class ChromaDBInjector:
    """Direct ChromaDB injection (bypasses Vanna API layers)."""
    
    def __init__(self):
        self.chroma_path = os.getenv("CHROMA_PATH")
        self.collection_name = os.getenv("CHROMA_COLLECTION")
        self.client = None
        self.collection = None
    
    def initialize(self) -> bool:
        """Initialize ChromaDB client and collection."""
        try:
            logger.info(f"📦 Initializing ChromaDB at: {self.chroma_path}")
            
            # Ensure directory exists
            Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
            
            # Create persistent client
            self.client = chromadb.PersistentClient(path=self.chroma_path)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )
            
            logger.info(f"✓ ChromaDB initialized (collection: {self.collection_name})")
            return True
        
        except Exception as e:
            logger.error(f"✗ ChromaDB initialization failed: {e}")
            return False
    
    def inject_ddl(self, table_name: str, ddl_text: str) -> bool:
        """Inject DDL directly into ChromaDB with proper metadata."""
        if not self.collection:
            logger.error("ChromaDB collection not initialized")
            return False
        
        try:
            # Create unique ID for this DDL entry
            doc_id = f"ddl_{table_name}_{datetime.now().timestamp()}"
            
            # Metadata for RAG retrieval
            metadata = {
                "type": "ddl",
                "table": table_name,
                "source": "oracle_training",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Inject into ChromaDB
            self.collection.add(
                documents=[ddl_text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            logger.info(f"✓ DDL injected for {table_name} (ID: {doc_id})")
            return True
        
        except Exception as e:
            logger.error(f"✗ DDL injection failed for {table_name}: {e}")
            return False
    
    def get_collection_count(self) -> int:
        """Get current collection item count."""
        if not self.collection:
            return 0
        try:
            return self.collection.count()
        except Exception:
            return 0

# ==================================================================================
# 5. MAIN TRAINING WORKFLOW
# ==================================================================================

def run_training_workflow() -> bool:
    """
    Main training workflow:
    1. Validate environment
    2. Connect to Oracle
    3. Discover tables
    4. Retrieve DDL for each table
    5. Inject into ChromaDB
    6. Report results
    """
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║              Oracle DDL Training Script — Tier-2                        ║
    ║                                                                          ║
    ║  This will train the Vanna agent on your database schema.              ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # ===== STEP 1: Validate Environment =====
    env_ok, env_msg = validate_environment()
    if not env_ok:
        logger.error(f"Environment validation failed: {env_msg}")
        return False
    
    # ===== STEP 2: Initialize ChromaDB =====
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Initialize ChromaDB")
    logger.info("="*80)
    
    chroma_injector = ChromaDBInjector()
    if not chroma_injector.initialize():
        logger.error("Failed to initialize ChromaDB")
        return False
    
    initial_count = chroma_injector.get_collection_count()
    logger.info(f"ChromaDB collection has {initial_count} items before training")
    
    # ===== STEP 3: Connect to Oracle =====
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Connect to Oracle")
    logger.info("="*80)
    
    oracle_reader = OracleSchemaReader()
    if not oracle_reader.connect():
        logger.error("Failed to connect to Oracle")
        return False
    
    # ===== STEP 4: Discover Tables =====
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Discover Tables")
    logger.info("="*80)
    
    tables = oracle_reader.discover_tables()
    if not tables:
        logger.error("No tables found in schema")
        oracle_reader.close()
        return False
    
    # ===== STEP 5: Train on Each Table =====
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Train on Each Table (DDL Injection)")
    logger.info("="*80)
    
    trained = []
    failed = []
    
    for i, table in enumerate(tables, 1):
        logger.info(f"\n[{i}/{len(tables)}] Processing: {table}")
        
        # Retrieve DDL
        ddl = oracle_reader.get_table_ddl(table)
        if not ddl:
            logger.warning(f"⚠ Skipping {table} (no DDL retrieved)")
            failed.append((table, "No DDL retrieved"))
            continue
        
        # Inject into ChromaDB
        if chroma_injector.inject_ddl(table, ddl):
            trained.append(table)
        else:
            failed.append((table, "ChromaDB injection failed"))
    
    # ===== STEP 6: Cleanup =====
    logger.info("\n" + "="*80)
    logger.info("STEP 5: Cleanup")
    logger.info("="*80)
    
    oracle_reader.close()
    
    # ===== STEP 7: Report Results =====
    logger.info("\n" + "="*80)
    logger.info("TRAINING RESULTS")
    logger.info("="*80)
    
    final_count = chroma_injector.get_collection_count()
    
    print(f"""
    📊 Training Summary
    ───────────────────────────────────────────
    
    Total Tables Discovered:     {len(tables)}
    Successfully Trained:        {len(trained)} ✓
    Failed:                      {len(failed)} ✗
    
    ChromaDB Items (before):     {initial_count}
    ChromaDB Items (after):      {final_count}
    Items Added:                 {final_count - initial_count}
    
    ───────────────────────────────────────────
    """)
    
    if trained:
        print("✓ Successfully Trained Tables:")
        for table in trained:
            print(f"  ✓ {table}")
    
    if failed:
        print("\n✗ Failed Tables:")
        for table, reason in failed:
            print(f"  ✗ {table}: {reason}")
    
    print("\n" + "="*80)
    
    if failed:
        logger.warning(f"⚠ Training completed with {len(failed)} failures")
        print(f"\n⚠️  {len(failed)} table(s) failed to train.")
        print("   Check logs above for details.")
        return False
    else:
        logger.info("✅ Training completed successfully")
        print("\n✅ Training completed successfully!")
        print("   The Vanna agent is now ready to answer questions about your database.")
        print("\n   Next steps:")
        print("   1. Start the backend: python main.py")
        print("   2. Start the frontend: streamlit run ui.py")
        print("   3. Begin asking questions!")
        return True

# ==================================================================================
# 6. ENTRY POINT
# ==================================================================================

if __name__ == "__main__":
    try:
        success = run_training_workflow()
        exit_code = 0 if success else 1
    except KeyboardInterrupt:
        logger.info("\n⏹ Training interrupted by user")
        exit_code = 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        exit_code = 1
    
    sys.exit(exit_code)
