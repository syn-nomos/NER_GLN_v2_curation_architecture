import os
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import streamlit as st

import time

class DBManager:
    def __init__(self, db_url=None):
        # Try to get URL from argument, then env var, then Streamlit secrets
        self.db_url = db_url or os.getenv("DATABASE_URL")
        
        # Check Streamlit secrets if not found in env
        if not self.db_url:
            try:
                if hasattr(st, "secrets") and "DATABASE_URL" in st.secrets:
                    self.db_url = st.secrets["DATABASE_URL"]
            except FileNotFoundError:
                pass # Secrets file might not exist locally
            
        if not self.db_url:
            raise ValueError("DATABASE_URL is not set in environment variables or Streamlit secrets.")

        # --- URL Cleaning (Fix common copy-paste errors) ---
        # Fix: User reported "db@" typo in host (e.g. ...@db@aws-1...)
        if "@db@" in self.db_url:
            print("🧹 Fixing malformed URL: Removing extra 'db@'...")
            self.db_url = self.db_url.replace("@db@", "@")

        # Ensure SSL mode is set for Supabase (Required for transaction poolers)
        if "sslmode" not in self.db_url:
            separator = "&" if "?" in self.db_url else "?"
            self.db_url += f"{separator}sslmode=require"

        # Debug: Print connection target (masked)
        try:
            if "@" in self.db_url:
                part = self.db_url.split("@")[1]
                host_port = part.split("/")[0]
                print(f"🔌 Attempting to connect to: {host_port}")
                if ":5432" in host_port and "supabase" in host_port:
                    print("⚠️ WARNING: Connecting to Port 5432 (Direct). This may fail on Streamlit Cloud. Use Port 6543 (Pooler).")
        except:
            pass

        # Retry logic for connection
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Connect to the database
                # Removed keepalives as they might cause issues with some poolers
                self.conn = psycopg2.connect(
                    self.db_url, 
                    cursor_factory=RealDictCursor,
                    connect_timeout=10
                )
                self.ensure_schema_compatibility()
                print("✅ Database Connected Successfully")
                break # Success!
            except Exception as e:
                print(f"⚠️ Database Connection Attempt {attempt+1} Failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2) # Wait 2 seconds before retrying
                else:
                    print(f"❌ All connection attempts failed.")
                    raise e

    def check_connection(self):
        """Checks if the connection is alive."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT 1")
            return True
        except Exception:
            return False

    def ensure_schema_compatibility(self):
        """Checks for missing columns (like is_rejected, manual_edit_type) and adds them if needed."""
        try:
            cursor = self.conn.cursor()
            
            # List of columns to check and their definitions
            columns_to_check = {
                'is_rejected': 'BOOLEAN DEFAULT FALSE',
                'manual_edit_type': 'TEXT'
            }

            for col_name, col_def in columns_to_check.items():
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='annotations' AND column_name=%s;
                """, (col_name,))
                
                if not cursor.fetchone():
                    print(f"⚠️ Column '{col_name}' missing in Postgres. Adding it...")
                    cursor.execute(f"ALTER TABLE annotations ADD COLUMN {col_name} {col_def};")
                    self.conn.commit()
                    print(f"✅ Column '{col_name}' added.")
            
            # Create Indices for Performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ann_sentence_id ON annotations(sentence_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ann_is_accepted ON annotations(is_accepted);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ann_is_rejected ON annotations(is_rejected);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ann_label ON annotations(label);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ann_confidence ON annotations(confidence);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ann_sid_accepted ON annotations(sentence_id, is_accepted);")
            self.conn.commit()
                    
        except Exception as e:
            print(f"⚠️ Schema check failed: {e}")
            self.conn.rollback()
        
    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS entity_centroids (
            id SERIAL PRIMARY KEY,
            text_span TEXT NOT NULL,
            label TEXT NOT NULL,
            vector BYTEA,
            count INTEGER DEFAULT 1,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(text_span, label)
        );
        CREATE TABLE IF NOT EXISTS memory_centroids (
            source_annotation_id INTEGER PRIMARY KEY,
            text_span TEXT,
            label TEXT,
            vector BYTEA,
            count INTEGER DEFAULT 1,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        self.conn.commit()

    def get_filtered_sentences(self, filters, limit=50, offset=0):
        cursor = self.conn.cursor()
        params = []
        
        # 1. Build Annotation Filter Clause
        ann_clauses = []
        ann_params = []
        
        if filters.get('source_agent'):
            agents = filters['source_agent']
            if agents:
                sub = []
                for agent in agents:
                    sub.append("source_agent LIKE %s")
                    ann_params.append(f"%{agent}%")
                ann_clauses.append("(" + " OR ".join(sub) + ")")

        if filters.get('label'):
            labels = filters['label']
            if labels:
                placeholders = ','.join(['%s'] * len(labels))
                ann_clauses.append(f"label IN ({placeholders})")
                ann_params.extend(labels)
            
        if filters.get('confidence_min') is not None:
            min_c = filters['confidence_min']
            if min_c > 0.01:
                ann_clauses.append("confidence >= %s")
                ann_params.append(min_c)
            
        if filters.get('confidence_max') is not None:
            max_c = filters['confidence_max']
            if max_c < 0.99:
                ann_clauses.append("confidence <= %s")
                ann_params.append(max_c)

        # 2. Build Main Query
        query = "SELECT id, text, status, is_flagged, comments FROM sentences s WHERE 1=1"
        
        if filters.get('dataset_split'):
            query += " AND dataset_split = %s"
            params.append(filters['dataset_split'])

        if filters.get('status') == 'pending':
            query += " AND EXISTS (SELECT 1 FROM annotations a_check WHERE a_check.sentence_id = s.id AND a_check.is_accepted = FALSE AND (a_check.is_rejected = FALSE OR a_check.is_rejected IS NULL))"
        elif filters.get('status') == 'completed':
            query += " AND NOT EXISTS (SELECT 1 FROM annotations a_check WHERE a_check.sentence_id = s.id AND a_check.is_accepted = FALSE AND (a_check.is_rejected = FALSE OR a_check.is_rejected IS NULL))"
            
        if filters.get('flagged_only'):
            query += " AND is_flagged IS TRUE"

        if ann_clauses:
            query += f" AND EXISTS (SELECT 1 FROM annotations a WHERE a.sentence_id = s.id AND {' AND '.join(ann_clauses)})"
            params.extend(ann_params)

        # Annotation Count Filter
        if filters.get('annotation_count_type'):
            count_type = filters['annotation_count_type']
            min_a = filters.get('min_annotations', 0)
            max_a = filters.get('max_annotations', 9999)
            
            count_cond = ""
            if count_type == "Pending":
                count_cond = "AND is_accepted = FALSE AND (is_rejected = FALSE OR is_rejected IS NULL)"
            elif count_type == "Accepted":
                count_cond = "AND is_accepted = TRUE"
            elif count_type == "Rejected":
                count_cond = "AND is_rejected = TRUE"
            else: # Total (Active)
                count_cond = "AND (is_rejected = FALSE OR is_rejected IS NULL)"
                
            query += f" AND (SELECT COUNT(*) FROM annotations a_cnt WHERE a_cnt.sentence_id = s.id {count_cond}) BETWEEN %s AND %s"
            params.extend([min_a, max_a])

        query += " ORDER BY id LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        cursor.execute(query, tuple(params))
        return [dict(row) for row in cursor.fetchall()]

    def get_total_filtered_count(self, filters):
        cursor = self.conn.cursor()
        params = []
        
        # 1. Build Annotation Filter Clause
        ann_clauses = []
        ann_params = []
        
        if filters.get('source_agent'):
            agents = filters['source_agent']
            if agents:
                sub = []
                for agent in agents:
                    sub.append("source_agent LIKE %s")
                    ann_params.append(f"%{agent}%")
                ann_clauses.append("(" + " OR ".join(sub) + ")")

        if filters.get('label'):
            labels = filters['label']
            if labels:
                placeholders = ','.join(['%s'] * len(labels))
                ann_clauses.append(f"label IN ({placeholders})")
                ann_params.extend(labels)
            
        if filters.get('confidence_min') is not None:
            min_c = filters['confidence_min']
            if min_c > 0.01:
                ann_clauses.append("confidence >= %s")
                ann_params.append(min_c)
            
        if filters.get('confidence_max') is not None:
            max_c = filters['confidence_max']
            if max_c < 0.99:
                ann_clauses.append("confidence <= %s")
                ann_params.append(max_c)

        # 2. Build Main Query
        query = "SELECT COUNT(*) as count FROM sentences s WHERE 1=1"
        
        if filters.get('dataset_split'):
            query += " AND dataset_split = %s"
            params.append(filters['dataset_split'])
        
        if filters.get('status') == 'pending':
            query += " AND EXISTS (SELECT 1 FROM annotations a_check WHERE a_check.sentence_id = s.id AND a_check.is_accepted = FALSE AND (a_check.is_rejected = FALSE OR a_check.is_rejected IS NULL))"
        elif filters.get('status') == 'completed':
            query += " AND NOT EXISTS (SELECT 1 FROM annotations a_check WHERE a_check.sentence_id = s.id AND a_check.is_accepted = FALSE AND (a_check.is_rejected = FALSE OR a_check.is_rejected IS NULL))"
            
        if filters.get('flagged_only'):
            query += " AND is_flagged IS TRUE"

        if ann_clauses:
            query += f" AND EXISTS (SELECT 1 FROM annotations a WHERE a.sentence_id = s.id AND {' AND '.join(ann_clauses)})"
            params.extend(ann_params)
            
        # Annotation Count Filter
        if filters.get('annotation_count_type'):
            count_type = filters['annotation_count_type']
            min_a = filters.get('min_annotations', 0)
            max_a = filters.get('max_annotations', 9999)
            
            count_cond = ""
            if count_type == "Pending":
                count_cond = "AND is_accepted = FALSE AND (is_rejected = FALSE OR is_rejected IS NULL)"
            elif count_type == "Accepted":
                count_cond = "AND is_accepted = TRUE"
            elif count_type == "Rejected":
                count_cond = "AND is_rejected = TRUE"
            else: # Total (Active)
                count_cond = "AND (is_rejected = FALSE OR is_rejected IS NULL)"
                
            query += f" AND (SELECT COUNT(*) FROM annotations a_cnt WHERE a_cnt.sentence_id = s.id {count_cond}) BETWEEN %s AND %s"
            params.extend([min_a, max_a])
            
        cursor.execute(query, tuple(params))
        result = cursor.fetchone()
        return result['count'] if result else 0

    def get_annotations_for_sentence(self, sentence_id):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM annotations 
            WHERE sentence_id = %s 
            ORDER BY start_char
        """, (sentence_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_unique_values(self, column):
        allowed_columns = ['source_agent', 'label', 'status', 'manual_edit_type']
        if column not in allowed_columns:
            return []
            
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT DISTINCT {column} FROM annotations WHERE {column} IS NOT NULL")
        return [row[column] for row in cursor.fetchall()]

    def get_confidence_range(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT MIN(confidence) as min_c, MAX(confidence) as max_c FROM annotations")
        row = cursor.fetchone()
        if row and row['min_c'] is not None and row['max_c'] is not None:
            return row['min_c'], row['max_c']
        return 0.0, 1.0

    def get_status_counts(self, dataset_split=None):
        """Returns a dictionary of status counts based on dynamic annotation state."""
        cursor = self.conn.cursor()
        
        params = []
        split_clause = ""
        if dataset_split:
            # We need to qualify the column if joining, but dataset_split is on 'sentences'
            split_clause = " AND s.dataset_split = %s"
            params.append(dataset_split)

        # Count Pending: Sentences with at least one pending annotation
        # Use simple string concatenation for split_clause since it is safe (params used for value)
        query_pending = f"""
            SELECT COUNT(DISTINCT s.id) as count
            FROM sentences s
            JOIN annotations a ON s.id = a.sentence_id
            WHERE a.is_accepted = FALSE 
            AND (a.is_rejected = FALSE OR a.is_rejected IS NULL)
            {split_clause}
        """
        cursor.execute(query_pending, tuple(params))
        pending_count = cursor.fetchone()['count']
        
        # Count Completed: Sentences with NO pending annotations
        # (Total Sentences in Split - Pending Sentences in Split)
        query_total = f"SELECT COUNT(*) as count FROM sentences s WHERE 1=1 {split_clause}"
        cursor.execute(query_total, tuple(params))
        total_count = cursor.fetchone()['count']
        
        completed_count = total_count - pending_count
        
        return {
            'pending': pending_count,
            'completed': completed_count
        }

    def insert_or_update_annotation(self, sentence_id, text, label, start, end, vector, is_gold=True):
        cursor = self.conn.cursor()
        vec_blob = vector.tobytes() if vector is not None else None
        
        # Check if exists
        cursor.execute("""
            SELECT id, frequency FROM annotations 
            WHERE text_span = %s AND label = %s AND is_golden = TRUE
            LIMIT 1
        """, (text, label))
        
        row = cursor.fetchone()
        
        if row:
            # UPDATE
            new_freq = row['frequency'] + 1
            cursor.execute("""
                UPDATE annotations 
                SET frequency = %s, sentence_id = %s 
                WHERE id = %s
            """, (new_freq, sentence_id, row['id']))
        else:
            # INSERT
            cursor.execute("""
                INSERT INTO annotations (sentence_id, text_span, label, start_char, end_char, vector, frequency, source_agent, is_golden)
                VALUES (%s, %s, %s, %s, %s, %s, 1, 'gold_dataset', %s)
            """, (sentence_id, text, label, start, end, vec_blob, is_gold))
            
        self.conn.commit()

    def save_prototype(self, label, vector, count):
        cursor = self.conn.cursor()
        vec_blob = vector.tobytes()
        # Postgres UPSERT
        cursor.execute("""
            INSERT INTO prototypes (label, vector, count, last_updated)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (label) 
            DO UPDATE SET 
                vector = EXCLUDED.vector,
                count = EXCLUDED.count,
                last_updated = CURRENT_TIMESTAMP
        """, (label, vec_blob, count))
        self.conn.commit()

    def find_similar_sentences(self, text_span, limit=20):
        """
        Finds pending sentences containing the text_span (Smart Fuzzy Match).
        1. Tries exact substring match.
        2. If few results, tries 'Word Intersection' (all significant words must appear).
        """
        cursor = self.conn.cursor()
        
        # 1. Exact Substring (Normalized)
        search_term = f"%{text_span.lower()}%"
        query = """
            SELECT id, text 
            FROM sentences 
            WHERE status = 'pending' 
            AND lower(text) LIKE %s
            LIMIT %s
        """
        cursor.execute(query, (search_term, limit))
        results = [dict(row) for row in cursor.fetchall()]
        
        # 2. Word Intersection (if results < limit)
        if len(results) < limit:
            # Split into words, keep only significant ones (>2 chars)
            words = [w for w in text_span.lower().split() if len(w) > 2]
            
            if len(words) > 1: # Only if it's a multi-word phrase
                # Construct query: text LIKE %w1% AND text LIKE %w2% ...
                conditions = []
                params = []
                for w in words:
                    conditions.append("lower(text) LIKE %s")
                    params.append(f"%{w}%")
                
                # Exclude already found IDs
                found_ids = [r['id'] for r in results]
                if found_ids:
                    placeholders = ','.join(['%s'] * len(found_ids))
                    exclude_clause = f"AND id NOT IN ({placeholders})"
                    params.extend(found_ids)
                else:
                    exclude_clause = ""
                
                params.append(limit - len(results))
                
                query_fuzzy = f"""
                    SELECT id, text 
                    FROM sentences 
                    WHERE status = 'pending' 
                    AND {' AND '.join(conditions)}
                    {exclude_clause}
                    LIMIT %s
                """
                cursor.execute(query_fuzzy, tuple(params))
                results.extend([dict(row) for row in cursor.fetchall()])
                
        return results

    def get_similar_pending_annotations(self, target_vector_blob, threshold=0.9, limit=None, label_filter=None, filter_mode="All"):
        """
        Finds pending annotations with high cosine similarity to the target vector.
        Supports label filtering (Same/Diff/All).
        """
        if target_vector_blob is None:
            return []
            
        target_vec = np.frombuffer(target_vector_blob, dtype=np.float32)
        norm_target = np.linalg.norm(target_vec)
        if norm_target == 0: return []

        cursor = self.conn.cursor()
        
        # Base Query
        query = """
            SELECT a.id, a.text_span, a.label, a.confidence, a.vector, a.sentence_id, a.start_char, a.end_char, s.text as sentence_text
            FROM annotations a
            JOIN sentences s ON a.sentence_id = s.id
            WHERE a.is_accepted = FALSE 
            AND (a.is_rejected = FALSE OR a.is_rejected IS NULL)
            AND a.vector IS NOT NULL
        """
        params = []
        
        # Apply Filters
        if filter_mode == "Same" and label_filter:
            query += " AND a.label = %s"
            params.append(label_filter)
        elif filter_mode == "Diff" and label_filter:
            query += " AND a.label != %s"
            params.append(label_filter)
            
        cursor.execute(query, tuple(params))
        
        results = []
        rows = cursor.fetchall()
        
        for row in rows:
            vec_blob = row['vector']
            if not vec_blob: continue
            
            # Postgres BYTEA comes as memoryview or bytes
            cand_vec = np.frombuffer(vec_blob, dtype=np.float32)
            norm_cand = np.linalg.norm(cand_vec)
            
            if norm_cand == 0: continue
            
            # Cosine Similarity
            sim = np.dot(target_vec, cand_vec) / (norm_target * norm_cand)
            
            if sim >= threshold:
                res = dict(row)
                res['similarity'] = float(sim)
                del res['vector']
                results.append(res)
        
        # Sort by similarity desc
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]

    def reject_overlapping_annotations(self, sentence_id, start, end, exclude_id=None):
        """
        Marks overlapping annotations in the same sentence as rejected.
        """
        cursor = self.conn.cursor()
        
        query = """
            UPDATE annotations 
            SET is_rejected = TRUE, is_accepted = FALSE
            WHERE sentence_id = %s 
            AND (is_rejected = FALSE OR is_rejected IS NULL)
            AND start_char < %s 
            AND end_char > %s
        """
        params = [sentence_id, end, start]
        
        if exclude_id is not None:
            query += " AND id != %s"
            params.append(exclude_id)
            
        cursor.execute(query, tuple(params))
        self.conn.commit()

    def execute_query(self, query, params=()):
        cursor = self.conn.cursor()
        # Basic conversion of SQLite ? to Postgres %s if needed
        if '?' in query:
            query = query.replace('?', '%s')
            
        cursor.execute(query, params)
        self.conn.commit()
        return cursor

    def update_memory_centroid(self, source_annotation_id, text_span, label, new_vector_blob):
        """
        Updates the centroid for a specific Source Annotation ID.
        """
        if new_vector_blob is None: return

        new_vec = np.frombuffer(new_vector_blob, dtype=np.float32)
        cursor = self.conn.cursor()
        
        # Check if exists
        cursor.execute("SELECT vector, count FROM memory_centroids WHERE source_annotation_id = %s", (source_annotation_id,))
        row = cursor.fetchone()
        
        if row:
            # Update existing
            old_vec_blob = row['vector']
            count = row['count']
            
            if old_vec_blob:
                # Postgres BYTEA handling
                if isinstance(old_vec_blob, memoryview):
                    old_vec_blob = bytes(old_vec_blob)
                old_vec = np.frombuffer(old_vec_blob, dtype=np.float32)
                updated_vec = (old_vec * count + new_vec) / (count + 1)
            else:
                updated_vec = new_vec
                
            updated_blob = updated_vec.tobytes()
            cursor.execute("""
                UPDATE memory_centroids 
                SET vector = %s, count = count + 1, last_updated = CURRENT_TIMESTAMP 
                WHERE source_annotation_id = %s
            """, (updated_blob, source_annotation_id))
        else:
            # Insert new
            cursor.execute("""
                INSERT INTO memory_centroids (source_annotation_id, text_span, label, vector, count)
                VALUES (%s, %s, %s, %s, 1)
            """, (source_annotation_id, text_span, label, new_vector_blob))
            
        self.conn.commit()

    def update_entity_centroid(self, text_span, label, new_vector_blob):
        """
        Updates the centroid for a specific (text_span, label) pair.
        """
        if new_vector_blob is None: return

        new_vec = np.frombuffer(new_vector_blob, dtype=np.float32)
        cursor = self.conn.cursor()
        
        # Check if exists
        cursor.execute("SELECT vector, count FROM entity_centroids WHERE text_span = %s AND label = %s", (text_span, label))
        row = cursor.fetchone()
        
        if row:
            # Update existing
            old_vec_blob = row['vector']
            count = row['count']
            
            if old_vec_blob:
                # Postgres BYTEA handling
                if isinstance(old_vec_blob, memoryview):
                    old_vec_blob = bytes(old_vec_blob)
                old_vec = np.frombuffer(old_vec_blob, dtype=np.float32)
                updated_vec = (old_vec * count + new_vec) / (count + 1)
            else:
                updated_vec = new_vec
                
            updated_blob = updated_vec.tobytes()
            cursor.execute("""
                UPDATE entity_centroids 
                SET vector = %s, count = count + 1, last_updated = CURRENT_TIMESTAMP 
                WHERE text_span = %s AND label = %s
            """, (updated_blob, text_span, label))
        else:
            # Insert new
            cursor.execute("""
                INSERT INTO entity_centroids (text_span, label, vector, count)
                VALUES (%s, %s, %s, 1)
            """, (text_span, label, new_vector_blob))
            
        self.conn.commit()

    def check_overlapping_annotations(self, sentence_id, start, end, exclude_id=None):
        """
        Checks for overlapping annotations that are NOT rejected.
        Returns list of dicts with keys: id, text_span, label, start_char, end_char
        """
        cursor = self.conn.cursor()
        query = """
            SELECT id, text_span, label, start_char, end_char FROM annotations 
            WHERE sentence_id = %s 
            AND (is_rejected = FALSE OR is_rejected IS NULL)
            AND start_char < %s 
            AND end_char > %s
        """
        params = [sentence_id, end, start]
        
        if exclude_id is not None:
            query += " AND id != %s"
            params.append(exclude_id)
            
        cursor.execute(query, tuple(params))
        return cursor.fetchall()
