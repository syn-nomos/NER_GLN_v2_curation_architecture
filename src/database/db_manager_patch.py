    def get_status_counts(self, dataset_split=None):
        """Returns a dictionary of status counts based on dynamic annotation state."""
        cursor = self.conn.cursor()
        
        params = []
        split_clause = ""
        if dataset_split:
            split_clause = " AND s.dataset_split = ?"
            params.append(dataset_split)
        
        # Count Pending
        query_pending = f"""
            SELECT COUNT(DISTINCT s.id) as count
            FROM sentences s
            JOIN annotations a ON s.id = a.sentence_id
            WHERE a.is_accepted = 0 
            AND (a.is_rejected = 0 OR a.is_rejected IS NULL)
            {split_clause}
        """
        cursor.execute(query_pending, tuple(params))
        r_p = cursor.fetchone()
        pending_count = r_p['count'] if r_p else 0
        
        # Count Total in split
        query_total = f"SELECT COUNT(*) as count FROM sentences s WHERE 1=1 {split_clause}"
        cursor.execute(query_total, tuple(params))
        r_t = cursor.fetchone()
        total_count = r_t['count'] if r_t else 0
        
        completed_count = total_count - pending_count
        
        return {
            'pending': pending_count,
            'completed': completed_count
        }
