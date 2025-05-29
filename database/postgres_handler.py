import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, List, Any, Optional
from config.database_config import POSTGRES_CONFIG

class PostgresHandler:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=POSTGRES_CONFIG['host'],
            port=POSTGRES_CONFIG['port'],
            database=POSTGRES_CONFIG['database'],
            user=POSTGRES_CONFIG['user'],
            password=POSTGRES_CONFIG['password']
        )
        self.cur = self.conn.cursor(cursor_factory=RealDictCursor)

    def initialize_tables(self):
        """Create necessary tables for EHR data"""
        try:
            # Create patients table
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    id SERIAL PRIMARY KEY,
                    mrn VARCHAR(50) UNIQUE NOT NULL,
                    first_name VARCHAR(100),
                    last_name VARCHAR(100),
                    dob DATE,
                    gender VARCHAR(20),
                    contact_info JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create medical_records table
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS medical_records (
                    id SERIAL PRIMARY KEY,
                    patient_id INTEGER REFERENCES patients(id),
                    record_type VARCHAR(50),
                    record_date TIMESTAMP,
                    provider VARCHAR(100),
                    notes TEXT,
                    data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create medications table
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS medications (
                    id SERIAL PRIMARY KEY,
                    patient_id INTEGER REFERENCES patients(id),
                    medication_name VARCHAR(200),
                    dosage VARCHAR(100),
                    frequency VARCHAR(100),
                    start_date DATE,
                    end_date DATE,
                    prescriber VARCHAR(100),
                    status VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create appointments table
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS appointments (
                    id SERIAL PRIMARY KEY,
                    patient_id INTEGER REFERENCES patients(id),
                    appointment_type VARCHAR(100),
                    appointment_date TIMESTAMP,
                    provider VARCHAR(100),
                    status VARCHAR(50),
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            self.conn.commit()
            print("Successfully initialized PostgreSQL tables")
        except Exception as e:
            self.conn.rollback()
            print(f"Error initializing tables: {e}")
            raise

    def add_patient(self, patient_data: Dict[str, Any]) -> int:
        """Add a new patient to the database"""
        try:
            self.cur.execute("""
                INSERT INTO patients (mrn, first_name, last_name, dob, gender, contact_info)
                VALUES (%(mrn)s, %(first_name)s, %(last_name)s, %(dob)s, %(gender)s, %(contact_info)s)
                RETURNING id
            """, patient_data)
            patient_id = self.cur.fetchone()['id']
            self.conn.commit()
            return patient_id
        except Exception as e:
            self.conn.rollback()
            raise

    def add_medical_record(self, record_data: Dict[str, Any]) -> int:
        """Add a new medical record"""
        try:
            self.cur.execute("""
                INSERT INTO medical_records (patient_id, record_type, record_date, provider, notes, data)
                VALUES (%(patient_id)s, %(record_type)s, %(record_date)s, %(provider)s, %(notes)s, %(data)s)
                RETURNING id
            """, record_data)
            record_id = self.cur.fetchone()['id']
            self.conn.commit()
            return record_id
        except Exception as e:
            self.conn.rollback()
            raise

    def get_patient_records(self, patient_id: int) -> List[Dict[str, Any]]:
        """Get all medical records for a patient"""
        try:
            self.cur.execute("""
                SELECT * FROM medical_records 
                WHERE patient_id = %s 
                ORDER BY record_date DESC
            """, (patient_id,))
            return self.cur.fetchall()
        except Exception as e:
            raise

    def close(self):
        """Close database connection"""
        self.cur.close()
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 