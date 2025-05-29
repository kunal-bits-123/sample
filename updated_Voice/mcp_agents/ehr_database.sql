ehr_agent = EHRAgent(
    groq_api_key="your-groq-key",
    db_config={
        "user": "postgres",
        "password": "yourpassword",
        "database": "ehr_db",
        "host": "localhost",
        "port": "5432"
    },
    debug=True
)
