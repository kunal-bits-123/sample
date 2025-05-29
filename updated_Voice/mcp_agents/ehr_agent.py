import asyncpg
import json
from typing import Dict, Any, Optional
from .base_agent import BaseMCPAgent

class EHRAgent(BaseMCPAgent):
    def __init__(self, groq_api_key: str, db_config: Dict[str, str], debug: bool = False):
        super().__init__("EHR Agent", groq_api_key)
        self.db_config = db_config
        self.debug = debug
        self.pool: Optional[asyncpg.Pool] = None

    async def _connect_db(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(**self.db_config)

    def _get_system_prompt(self) -> str:
        return """You are an EHR (Electronic Health Record) Agent responsible for managing patient information...
        """  # (Keep the full system prompt as-is)

    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        await self._connect_db()
        try:
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": f"Process this request: {message}"}
            ]

            response = await self._call_llm(messages)
            try:
                response_data = json.loads(response)
            except json.JSONDecodeError as e:
                # Fix JSON formatting if needed
                try:
                    fixed_response = response.replace('\\n', ' ').replace('\\"', '"')
                    response_data = json.loads(fixed_response)
                except:
                    return {"operation": "unknown", "status": "error", "data": None, "error": str(e)}

            if self.debug:
                print("[DEBUG] EHRAgent LLM response:", json.dumps(response_data, indent=2))

            operation = response_data.get("operation", "")
            data = response_data.get("data", {})
            error = response_data.get("error", None)

            if error:
                return {"operation": operation, "status": "error", "data": None, "error": error}
            if not data:
                return {"operation": operation, "status": "error", "data": None, "error": "No data provided"}

            if operation == "retrieve":
                return await self._handle_retrieve(data)
            elif operation == "update":
                return await self._handle_update(data)
            elif operation == "create":
                return await self._handle_create(data)
            else:
                return {"operation": operation, "status": "error", "data": None, "error": "Unsupported operation"}

        except Exception as e:
            return {"operation": "unknown", "status": "error", "data": None, "error": str(e)}

    async def _handle_retrieve(self, data: Dict[str, Any]) -> Dict[str, Any]:
        patient_id = data.get("patient_id")
        async with self.pool.acquire() as conn:
            if patient_id:
                record = await conn.fetchrow("SELECT * FROM patients WHERE patient_id = $1", patient_id)
                if record:
                    return {"status": "success", "data": dict(record), "error": None}
                else:
                    return {"status": "error", "data": None, "error": f"Patient {patient_id} not found"}
            else:
                records = await conn.fetch("SELECT * FROM patients")
                return {"status": "success", "data": [dict(r) for r in records], "error": None}

    async def _handle_update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        patient_id = data.get("patient_id")
        updates = data.get("updates", {})
        if not updates:
            return {"status": "error", "data": None, "error": "No update fields provided"}

        async with self.pool.acquire() as conn:
            update_fragments = []
            values = []
            for i, (field, value) in enumerate(updates.items(), start=1):
                update_fragments.append(f"{field} = ${i}")
                values.append(value)
            values.append(patient_id)
            query = f"UPDATE patients SET {', '.join(update_fragments)} WHERE patient_id = ${len(values)}"
            result = await conn.execute(query, *values)

            if result == "UPDATE 1":
                return {"status": "success", "data": {"patient_id": patient_id, "updates": updates}, "error": None}
            else:
                return {"status": "error", "data": None, "error": f"Patient {patient_id} not found"}

    async def _handle_create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        async with self.pool.acquire() as conn:
            patient_id = f"P{int((await conn.fetchval('SELECT COUNT(*) FROM patients')) or 0) + 1:03d}"
            query = """
                INSERT INTO patients (patient_id, name, medical_history, medications, allergies)
                VALUES ($1, $2, $3, $4, $5)
            """
            await conn.execute(query,
                               patient_id,
                               data.get("name", "Unknown"),
                               json.dumps(data.get("medical_history", [])),
                               json.dumps(data.get("medications", [])),
                               json.dumps(data.get("allergies", [])))
            return {"status": "success", "data": {"patient_id": patient_id, **data}, "error": None}
