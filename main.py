import os
import json
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI(title="DataBridge Dynamic ETL Pipeline")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI()

# 100% Compliant OpenAI Strict JSON Schema definitions using anyOf for nullable fields
OPENAI_TYPE_MAP = {
    "string": {
        "anyOf": [{"type": "string"}, {"type": "null"}]
    },
    "integer": {
        "anyOf": [{"type": "integer"}, {"type": "null"}]
    },
    "float": {
        "anyOf": [{"type": "number"}, {"type": "null"}]
    },
    "boolean": {
        "anyOf": [{"type": "boolean"}, {"type": "null"}]
    },
    "date": {
        "anyOf": [
            {"type": "string", "description": "ISO date string formatted as YYYY-MM-DD"},
            {"type": "null"}
        ]
    },
    "array[string]": {
        "anyOf": [
            {"type": "array", "items": {"type": "string"}},
            {"type": "null"}
        ]
    },
    "array[integer]": {
        "anyOf": [
            {"type": "array", "items": {"type": "integer"}},
            {"type": "null"}
        ]
    },
}

class ExtractionRequest(BaseModel):
    text: str
    schema_def: Dict[str, str] = Field(..., alias="schema")

    class Config:
        populate_by_name = True


@app.post("/dynamic-extract")
async def dynamic_extract(payload: ExtractionRequest):
    text = payload.text
    schema_def = payload.schema_def

    # 1. Manually build the target properties structure to be compliant with OpenAI Strict Mode
    properties = {}
    for field_name, type_str in schema_def.items():
        if type_str not in OPENAI_TYPE_MAP:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported type '{type_str}' for field '{field_name}'"
            )
        properties[field_name] = OPENAI_TYPE_MAP[type_str]

    # 2. Build the perfect base JSON Schema definition matching OpenAI guidelines
    json_schema = {
        "type": "object",
        "properties": properties,
        "required": list(schema_def.keys()),  # Strict mode requires all properties to be declared here
        "additionalProperties": False        # Strict mode requires additionalProperties to be false
    }

    prompt = (
        "You are an expert data extraction agent. Analyze the provided text source and populate "
        "every single field in the target schema.\n\n"
        "Rules:\n"
        "- If a field is not present or cannot be explicitly found in the text, you MUST return null for that field.\n"
        "- Dates must strictly follow ISO format: YYYY-MM-DD.\n"
        "- Numbers/integers must be valid JSON numeric outputs, not strings.\n"
        "- Do not extrapolate or hallucinate data."
    )

    try:
        # 3. Call OpenAI using strict schema validation parameters
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Text to analyze:\n{text}"}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "dynamic_extraction_schema",
                    "strict": True,
                    "schema": json_schema
                }
            },
            temperature=0.0
        )

        raw_content = response.choices[0].message.content
        extracted_data = json.loads(raw_content)

        # 4. Runtime type conversion & coercion logic
        final_output = {}
        for field_name, type_str in schema_def.items():
            val = extracted_data.get(field_name)
            
            if val is None:
                final_output[field_name] = None
                continue

            try:
                if type_str == "integer":
                    final_output[field_name] = int(val)
                elif type_str == "float":
                    final_output[field_name] = float(val)
                elif type_str == "boolean":
                    if isinstance(val, str):
                        final_output[field_name] = val.lower() in ("true", "1", "yes")
                    else:
                        final_output[field_name] = bool(val)
                elif type_str == "date":
                    if isinstance(val, str):
                        cleaned_date = val.split("T")[0].strip()
                        parsed_date = datetime.strptime(cleaned_date, "%Y-%m-%d").date()
                        final_output[field_name] = parsed_date.isoformat()
                    else:
                        final_output[field_name] = None
                elif type_str == "array[integer]":
                    final_output[field_name] = [int(x) for x in val if x is not None]
                elif type_str == "array[string]":
                    final_output[field_name] = [str(x) for x in val if x is not None]
                else:
                    final_output[field_name] = str(val)
            except Exception:
                # Fallback to null safely if format parsing encounters a breakdown
                final_output[field_name] = None

        return final_output

    except Exception as e:
        # Crucial: Prints the actual underlying error trace in your Render logs for visibility
        print(f"Extraction exception caught: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dynamic extraction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
