import os
import json
from datetime import date
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, create_model, Field
from openai import OpenAI

app = FastAPI(title="DataBridge Dynamic ETL Pipeline")

# Enable CORS as requested
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI Client (Ensure OPENAI_API_KEY is set in environment variables)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Mapping incoming string type definitions to actual Python/Pydantic types
TYPE_MAPPING = {
    "string": (Optional[str], Field(default=None)),
    "integer": (Optional[int], Field(default=None)),
    "float": (Optional[float], Field(default=None)),
    "boolean": (Optional[bool], Field(default=None)),
    "date": (Optional[date], Field(default=None)),
    "array[string]": (Optional[List[str]], Field(default=None)),
    "array[integer]": (Optional[List[int]], Field(default=None)),
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

    # 1. Dynamically build the Pydantic model at runtime
    fields = {}
    for field_name, type_str in schema_def.items():
        if type_str not in TYPE_MAPPING:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported type '{type_str}' for field '{field_name}'"
            )
        fields[field_name] = TYPE_MAPPING[type_str]

    # Create a dynamic Pydantic model that defaults missing fields to None
    DynamicModel = create_model("DynamicExtractionModel", **fields)

    # 2. Extract a JSON schema to pass directly into OpenAI's Structured Outputs (JSON Schema)
    # This guarantees the LLM returns exactly the required keys and types
    json_schema = DynamicModel.model_json_schema()
    
    # Enforce that all fields from the requested schema are included in the JSON output object
    json_schema["required"] = list(schema_def.keys())

    prompt = (
        "You are an expert data extraction system. Extract structured data from the text provided "
        "matching the requested schema exactly. For fields that are completely missing or cannot "
        "be explicitly deduced from the text, return null. "
        "Ensure dates are formatted as ISO YYYY-MM-DD strings."
    )

    try:
        # 3. Call OpenAI using Strict JSON Schema Mode
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Highly accurate and fast for structured extraction
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Text to extract from:\n{text}"}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "dynamic_extraction",
                    "strict": True,
                    "schema": json_schema
                }
            },
            temperature=0.0 # Force determinism
        )

        raw_content = response.choices[0].message.content
        extracted_json = json.loads(raw_content)

        # 4. Use the dynamic Pydantic model to validate and coerce data types 
        # (e.g. converting date strings to date objects, numeric strings to ints/floats)
        validated_data = DynamicModel(**extracted_json)

        # 5. Serialize back to native JSON format (dates automatically turn to ISO format strings)
        return validated_data.model_dump(mode="json")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
