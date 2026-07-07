import os
import json
from datetime import date
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, create_model, Field
from openai import OpenAI

app = FastAPI(title="DataBridge Dynamic ETL Pipeline")

# Enable CORS as requested by the specification
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI Client (Will naturally read OPENAI_API_KEY from environment vars)
client = OpenAI()

# Strict mapping of incoming user string schemas to proper Python/Pydantic validation types
# By default, all fields allow None (null) if they are missing from the parsed text source.
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

    # 1. Dynamically build the validation fields map at runtime
    fields = {}
    for field_name, type_str in schema_def.items():
        if type_str not in TYPE_MAPPING:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported type '{type_str}' for field '{field_name}'"
            )
        fields[field_name] = TYPE_MAPPING[type_str]

    # Create the dedicated runtime validation model
    DynamicModel = create_model("DynamicExtractionModel", **fields)

    # 2. Extract a JSON schema definition directly into OpenAI's Strict Structured Outputs format
    json_schema = DynamicModel.model_json_schema()
    
    # Force OpenAI to treat all keys as strictly required (they can still have null values)
    json_schema["required"] = list(schema_def.keys())
    json_schema["additionalProperties"] = False  # Strict rule alignment

    prompt = (
        "You are an expert data extraction agent. Analyze the provided text source and populate "
        "every single field in the target schema structure. "
        "Rules:\n"
        "- If a field is not present or cannot be extracted from the text, you MUST populate it as null.\n"
        "- Dates must strictly follow ISO format: YYYY-MM-DD.\n"
        "- Do not make up information."
    )

    try:
        # 3. Call OpenAI using native strict schema validation parameters
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
        extracted_json = json.loads(raw_content)

        # 4. Coerce data types using the dynamically generated Pydantic model
        # This converts strings into Python numbers/dates natively, verifying structural integrity
        validated_data = DynamicModel(**extracted_json)

        # 5. Return native serialization compliant with standard endpoint structures (dates -> strings)
        return validated_data.model_dump(mode="json")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dynamic extraction failed: {str(e)}")


# CRITICAL FIX:
