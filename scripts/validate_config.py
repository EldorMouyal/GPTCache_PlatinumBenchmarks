# scripts/validate_config.py
import yaml
import json
from jsonschema import Draft202012Validator

def validate_yaml():
    with open("experiments/experiment.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    print("✅ experiment.yaml is valid YAML")
    return data

def validate_schema():
    with open("schema/result.schema.json", "r", encoding="utf-8") as f:
        schema = json.load(f)
    Draft202012Validator.check_schema(schema)
    print("✅ result.schema.json is a valid JSON Schema")
    return schema

if __name__ == "__main__":
    cfg = validate_yaml()
    schema = validate_schema()
    print("\n--- experiment.yaml content ---")
    print(cfg)
