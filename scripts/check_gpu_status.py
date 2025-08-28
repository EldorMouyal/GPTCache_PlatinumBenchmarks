#!/usr/bin/env python3
"""
Check the status of Ollama server (local or remote GPU).

Usage:
    python scripts/check_gpu_status.py
    
This script reads the current base_url from experiment.yaml and tests connectivity.
"""

import requests
import yaml
import json
import time
from pathlib import Path
import sys


def load_config() -> dict:
    """Load the current experiment configuration."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    config_path = project_root / "experiments" / "experiment.yaml"
    
    if not config_path.exists():
        print(f"❌ Error: Config file not found at {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def check_ollama_status(base_url: str, model: str) -> None:
    """Check Ollama server status and available models."""
    
    print(f"🔍 Checking Ollama at: {base_url}")
    print("=" * 50)
    
    # Test 1: Basic connectivity
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            print("✅ Server is responding")
            models_data = response.json()
            available_models = [m['name'] for m in models_data.get('models', [])]
            print(f"📦 Available models: {available_models}")
        else:
            print(f"⚠️  Server responded with status {response.status_code}")
            print(f"   Response: {response.text}")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Test 2: Check if required model is available
    if model in available_models:
        print(f"✅ Required model '{model}' is available")
    else:
        print(f"⚠️  Required model '{model}' not found")
        print(f"   Run: ollama pull {model}")
        return
    
    # Test 3: Performance test with simple query
    print("\n🧪 Running performance test...")
    test_prompt = "What is 2+2? Answer briefly."
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": test_prompt,
                "stream": False
            },
            timeout=30
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', 'No response')[:100]
            duration = end_time - start_time
            
            print(f"✅ Generation successful ({duration:.2f}s)")
            print(f"📝 Response: {response_text}...")
            
            # Check for GPU usage indicators
            if "gpu" in response.text.lower() or duration < 5:
                print("🚀 Likely using GPU acceleration")
            else:
                print("🐌 Possibly CPU-only (slow response)")
                
        else:
            print(f"❌ Generation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out (server might be overloaded)")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")


def main():
    try:
        config = load_config()
        
        base_url = config.get('model', {}).get('base_url', 'http://localhost:11434')
        model = config.get('model', {}).get('name', 'unknown')
        
        print("🔧 Current Configuration:")
        print(f"   Base URL: {base_url}")
        print(f"   Model: {model}")
        print()
        
        check_ollama_status(base_url, model)
        
        print("\n" + "=" * 50)
        if "localhost" in base_url:
            print("💡 Using local Ollama. For GPU acceleration:")
            print("   1. Run the Kaggle notebook: kaggle_ollama_server.ipynb")
            print("   2. Get the ngrok URL")
            print("   3. Run: python scripts/setup_gpu_config.py <ngrok_url>")
        else:
            print("🚀 Using remote GPU Ollama")
            print("   Make sure the Kaggle notebook is still running!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()