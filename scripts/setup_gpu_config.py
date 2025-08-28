#!/usr/bin/env python3
"""
Setup GPU configuration by updating the base_url in experiment.yaml

Usage:
    python scripts/setup_gpu_config.py <ollama_url>
    python scripts/setup_gpu_config.py http://localhost:11434     # Reset to local
    python scripts/setup_gpu_config.py https://abc123.ngrok.io    # Use GPU server
"""

import sys
import yaml
from pathlib import Path


def update_ollama_url(config_path: str, new_url: str) -> None:
    """Update the base_url in the experiment config file."""
    
    # Read current config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Store old URL for logging
    old_url = config.get('model', {}).get('base_url', 'unknown')
    
    # Update the URL
    if 'model' not in config:
        config['model'] = {}
    config['model']['base_url'] = new_url
    
    # Write updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Updated {config_path}")
    print(f"   Old URL: {old_url}")
    print(f"   New URL: {new_url}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/setup_gpu_config.py <ollama_url>")
        print("Examples:")
        print("  python scripts/setup_gpu_config.py http://localhost:11434")
        print("  python scripts/setup_gpu_config.py https://abc123.ngrok.io")
        sys.exit(1)
    
    new_url = sys.argv[1]
    
    # Validate URL format
    if not new_url.startswith(('http://', 'https://')):
        print("❌ Error: URL must start with http:// or https://")
        sys.exit(1)
    
    # Find project root and config file
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    config_path = project_root / "experiments" / "experiment.yaml"
    
    if not config_path.exists():
        print(f"❌ Error: Config file not found at {config_path}")
        sys.exit(1)
    
    try:
        update_ollama_url(str(config_path), new_url)
        
        # Show next steps
        if "localhost" in new_url:
            print("\n🏠 Switched to local Ollama")
            print("   Make sure Ollama is running locally: ollama serve")
        else:
            print("\n🚀 Switched to remote GPU Ollama")
            print("   Make sure your remote server is accessible")
            print(f"   Test with: python scripts/check_gpu_status.py")
            
    except Exception as e:
        print(f"❌ Error updating config: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()