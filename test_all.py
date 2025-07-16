import subprocess

def run_test(name, filename):
    print(f"\n🔧 Running {name}...\n" + "-"*60)
    result = subprocess.run(["python", filename], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("❌ Error Output:\n" + result.stderr)
    else:
        print(f"✅ {name} completed without errors.\n")

if __name__ == "__main__":
    run_test("Dataset Loading Test", "dataset_loading_test.py")
    run_test("Ollama Connection Test", "ollama_test.py")
    run_test("GPTCache Basic Test", "GPTCache_basic_test.py")
