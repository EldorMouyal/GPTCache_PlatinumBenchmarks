from gptcache import cache
from gptcache.adapter.api import put, get
from gptcache.embedding import Onnx
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

# Step 1: Setup embedding model
embedding_model = Onnx()

# Step 2: Initialize cache with custom embedding and similarity logic
cache.init(
    embedding_func=embedding_model.to_embeddings,
    data_manager=get_data_manager(
        CacheBase("sqlite"),
        VectorBase("faiss", dimension=embedding_model.dimension)
    ),
    similarity_evaluation=SearchDistanceEvaluation()
)

# Step 3: Dummy LLM to simulate an external call
def dummy_llm(prompt: str) -> str:
    print("🔄 LLM called for:", prompt)
    return f"Dummy Ollama response: {prompt[::-1]}"

# Step 4: Test prompts
prompts = [
    "What is GPTCache?",
    "Tell me about GPTCache.",
    "What is GPTCache?"  # Should trigger a cache hit
]

for i, prompt in enumerate(prompts):
    cached = get(prompt)
    if cached:
        print(f"\n🟢 Cache HIT {i+1}: {cached}")
    else:
        print(f"\n🔴 Cache MISS {i+1}")
        response = dummy_llm(prompt)
        put(prompt, response)
        print(f"✅ Stored {i+1}: {response}")
