from gptcache import Cache
from gptcache.adapter.api import get, put
from gptcache.embedding import Onnx
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.processor.pre import get_prompt

# Explicit cache instance (avoid global singleton)
embedding_model = Onnx()
chat_cache = Cache()
chat_cache.init(
    pre_embedding_func=get_prompt,  # we will pass data={"prompt": ...}
    embedding_func=embedding_model.to_embeddings,
    data_manager=get_data_manager(
        CacheBase("sqlite", path="cache.db"),
        VectorBase("faiss", dimension=embedding_model.dimension),
    ),
    similarity_evaluation=SearchDistanceEvaluation(),
)

# Dummy LLM (simulates an external model)
def dummy_llm(prompt: str) -> str:
    print("🔄 LLM called for:", prompt)
    return f"Dummy Ollama response: {prompt[::-1]}"

prompts = [
    "What is GPTCache?",
    "Tell me about GPTCache.",
    "What is GPTCache?",  # should be a cache hit now
]

for i, prompt in enumerate(prompts, 1):
    data = {"prompt": prompt}  # IMPORTANT: matches get_prompt preprocessor
    ans = get(prompt, cache_obj=chat_cache, data=data)
    if ans is None:
        # MISS -> call LLM -> store
        ans = dummy_llm(prompt)
        # NOTE: put expects 'data' as the first positional argument
        put(data, ans, cache_obj=chat_cache)
        print(f"\n🔴 MISS {i} -> stored")
    else:
        print(f"\n🟢 HIT  {i}")
    print(f"✅ Result {i}: {ans}")
