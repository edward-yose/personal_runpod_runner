import runpod
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class vLLMEngine:
    def __init__(self):
        self.llm = None
        try:
            logger.info("Initializing vLLM engine...")
            engine_args = AsyncEngineArgs(
                model="openai/gpt-oss-20b",
                dtype="auto",
                gpu_memory_utilization=0.9,
                max_model_len=4096,
                trust_remote_code=False,
                tokenizer_mode="auto",
                disable_log_stats=False,
                enable_chunked_prefill=False,
            )
            self.llm = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info("vLLM engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}", exc_info=True)
            raise

    async def generate(self, prompt: str, sampling_params: SamplingParams):
        """Generate text from prompt"""
        request_id = f"req_{id(prompt)}"
        
        results_generator = self.llm.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id
        )
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
            
        if final_output:
            return final_output.outputs[0].text
        return ""

# Global engine instance
engine = None

async def async_handler(job):
    """Async handler for RunPod"""
    global engine
    
    if engine is None:
        engine = vLLMEngine()
    
    job_input = job["input"]
    prompt = job_input.get("prompt", "")
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=job_input.get("temperature", 0.7),
        top_p=job_input.get("top_p", 0.9),
        max_tokens=job_input.get("max_tokens", 512),
    )
    
    try:
        output = await engine.generate(prompt, sampling_params)
        return {"output": output}
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        return {"error": str(e)}

def handler(job):
    """Synchronous wrapper for RunPod"""
    return asyncio.run(async_handler(job))


runpod.serverless.start({"handler": handler})