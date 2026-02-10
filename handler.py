"""
RunPod Serverless Handler for vLLM
Following: https://docs.runpod.io/serverless/overview#handler-functions
"""

import runpod
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import asyncio
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class vLLMEngineManager:
    """Singleton vLLM engine manager"""
    _instance = None
    _engine = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self):
        """Initialize the vLLM engine once"""
        if self._initialized:
            logger.info("Engine already initialized")
            return

        try:
            logger.info("Initializing vLLM AsyncEngine...")
            
            engine_args = AsyncEngineArgs(
                model="openai/gpt-oss-20b",
                dtype="auto",
                gpu_memory_utilization=0.9,
                max_model_len=4096,
                trust_remote_code=False,
                tokenizer_mode="auto",
                disable_log_stats=False,
                max_num_seqs=256,
                enforce_eager=False,
                enable_chunked_prefill=False,
            )
            
            self._engine = AsyncLLMEngine.from_engine_args(engine_args)
            self._initialized = True
            logger.info("✓ vLLM engine initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize vLLM engine: {e}", exc_info=True)
            raise

    @property
    def engine(self) -> AsyncLLMEngine:
        """Get the engine instance"""
        if not self._initialized:
            self.initialize()
        return self._engine

    async def generate(self, prompt: str, params: SamplingParams) -> str:
        """Generate text using vLLM"""
        request_id = f"req_{id(prompt)}"
        
        try:
            # Create async generator
            results_generator = self.engine.generate(
                prompt=prompt,
                sampling_params=params,
                request_id=request_id
            )
            
            # Iterate through results
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            
            # Return generated text
            if final_output and final_output.outputs:
                return final_output.outputs[0].text
            
            return ""
            
        except Exception as e:
            logger.error(f"Generation failed for {request_id}: {e}")
            raise


# Global engine manager
engine_manager = vLLMEngineManager()


def validate_input(job_input: Dict[str, Any]) -> Optional[str]:
    """
    Validate the input according to RunPod best practices
    Returns error message if invalid, None if valid
    """
    if not job_input:
        return "No input provided"
    
    if "prompt" not in job_input:
        return "Missing required field: 'prompt'"
    
    if not isinstance(job_input["prompt"], str):
        return "'prompt' must be a string"
    
    if len(job_input["prompt"]) == 0:
        return "'prompt' cannot be empty"
    
    # Validate optional parameters
    if "max_tokens" in job_input:
        if not isinstance(job_input["max_tokens"], (int, float)):
            return "'max_tokens' must be a number"
        if job_input["max_tokens"] <= 0:
            return "'max_tokens' must be positive"
    
    if "temperature" in job_input:
        if not isinstance(job_input["temperature"], (int, float)):
            return "'temperature' must be a number"
        if not 0 <= job_input["temperature"] <= 2:
            return "'temperature' must be between 0 and 2"
    
    return None


async def async_generator_handler(job: Dict[str, Any]):
    """
    Async generator handler for streaming responses
    Follows RunPod's streaming pattern
    """
    job_input = job.get("input", {})
    
    # Validate input
    error = validate_input(job_input)
    if error:
        yield {"error": error}
        return
    
    # Extract parameters
    prompt = job_input["prompt"]
    
    # Build sampling parameters
    sampling_params = SamplingParams(
        temperature=float(job_input.get("temperature", 0.7)),
        top_p=float(job_input.get("top_p", 0.9)),
        max_tokens=int(job_input.get("max_tokens", 512)),
        top_k=int(job_input.get("top_k", -1)),
        presence_penalty=float(job_input.get("presence_penalty", 0.0)),
        frequency_penalty=float(job_input.get("frequency_penalty", 0.0)),
    )
    
    try:
        # Initialize engine if needed
        if not engine_manager._initialized:
            engine_manager.initialize()
        
        request_id = f"stream_{id(job)}"
        
        # Stream generation results
        async for request_output in engine_manager.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id
        ):
            # Yield incremental output
            if request_output.outputs:
                yield {
                    "text": request_output.outputs[0].text,
                    "finished": request_output.finished,
                    "tokens_generated": len(request_output.outputs[0].token_ids)
                }
        
    except Exception as e:
        logger.error(f"Streaming generation error: {e}", exc_info=True)
        yield {"error": str(e)}


async def async_handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Async handler for non-streaming responses
    This is the main handler following RunPod's async pattern
    
    Args:
        job: Dictionary containing 'input' and optionally 'id'
        
    Returns:
        Dictionary with 'output' or 'error'
    """
    job_input = job.get("input", {})
    
    # Validate input
    error = validate_input(job_input)
    if error:
        return {"error": error}
    
    # Extract parameters
    prompt = job_input["prompt"]
    
    # Build sampling parameters
    sampling_params = SamplingParams(
        temperature=float(job_input.get("temperature", 0.7)),
        top_p=float(job_input.get("top_p", 0.9)),
        max_tokens=int(job_input.get("max_tokens", 512)),
        top_k=int(job_input.get("top_k", -1)),
        presence_penalty=float(job_input.get("presence_penalty", 0.0)),
        frequency_penalty=float(job_input.get("frequency_penalty", 0.0)),
        stop=job_input.get("stop", None),
    )
    
    try:
        # Initialize engine if needed (lazy initialization)
        if not engine_manager._initialized:
            logger.info("First request - initializing engine...")
            engine_manager.initialize()
        
        # Generate response
        logger.info(f"Generating response for prompt: {prompt[:50]}...")
        output_text = await engine_manager.generate(prompt, sampling_params)
        
        # Return result in RunPod format
        return {
            "output": output_text,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)
        return {
            "error": str(e),
            "status": "failed"
        }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous wrapper for the async handler
    RunPod supports both sync and async handlers
    
    For async handlers, RunPod will automatically handle the event loop
    But if you need sync compatibility, use this wrapper
    """
    return asyncio.run(async_handler(job))


# RunPod initialization
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 Starting vLLM RunPod Serverless Handler")
    logger.info("=" * 60)
    
    # Choose your handler type:
    
    # Option 1: Async handler (recommended for vLLM)
    runpod.serverless.start({
        "handler": async_handler,  # Pass async function directly
        "return_aggregate_stream": True  # Enable if using streaming
    })