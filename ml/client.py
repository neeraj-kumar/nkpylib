"""Client for our LLM/embeddings server.

You can safely import this code without needing numpy, torch, etc.

There are a number of different "core" functions, all of which make calls to the server. These
functions are wrapped in a `FunctionWrapper` object which allows you to call them in various ways:
- `single`: single input synchronous mode, passing a single input and returning a single output
- `batch`: batch input synchronous mode, processing a list of inputs and returning a list of outputs
  - Note that this uses a ThreadPoolExecutor to parallelize the calls under the hood
  - '__call__' is an alias for 'batch'
- `single_async`: single input asynchronous mode, passing a single input and returning a single output
- `batch_async`: batch input asynchronous mode, processing a list of inputs and returning a list of outputs
- `single_future`: single input futures mode which returns a future with the started computation
- `batch_futures`: batch input futures mode which returns a list of futures with the started computations

In addition, the wrapper allows you to specify the following:
- `mode`: 'raw' or 'final'. The 'raw' mode returns the JSON response from the server as a dict. The
  'final' mode processes the response and returns a more user-friendly output, such as the text of
  the llm call, or the embedding from the response (see below for details per func). We default to
  'final'.
- `executor`: A concurrent.futures Executor to use for parallel calls. By default, we create a new
  `ThreadPoolExecutor` for each FunctionWrapper instance. Note that because we're calling a server
  (running a different process) for the actual work, there is no benefit to using a
  `ProcessPoolExecutor`.
- `progress_msg`: A string to display as a progress message using tqdm. If set to an empty string
  (the default), no progress bar is shown. If specified, it is used as the description for the tqdm
  progress bar.

The core functions and their inputs and other parameters are:
- `call_llm`: LLM chat completion with a given input prompt or list of prompts
  - `model`: The model to use. Default is 'mistral-7b-instruct-v0.2.Q4_K_M.gguf'
  - In `final` mode, returns the text of the first choice
- `embed_text`: Embeds an input string using the specified model
  - `model`: The model to use. Default is 'sentence', which maps to 'BAAI/bge-large-en-v1.5'
  - In `final` mode, returns the embedding directly
- `strsim`: Computes the similarity between two strings (passed as a tuple)
  - `model`: The model to use. Default is 'clip', which maps to 'openai/clip-vit-large-patch14'
  - In `final` mode, returns the similarity score
- `embed_image_url`: Embeds an image from a URL or local path using the specified model
  - `model`: The model to use. Default is 'image', which maps to 'openai/clip-vit-large-patch14'
  - In `final` mode, returns the embedding directly
- `embed_image`: Embeds an image (loaded PIL image) using the specified model
  - Exactly the same as `embed_image_url`, but takes a PIL image as input
- `get_text`: Extracts text from a file (pdf using pdftotext, image using ocr, or text)
  - In `final` mode, returns the extracted text


So if you wanted raw outputs for string similarity using the 'sentence' model, returned as a list of
futures, you would do:

    from nkpylib.ml.client import strsim

    strsim.mode = 'raw'
    futures = strsim.batch_futures([("dog", "cat"), ("dog", "philosophy")], model='sentence')


The server also supports caching for all calls (except Images passed in directly). This is enabled by
default, but can be disabled by passing `use_cache=False` to any of the function calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
import time

from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps
from os.path import join
from tqdm import tqdm
from typing import Any, Optional, Union, Sequence, Callable, Iterator

import requests

from nkpylib.ml.constants import SERVER_BASE_URL, SERVER_API_VERSION, Role, Msg

logger = logging.getLogger(__name__)

def chunked(lst: Sequence[Any], n: int) -> Iterator[Sequence[Any]]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# typedef for raw json response from server #TODO tighten this up
ResponseT = dict[str, Any]

class FunctionWrapper:
    """Wrapper for a function to make it executable in various ways."""
    core_func: Callable[..., ResponseT]
    final_func: Optional[Callable[[ResponseT], Any]]
    executor: ThreadPoolExecutor
    progress_msg: str

    def __init__(self,
                 core_func: Callable[..., ResponseT],
                 final_func: Optional[Callable[[ResponseT], Any]] = None,
                 mode: str = 'final',
                 executor: Optional[ThreadPoolExecutor] = None,
                 progress_msg: str = ''):
        """Initialize the FunctionWrapper with the core function and optional final function.

        The core function should take the input and any additional arguments and return the raw
        JSON response from the server. The final function should take the raw response and return
        a more user-friendly output. If no final function is provided, we default to 'raw' mode (and
        disallow setting mode to 'final').

        By default, we use 'final' mode, which processes the response and returns a more
        user-friendly output. You can set the mode to 'raw' to return the raw JSON response.

        We also allow you to pass in an executor to use for parallel calls. By default, we create a
        new `ThreadPoolExecutor` for each `FunctionWrapper` instance.

        The core function should take the input and any additional arguments and return the raw
        JSON response from the server. The final function should take the raw response and return
        a more user-friendly output. If no final function is provided, we default to 'raw' mode (and
        disallow setting mode to 'final').

        By default, we use 'final' mode, which processes the response and returns a more
        user-friendly output. You can set the mode to 'raw' to return the raw JSON response.

        We also allow you to pass in an executor to use for parallel calls. By default, we create a
        new ThreadPoolExecutor for each FunctionWrapper instance.

        The `progress_msg` parameter allows you to specify a message to display with a tqdm progress
        bar during batch processing. If `progress_msg` is an empty string (the default), no progress
        bar is shown. If specified, it is used as the description for the tqdm progress bar.
        """
        self.core_func = core_func
        if executor is None:
            executor = ThreadPoolExecutor()
        self.executor = executor
        self.final_func = final_func
        self.progress_msg = progress_msg
        if not final_func:
            mode = "raw"
        self.mode = mode

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        """Sets the mode to `value`, checking that it's valid for this function."""
        if value == "final" and not self.final_func:
            raise ValueError("This function does not have a final output processor.")
        self._mode = value

    def single(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        """Single input synchronous mode, passing a single input"""
        result = self.core_func(input, *args, **kwargs)
        return self._process_output(result)

    def batch(self, inputs: list[Any], *args: Any, **kwargs: Any) -> list[Any]:
        """Batch input synchronous mode, processing a list of inputs"""
        futures = self.batch_futures(inputs, *args, **kwargs)
        if self.progress_msg:
            return [f.result() for f in tqdm(futures, desc=self.progress_msg)]
        else:
            return [f.result() for f in futures]

    def __call__(self, inputs: list[Any], *args: Any, **kwargs: Any) -> list[Any]:
        """Call the batch sync function"""
        return self.batch(inputs, *args, **kwargs)

    async def single_async(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        """Single input asynchronous mode, passing a single input.

        This is an async function that can be awaited.
        """
        loop = asyncio.get_running_loop()
        # async executor doesn't allow kwargs, so make a partial
        task = partial(self.core_func, input, *args, **kwargs)
        #TODO see if we want to set this executor to our instance of executor
        result = await loop.run_in_executor(None, task)
        return self._process_output(result)

    async def batch_async(self, inputs: list[Any], *args: Any, **kwargs: Any) -> list[Any]:
        """Batch input asynchronous mode, processing a list of inputs."""
        loop = asyncio.get_running_loop()
        task = partial(self.core_func, *args, **kwargs)
        tasks = [loop.run_in_executor(None, task, inp) for inp in inputs]
        if self.progress_msg:
            tasks = list(tqdm(asyncio.as_completed(tasks), total=len(inputs), desc=self.progress_msg))
        results = await asyncio.gather(*tasks)
        return [self._process_output(result) for result in results]

    def single_future(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        """Single input futures mode which returns a future with the started computation."""
        def task():
            ret = self.core_func(input, *args, **kwargs)
            return self._process_output(ret)

        return self.executor.submit(task)

    def batch_futures(self, inputs: list[Any], *args: Any, **kwargs: Any) -> list[Any]:
        """Batch input futures mode which returns a list of futures with the started computations."""
        return [self.single_future(inp, *args, **kwargs) for inp in inputs]

    def _process_output(self, result: Any) -> Any:
        """Process the output of the core function based on our `mode`."""
        if self.mode == "raw":
            return result
        elif self.mode == "final":
            assert self.final_func is not None
            return self.final_func(result)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def execution_wrapper(**decorator_kwargs):
    """Decorator for a function to make it executable in various ways.

    You can use this with kwargs which are passed to the FunctionWrapper initialization.
    For example:

    @execution_wrapper(final_func=lambda x: x['choices'][0]['text'])
    def my_func(input):
        ...
    """
    def actual_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            """Pass decorator arguments into ParallelWrapper initialization"""
            return FunctionWrapper(func, **decorator_kwargs)

        ret = wrapper()
        ret.__name__ = f"{decorator_kwargs.get('name', func.__name__)} (wrapped)"
        return ret
    return actual_decorator


## CORE FUNCTIONALITY
def single_call(endpoint: str, model:Optional[str]=None, **kw) -> ResponseT:
    """Calls a single endpoint on the server. Returns the raw json response (as a dict)."""
    url = f"{SERVER_BASE_URL}/v{SERVER_API_VERSION}/{endpoint}"
    data = dict(**kw)
    if model is not None:
        data['model'] = model
    return requests.post(url, json=data).json()

@execution_wrapper(final_func=lambda x: x['choices'][0]['text'])
def call_llm_completion(prompt: str, max_tokens:int =1024, model:Optional[str] =None, use_cache=True, **kw) -> ResponseT:
    """Calls our local llm server for a completion.

    Uses the 'mistral-7b-instruct-v0.2.Q4_K_M.gguf' model by default.

    Returns the raw json response (as a dict).
    """
    return single_call("completions",
                       prompt=prompt,
                       max_tokens=max_tokens,
                       model=model,
                       use_cache=use_cache,
                       **kw)


def call_llm_impl(prompts: str|list[Msg],
                  max_tokens:int,
                  image: str='',
                  model:Optional[str] =None,
                  session_id:str ='',
                  use_cache=True,
                  session_cache={},
                  **kw) -> ResponseT:
    """Implementation for llm and vlm chat completions.

    Returns the raw json response (as a dict).
    """
    if session_id:
        # we need to append the input prompt(s) to the cached history for this session
        if session_id in session_cache:
            lst = session_cache[session_id][:]
        else:
            lst = []
        if isinstance(prompts, str):
            prompts = [('user', prompts)]
        lst += prompts
        prompts = lst
        logger.debug(f'for session {session_id}, using prompts {prompts}')
    call_kwargs = dict(prompts=prompts, max_tokens=max_tokens, model=model, use_cache=use_cache, **kw)
    if image:
        if isinstance(image, str): # it's already an image or url
            ret = single_call("vlm", image=image, **call_kwargs)
        else: # it's a PIL Image
            with tempfile.NamedTemporaryFile(suffix=".png") as f:
                image.save(f.name)
                ret = single_call("vlm", image=f.name, **call_kwargs)
    else:
        ret = single_call("chat", **call_kwargs)
    logger.debug(f'chat response: {ret}')
    if session_id:
        msg = ret['choices'][0]['message']
        assert isinstance(prompts, list)
        session_cache[session_id] = prompts + [(msg['role'], msg['content'])]
    return ret


@execution_wrapper(final_func=lambda x: x['choices'][0]['message']['content'])
def call_llm(prompts: str|list[Msg],
             max_tokens:int =1024,
             model:Optional[str] =None,
             session_id:str ='',
             use_cache=True,
             session_cache={},
             **kw) -> ResponseT:
    """Calls our local llm server for a chat completion.

    You can either pass in a single prompt (str), or a list of (role, text) tuples.

    By default, this has no memory of previous calls (i.e., you have to keep track of history
    yourself and pass in the full list of prompts each time). If you want to not have to remember
    past messages, you can pass in a `session_id` which will be used to keep track of the
    conversation history. In that case, any `prompts` you pass in, and each response from the
    server, will be appended to the history indexed by `session_cache[session_id]`.

    Uses the 'chat' model in DEFAULT_MODELS by default.

    Returns the raw json response (as a dict).
    """
    return call_llm_impl(prompts=prompts,
                         max_tokens=max_tokens,
                         model=model,
                         session_id=session_id,
                         use_cache=use_cache,
                         session_cache=session_cache)


@execution_wrapper(final_func=lambda x: x['choices'][0]['message']['content'])
def call_vlm(inputs: tuple[str, str|list[Msg]],
             max_tokens:int =1024,
             model:Optional[str] =None,
             session_id:str ='',
             use_cache=True,
             session_cache={},
             **kw) -> ResponseT:
    """Calls our local vlm server for a VLM chat completion.

    The inputs are a tuple of (image_url, prompts). You can either pass in a single prompt (str), or
    a list of (role, text) tuples. The image is entered into the first "user" message.

    By default, this has no memory of previous calls (i.e., you have to keep track of history
    yourself and pass in the full list of prompts each time). If you want to not have to remember
    past messages, you can pass in a `session_id` which will be used to keep track of the
    conversation history. In that case, any `prompts` you pass in, and each response from the
    server, will be appended to the history indexed by `session_cache[session_id]`.

    Uses the 'vlm' model in DEFAULT_MODELS by default.

    Returns the raw json response (as a dict).
    """
    image, prompts = inputs
    return call_llm_impl(prompts=prompts,
                         max_tokens=max_tokens,
                         image=image,
                         model=model,
                         session_id=session_id,
                         use_cache=use_cache,
                         session_cache=session_cache)


@execution_wrapper(final_func=lambda x: x['data'][0]['embedding'])
def embed_text(s: str, model: str='sentence', use_cache=True, **kw) -> ResponseT:
    """Embeds a string using the specified model.

    Models:
    - 'sentence': BAAI/bge-large-en-v1.5 [default]
    - 'clip': openai/clip-vit-large-patch14

    Returns the raw json response (as a dict).
    """
    return single_call("embeddings", input=s, model=model, use_cache=use_cache, **kw)

@execution_wrapper(final_func=lambda x: x['similarity'])
def strsim(input_: tuple[str, str], model='clip', use_cache=True, **kw) -> ResponseT:
    """Computes the similarity between two strings.

    Uses the 'openai/clip-vit-large-patch14' model by default.

    Returns the raw json response (as a dict).
    """
    a, b = input_
    return single_call("strsim", a=a, b=b, model=model, use_cache=use_cache, **kw)

@execution_wrapper(final_func=lambda x: x['data'][0]['embedding'])
def embed_image(img: str, model='image', use_cache=True, **kw) -> ResponseT:
    """Embeds an image (url or local path or loaded PIL image) using the specified model.

    Uses the 'openai/clip-vit-large-patch14' model by default.

    Returns the raw json response (as a dict).
    """
    # check if it's a url or path
    if isinstance(img, str):
        return single_call("image_embeddings", url=img, model=model, use_cache=use_cache, **kw)
    # else it's an image object, so write to disk temporarily and use that
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        img.save(f.name)
        ret = single_call("image_embeddings", url=f.name, model=model, use_cache=use_cache, **kw)

@execution_wrapper(final_func=lambda x: x['text'])
def get_text(url: str, use_cache=True, **kw) -> ResponseT:
    """Extracts text from a file (pdf using pdftotext, image using ocr, or text).

    Returns the raw json response (as a dict).
    """
    return single_call("get_text", url=url, use_cache=use_cache, **kw)

@execution_wrapper()
def transcribe_speech(audio: str|bytes,
                      model:Optional[str]=None,
                      use_cache=True,
                      **kw) -> ResponseT:
    """Runs speech transcription with given `audio` (local path, url, or bytes)."""
    return single_call("transcription",
                       url=audio,
                       model=model,
                       use_cache=use_cache,
                       **kw)


async def test_all():
    """Test all client functions"""
    # import library to make colored output
    try:
        import termcolor
    except ImportError:
        termcolor = None
    def myprint(prompt, *args, **kwargs):
        """Custom print that prepends timestamp, uses custom colors for time and for prompt""" 
        # print time with milliseconds
        t = time.strftime("%H:%M:%S", time.localtime())+f".{int(time.time()*1000)%1000:03d}"
        if termcolor:
            # print time in blue, prompt in green, rest in default
            print(termcolor.colored(f'\n{t}', 'blue'), termcolor.colored(prompt, 'green'), *args, **kwargs)
        else:
            print(f'\n{t} {prompt}', *args, **kwargs)

    s = "Once upon a time, "
    s2 = "The city of New York has "
    from PIL import Image
    img = Image.open("512px-Ichthyotitan_Size_Comparison.svg.png")
    inputs_by_func = {
        call_llm: ([s, s2], dict(model='llama3')),
        embed_text: ([s, s2], dict(model='clip')),
        strsim: ([("dog", "cat"), ("dog", "philosophy")], dict(model='sentence')),
        embed_image_url: (["https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Ichthyotitan_Size_Comparison.svg/512px-Ichthyotitan_Size_Comparison.svg.png", "512px-Ichthyotitan_Size_Comparison.svg.png"], None),
        embed_image: ([img, img], None),
        get_text: (['512px-Ichthyotitan_Size_Comparison.svg.png', 'https://www.adobe.com/support/products/enterprise/knowledgecenter/media/c4611_sample_explain.pdf'], None),
    }
    #for func in [call_llm, embed_text, strsim, embed_image_url, embed_image]:
    for func in [get_text]:
        inputs, kwargs = inputs_by_func[func]
        myprint(f'\nTesting function {func.__name__} with inputs {inputs}, kwargs {kwargs}')
        for mode in ['raw', 'final']:
            try:
                func.mode = mode
            except ValueError:
                continue
            myprint(f'Mode: {mode}\n')
            myprint('  Single call, default args:', func.single(inputs[0]))
            if kwargs:
                myprint(f'  Single call, {kwargs}:', func.single(inputs[0], **kwargs))
            myprint('  Batch call, default args:', func.batch(inputs))
            if kwargs:
                myprint(f'  Batch call, {kwargs}:', func.batch(inputs, **kwargs))
            myprint('  Single async call, default args:', await func.single_async(inputs[0]))
            if kwargs:
                myprint(f'  Single async call, {kwargs}:', await func.single_async(inputs[0], **kwargs))
            myprint('  Batch async call, default args:', await func.batch_async(inputs))
            if kwargs:
                myprint(f'  Batch async call, {kwargs}:', await func.batch_async(inputs, **kwargs))
            f = func.single_future(inputs[0])
            myprint('  Single future call, default args:', f)
            myprint('    Result', f.result())
            if kwargs:
                f = func.single_future(inputs[0], **kwargs)
                myprint(f'  Single future call, {kwargs}:', f)
                myprint('    Result', f.result())
            fs = func.batch_futures(inputs)
            myprint('  Batch future call, default args:', fs)
            myprint('    Results', [f.result() for f in fs])
            if kwargs:
                fs = func.batch_futures(inputs, **kwargs)
                myprint(f'  Batch future call, {kwargs}:', fs)
                myprint('    Results', [f.result() for f in fs])

def quick_test():
    # setup logging to include filename, function name, and line number as well
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
    test = 'emb'
    if test == 'llm1':
        print(call_llm.single([('system', 'you are a very terse answering bot'), ('user', "What is the capital of italy?")]))
    elif test == 'llm2':
        kwargs = dict(model='', session_id='a')
        kwargs['model'] = 'gpt-4o-mini'
        print(call_llm.single('describe light', **kwargs))
        print(call_llm.single('summarize that in one sentence', **kwargs))
        print(call_llm.single('summarize it in 3 sentences', **kwargs))
    elif test == 'vlm1':
        image = 'https://images.unsplash.com/photo-1582538885592-e70a5d7ab3d3?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1770&q=80'
        prompt = 'Can you describe this image?'
        print(call_vlm.single((image, prompt)))
        #print(call_vlm.single((image, prompt), model="accounts/fireworks/models/llama-v3p2-90b-vision-instruct"))
    elif test == 'vlm2':
        image = './simple-sales-invoice-modern-simple-1-1-f54b9a4c7ad8.webp'
        from PIL import Image
        image = Image.open(image)
        prompt = 'For the following image, return the following in JSON format: title of document, general category of document, detailed category of document, date, and a list of key-value pairs of other data contained within it. Give no preamble or other text, just the JSON object'
        for model in [
            'vlm',
            "meta-llama/Llama-Vision-Free",
            "accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
            "accounts/fireworks/models/phi-3-vision-128k-instruct",
            "accounts/fireworks/models/qwen2-vl-72b-instruct",
            ]:
            print(f'trying model {model}:', call_vlm.single((image, prompt), model=model))
    elif test == 'emb':
        s = 'hello'
        for model in 'e5 ada clip st'.split():
            ret = embed_text.single(s, model=model)
            print(f'Embedding for model {model} with {len(ret)} dims: {ret[:10]}')
    elif test == 'speech':
        dir = '/home/neeraj/dp/podcasts/audio/Chapo Trap House/'
        fname = '2022-09-28 - 666 - Chapo Goes To Hell (9-27-22).mp3'
        print(f'testing speech transcription for {fname}')
        ret = transcribe_speech.single(join(dir, fname))
        print(json.dumps(ret, indent=2))


if __name__ == '__main__':
    quick_test(); sys.exit();
    # check that we're not importing torch or numpy, etc
    disallowed = ['torch', 'numpy', 'transformers', 'PIL']
    for key in sys.modules.keys():
        if any([dis in key for dis in disallowed]):
            print(f"Error: {key} is imported.")
            sys.exit(1)
    # run test all in async mode
    asyncio.run(test_all())
