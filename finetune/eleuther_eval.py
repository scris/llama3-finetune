# v0.3.1, logger.info => print to get results

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.metadata
import sys
import time

from typing import Dict, List, Tuple, Union

import PIL

import torch
from omegaconf import DictConfig

from torchtune import config, training, utils
from torchtune.data import (
    format_content_with_images,
    left_pad_sequence,
    Message,
    padded_collate_tiled_images_and_mask,
)
from torchtune.generation import generate, sample
from torchtune.modules import TransformerDecoder
from torchtune.modules.model_fusion import DeepFusionModel
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform
from torchtune.recipe_interfaces import EvalRecipeInterface

try:
    import lm_eval
except ImportError:
    print(
        "You must install the EleutherAI Eval Harness to run this recipe. "
        "Please install with `pip install lm_eval>=0.4.2`"
    )
    sys.exit(1)

lm_eval_version = importlib.metadata.version("lm_eval")
if not lm_eval_version >= "0.4.2":
    print(
        "You must install the EleutherAI Eval Harness >= v0.4.2 to run this recipe. "
        "Please install with `pip install lm_eval>=0.4.2`"
    )
    sys.exit(1)

from lm_eval.evaluator import evaluate, get_task_list

# User doesn't have to have nightlies installed, they just won't be able
# to use the multimodal model
try:
    from lm_eval.models.hf_vlms import HFMultimodalLM
except ImportError as e:
    # Create a dummy class to avoid having to import the HF models
    # TODO (@joecummings): Remove this once v0.4.5 patch is released
    class HFMultimodalLM:
        def __init__(self, *args, **kwargs):
            pass


from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import get_task_dict, TaskManager


class _VLMEvalWrapper(HFMultimodalLM):
    """An EvalWrapper for EleutherAI's eval harness based on gpt-fast's
    EvalWrapper: https://github.com/pytorch-labs/gpt-fast/blob/main/eval.py.

    Note:
        This is ONLY for vision-language models.

    Args:
        model (DeepFusionModel): The VLM to evaluate.
        transform (Transform): The transform (tokenizer) to use for preprocessing.
        device (torch.device): The device to use.
        max_seq_length (int): The maximum sequence length.
        batch_size (int): The batch size.
        dtype (torch.dtype): dtype for the model caches during generation.
        enable_kv_cache (bool): Whether to enable KV cache for generation.
        image_tag (str): The string to use for the image token. Default is "<image>", which
            is the default used by the MMMU dataset.
        max_images_per_sample (int): The maximum number of images per sample. Defaults to
            the max number of images in MMMU.
    """

    def __init__(
        self,
        model: DeepFusionModel,
        transform: Transform,
        *,
        device: torch.device,
        max_seq_length: int = 4096,
        batch_size: int = 8,
        dtype: torch.dtype = torch.bfloat16,
        enable_kv_cache: bool = True,
        # TODO (@joecummings): Update these defaults once more multimodal
        # tasks are added to the eval harness
        image_tag: str = "<image>",
        max_images_per_sample: int = 7,
    ):
        self._model = model
        self._transform = transform
        self._device = device
        self._max_seq_length = max_seq_length
        self._batch_size = batch_size
        self._dtype = dtype
        # Defaulting KV cache to True for multimodal
        self._enable_kv_cache = True
        self._image_tag = image_tag
        self._max_images_per_sample = max_images_per_sample

    @property
    def model(self):
        # Not actually changing the dtype here, just adding it as a
        # property on the model
        self._model.dtype = self._dtype
        return self._model

    @property
    def model_transform(self):
        return self._transform

    @property
    def device(self):
        return self._device

    @property
    def cache_hook(self):
        # Dummy class to appease the Harness
        class DummyCacheHook:
            def __init__(self):
                self.add_partial = lambda x, y, z: True

        return DummyCacheHook()

    @property
    def rank(self):
        # Hardcoded for now b/c we only support single GPU eval
        return 0

    @property
    def world_size(self):
        # Hardcoded for now b/c we only support single GPU eval
        return 1

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def eos_token_id(self):
        return self._transform.tokenizer.eos_id

    @property
    def eot_token_id(self):
        return self._transform.tokenizer.eot_id

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def truncation(self):
        return True

    def tok_encode(self, string, **kwargs) -> List[int]:
        # This is only used to get a number of tokens for use in sorting samples in dataset
        # These values will not actually be used for eval
        return self._transform.tokenizer.encode(string, add_bos=False, add_eos=False)

    def tok_decode(self, tokens, skip_special_tokens=True) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        return self._transform.tokenizer.decode(
            tokens, skip_special_tokens=skip_special_tokens
        )

    def tok_batch_multimodal_encode(
        self,
        all_texts: List[str],
        all_images: List[List[PIL.Image.Image]],
        *args,
        **kwargs,
    ):
        # Eleuther already parses out the text and images, so we just need to get
        # it into a Message format for our tokenizer
        all_encoded_messages = []

        for text, images in zip(all_texts, all_images):
            # Ensure images are all RGB
            proper_images = []
            for image in images:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                proper_images.append(image)

            # Construct the messages
            messages = []
            content = format_content_with_images(
                text, image_tag=self._image_tag, images=proper_images
            )
            messages.append(Message(role="user", content=content))
            messages.append(Message(role="assistant", content=""))

            # Transform the messages
            tok_batch = self.model_transform({"messages": messages}, inference=True)
            all_encoded_messages.append(tok_batch)

        # Pad the encoded messages
        tok_batch = padded_collate_tiled_images_and_mask(
            all_encoded_messages,
            pad_direction="left",
            pad_max_images=self._max_images_per_sample,
        )
        utils.batch_to_device(tok_batch, self.device)

        # Convert the batch to the format expected by the HF
        tok_batch["input_ids"] = tok_batch.pop("tokens")
        return tok_batch

    @torch.inference_mode()
    def _model_multimodal_generate(
        self,
        batch: Dict[str, torch.Tensor],
        max_length: int,
        stop: List[str],
        **generation_kwargs,
    ):
        # 1. Validate inputs
        prompt = batch.pop("input_ids")
        bsz, seq_len = prompt.shape

        temperature = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", False)
        if do_sample or temperature != 0.0:
            raise RuntimeError(
                "Any decoding strategy other than greedy is not supported."
            )

        if bsz > 1:
            raise ValueError(
                f"Got a batch size of '{bsz}'. Batch size > 1 is not yet supported for "
                "multimodal generation."
            )

        # 2. Setup KV cache and masks for bsz 1
        with self.device:
            if self.model.caches_are_enabled():
                self.model.reset_caches()
            else:
                self.model.setup_caches(
                    batch_size=1,
                    dtype=self._dtype,
                    encoder_max_seq_len=self.model_transform.image_seq_len
                    * self._max_images_per_sample,
                    decoder_max_seq_len=self.max_length,
                )
            causal_mask = torch.tril(
                torch.ones(
                    size=(self.max_length, self.max_length),
                    dtype=torch.bool,
                )
            )
            input_pos = torch.arange(self.max_length)

        batch["input_pos"] = input_pos[None, :seq_len]
        batch["mask"] = causal_mask[None, :seq_len]

        # 3. Prefill step
        generated_tokens = []
        logits = self.model(prompt, **batch)[:, -1]
        token = sample(logits, temperature=0.0, top_k=None)
        generated_tokens.append(token.item())

        cache_mask = batch["encoder_mask"][:, -1:]

        # 4. Continue generating
        for _ in range(max_length):
            if token.item() in self.model_transform.stop_tokens:
                break
            logits = self.model(
                token,
                mask=causal_mask[None, seq_len, None, :],
                encoder_input=None,
                encoder_mask=cache_mask,
                input_pos=input_pos[None, seq_len],
            )[:, -1]
            token = sample(logits, temperature=0.0, top_k=None)
            generated_tokens.append(token.item())
            seq_len += 1

        # 5. Return generated tokens
        return torch.tensor(generated_tokens, dtype=torch.int32).unsqueeze(0)


class _LLMEvalWrapper(HFLM):
    """An EvalWrapper for EleutherAI's eval harness based on gpt-fast's
    EvalWrapper: https://github.com/pytorch-labs/gpt-fast/blob/main/eval.py.

    Note:
        This is for text-only decoder models.

    Args:
        model (TransformerDecoder): The model to evaluate.
        tokenizer (ModelTokenizer): Tokenizer associated with the model being evaluated.
            This should be the same tokenizer used when fine-tuning the model.
        device (torch.device): The device to use.
        max_seq_length (int): The maximum sequence length to use.
        batch_size (int): The batch size per GPU to use.
        dtype (torch.dtype): dtype for the model caches during generation.
        enable_kv_cache (bool): Whether to enable KV cache for generation.
    """

    def __init__(
        self,
        model: TransformerDecoder,
        tokenizer: ModelTokenizer,
        *,
        device: torch.device,
        max_seq_length: int = 4096,
        batch_size: int = 8,
        dtype: torch.dtype = torch.float32,
        enable_kv_cache: bool = True,
    ):
        # TODO (@joecummings): Remove this init function so we don't load in extraneous stuff
        super().__init__(pretrained="gpt2", device=str(device))
        self._model = model
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._batch_size = batch_size
        self._dtype = dtype
        self._enable_kv_cache = enable_kv_cache

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_id

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    @property
    def enable_kv_cache(self):
        return self._enable_kv_cache

    def tok_encode(self, text: str, **kwargs) -> List[int]:
        # Note on add_bos flag: setting to False as this gives better results, for example
        # +1% on truthfulqa_mc2 with a LoRA finetune. lit-gpt also sets this to False,
        # see https://github.com/Lightning-AI/lit-gpt/blob/main/eval/lm_eval_harness.py#L66,
        # though notably fast-gpt does the opposite
        # https://github.com/pytorch-labs/gpt-fast/blob/main/eval.py#L123.
        return self._tokenizer.encode(text=text, add_bos=False, add_eos=False)

    def tok_batch_encode(
        self, text: List[str], **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokenized_text = [self.tok_encode(x) for x in text]

        # pad left
        x = left_pad_sequence(
            [torch.tensor(x) for x in tokenized_text],
            batch_first=True,
            padding_value=self._tokenizer.pad_id,
        )

        return x, torch.ones_like(x)  # return 'mask' b/c it's expected by the harness

    def tok_decode(self, tokens: Union[List[int], int], **kwargs) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        return self._tokenizer.decode(tokens)

    def _model_call(self, inps: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._model(inps)

    @torch.inference_mode()
    def _model_generate(
        self, context: torch.Tensor, **generation_kwargs
    ) -> torch.Tensor:
        bsz, seq_len = context.shape

        temperature = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", False)
        if do_sample or temperature != 0.0:
            raise RuntimeError(
                "Any decoding strategy other than greedy is not supported."
            )

        # Setup KV caches OR reset them if they're already set up
        if self.enable_kv_cache:
            if self.model.caches_are_enabled():
                self.model.reset_caches()
            else:
                with self.device:
                    self.model.setup_caches(
                        batch_size=self.batch_size,
                        dtype=self._dtype,
                        decoder_max_seq_len=self.max_length,
                    )

        # if we've recieved fewer than self._batch_size samples in the current
        # batch we need to pad the batch out. here we're padding the end of the
        # current batch to the correct length. this is because when we use static
        # KV-caches, the model will expect a fixed batch size for all samples.
        maybe_padded_context = torch.nn.functional.pad(
            context,
            (0, 0, 0, self._batch_size - bsz),
            value=self._tokenizer.eos_id,  # pad with one of the tokenizer's stop tokens so generation can stop early
        )

        toks, _ = generate(
            self.model,
            maybe_padded_context,
            max_generated_tokens=self.max_gen_toks,
            temperature=temperature,
            top_k=None,
            stop_tokens=self._tokenizer.stop_tokens,
        )
        return toks[:bsz]


class EleutherEvalRecipe(EvalRecipeInterface):
    """
    This recipe runs evaluation on a trained model using EleutherAI's eval harness.
    This assumes the user has the EleutherAI eval harness installed. See
    https://github.com/EleutherAI/lm-evaluation-harness for more details.

    Features:
        - Single GPU evaluation. Multi-GPU evaluation is currently not supported.
        - Quantization (for text-only models) is supported.
        - Any task from the EleutherAI eval harness

    We recommend launching evaluation using the tune CLI::

        tune run eleuther_eval --config eleuther_evaluation \
            tasks=["truthfulqa_mc2","hellaswag"] \
            limit=50 \
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.device = utils.get_device(device=cfg.device)
        self.dtype = training.get_dtype(dtype=cfg.dtype, device=self.device)
        training.set_seed(seed=cfg.seed)

        # Eval specific variables
        self.limit = cfg.limit
        self.tasks = list(cfg.tasks)
        self.batch_size = cfg.batch_size
        self.enable_kv_cache = cfg.get("enable_kv_cache", True)
        self.include_path = cfg.get("include_path", None)

    def setup(self, cfg: DictConfig) -> None:
        # Initialize quantizer and quantization mode
        quantizer = config.instantiate(cfg.quantizer)
        quantization_mode = training.get_quantizer_mode(quantizer)

        # Load checkpoint
        checkpointer = config.instantiate(cfg.checkpointer)
        if quantization_mode is None:
            ckpt_dict = checkpointer.load_checkpoint()
        else:
            # weights_only needs to be False when loading a quantized model
            # currently loading a quantized model is only supported with the
            # FullModelTorchTuneCheckpointer
            ckpt_dict = checkpointer.load_checkpoint(weights_only=False)

        # Initialize model
        with training.set_default_dtype(self.dtype), self.device:
            model = config.instantiate(cfg.model)

        # Quantize model if requested
        if quantization_mode is not None:
            model = quantizer.quantize(model)
            model = model.to(device=self.device, dtype=self.dtype)
            for k, v in model_state_dict.items():
                model_state_dict[k] = v.to(self._device)
            model.load_state_dict(model_state_dict, assign=True)

        # Load model weights into initialized model
        model.load_state_dict(ckpt_dict[training.MODEL_KEY])
        print(f"Model is initialized with precision {self.dtype}.")

        # Put model in eval mode.
        # Note: This will not disable the dropout applied in SDPA,
        # see https://github.com/pytorch/pytorch/issues/124464
        model.eval()

        # Initialize tokenizer/transform
        model_transform = config.instantiate(cfg.tokenizer)
        max_seq_len = (
            model_transform.max_seq_len
            if model_transform.max_seq_len is not None
            else 4096  # default max_seq_len to 4096
        )

        # Finally, we setup the actual EvalWrapper class
        if isinstance(model, DeepFusionModel):
            eleuther_model_wrapper = _VLMEvalWrapper
            if not self.enable_kv_cache:
                print(
                    "Received enable_kv_cache=False, but KV cache is required for running "
                    "multimodal generation in a timely manner. Setting enable_kv_cache=True."
                )
        elif isinstance(model, TransformerDecoder):
            eleuther_model_wrapper = _LLMEvalWrapper
        self.eleuther_model_wrapper = eleuther_model_wrapper(
            model,
            model_transform,
            device=self.device,
            max_seq_length=max_seq_len,
            batch_size=self.batch_size,
            dtype=self.dtype,
            enable_kv_cache=self.enable_kv_cache,
        )

    def evaluate(self) -> None:
        # Initialize tasks for the harness
        task_manager = TaskManager(include_path=self.include_path)
        task_dict = get_task_dict(self.tasks, task_manager)
        task_types = set([t.task.OUTPUT_TYPE for t in get_task_list(task_dict)])
        if len(task_types) > 1 and "generate_until" in task_types:
            raise RuntimeError(
                "Evaluating on multiple task types where any one task involves "
                "generation is currently not supported. See the issue below for more info: "
                "https://github.com/pytorch/torchtune/issues/1621"
            )

        # Run evaluation
        t0 = time.time()
        print(f"Running evaluation on the following tasks: {self.tasks}")
        output = evaluate(
            self.eleuther_model_wrapper,
            task_dict,
            limit=self.limit,
        )
        t1 = time.time() - t0

        # Log metrics
        print(f"Eval completed in {t1:.02f} seconds.")
        print(
            f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB"
        )
        formatted_output = lm_eval.utils.make_table(output)
        print(f"\n\n{formatted_output}\n")


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """Entry point for the recipe."""
    config.log_config(recipe_name="EleutherEvalRecipe", cfg=cfg)
    recipe = EleutherEvalRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.evaluate()


if __name__ == "__main__":
    sys.exit(recipe_main())