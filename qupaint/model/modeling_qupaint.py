# --------------------------------------------------------
# QuPAINT: Physics-Aware Instruction Tuning Approach to
#          Quantum Material Discovery  (CVPR 2026, Findings)
#
# Inference-only model definition.
# Built on InternVL (c) 2024 OpenGVLab, MIT License.
# --------------------------------------------------------

import warnings
from typing import List, Optional, Tuple, Union

import torch
import transformers
from torch import nn
from transformers import AutoModelForCausalLM, CLIPVisionModel, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_qupaint import QuPAINTConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel, has_flash_attn

logger = logging.get_logger(__name__)

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def version_cmp(v1, v2, op="eq"):
    import operator

    from packaging import version

    return getattr(operator, op)(version.parse(v1), version.parse(v2))


class QuPAINTModel(PreTrainedModel):
    """QuPAINT vision-language model: InternViT encoder + MLP projector + Qwen3 LLM."""

    config_class = QuPAINTConfig
    main_input_name = "pixel_values"
    base_model_prefix = "language_model"
    _supports_flash_attn_2 = True
    _no_split_modules = ["InternVisionEncoderLayer", "Qwen3DecoderLayer"]

    def __init__(
        self,
        config: QuPAINTConfig,
        vision_model=None,
        language_model=None,
        use_flash_attn=True,
    ):
        super().__init__(config)

        assert version_cmp(transformers.__version__, "4.37.0", "ge")
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        if isinstance(image_size, (list, tuple)):
            assert image_size[0] == image_size[1], "Only square images are supported."
            image_size = image_size[0]
        if isinstance(patch_size, (list, tuple)):
            assert patch_size[0] == patch_size[1], "Only square patches are supported."
            patch_size = patch_size[0]

        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
        )

        use_flash_attn = use_flash_attn and has_flash_attn
        config.vision_config.use_flash_attn = bool(use_flash_attn)

        if vision_model is not None:
            self.vision_model = vision_model
        elif config.vision_config.architectures[0] == "InternVisionModel":
            self.vision_model = InternVisionModel(config.vision_config)
        elif config.vision_config.architectures[0] == "CLIPVisionModel":
            self.vision_model = CLIPVisionModel(config.vision_config)
        else:
            raise NotImplementedError(
                f"{config.vision_config.architectures[0]} is not implemented."
            )

        if language_model is not None:
            self.language_model = language_model
        else:
            self.language_model = AutoModelForCausalLM.from_config(config.llm_config)

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        # Auxiliary head carried by the released checkpoint. It is not used for
        # inference — flake boxes are decoded from the generated text — but it is
        # declared here so the checkpoint loads without missing/unexpected keys.
        self.bbox_head = nn.Linear(llm_hidden_size, 4)

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

    # ------------------------------------------------------------------ #
    # Visual feature extraction
    # ------------------------------------------------------------------ #
    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.ps_version == "v1":
            warnings.warn(
                "In ps_version 'v1' the height and width are not swapped back, "
                "which results in a transposed image."
            )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]  # drop the [CLS] token

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return self.mlp1(vit_embeds)

    def _merge_visual_embeds(self, input_ids, vit_embeds):
        """Scatter projected visual tokens into the <IMG_CONTEXT> positions."""
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        selected = input_ids.reshape(B * N) == self.img_context_token_id
        if selected.sum() == 0:
            raise ValueError(
                "No <IMG_CONTEXT> token found in the prompt. Did you include '<image>' "
                "in the question and set `img_context_token_id`?"
            )
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.dtype).to(
            input_embeds.device
        )
        return input_embeds.reshape(B, N, C)

    # ------------------------------------------------------------------ #
    # Forward / generation
    # ------------------------------------------------------------------ #
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Inference forward pass. Training losses are not part of this release."""
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is not None:
            vit_embeds = self.extract_feature(pixel_values)
            if image_flags is not None:
                vit_embeds = vit_embeds[image_flags.squeeze(-1) == 1]
            inputs_embeds = self._merge_visual_embeds(input_ids, vit_embeds)
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        if not return_dict:
            return outputs
        return CausalLMOutputWithPast(
            loss=None,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        assert self.img_context_token_id is not None, (
            "`img_context_token_id` is unset; call `chat()` or set it yourself."
        )
        if pixel_values is not None:
            vit_embeds = (
                visual_features
                if visual_features is not None
                else self.extract_feature(pixel_values)
            )
            inputs_embeds = self._merge_visual_embeds(input_ids, vit_embeds)
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        pixel_values,
        question,
        generation_config,
        history=None,
        return_history=False,
        num_patches_list=None,
        IMG_START_TOKEN=IMG_START_TOKEN,
        IMG_END_TOKEN=IMG_END_TOKEN,
        IMG_CONTEXT_TOKEN=IMG_CONTEXT_TOKEN,
        verbose=False,
    ):
        """Single-image chat. `pixel_values` comes from `qupaint.preprocess.load_image`."""
        if history is None and pixel_values is not None and "<image>" not in question:
            question = "<image>\n" + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        self.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        sep = template.sep.strip() if template.sep2 is None else template.sep2.strip()
        eos_token_id = tokenizer.convert_tokens_to_ids(sep)

        history = [] if history is None else list(history)
        for old_question, old_answer in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        for num_patches in num_patches_list:
            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)

        generation_config = dict(generation_config)
        generation_config["eos_token_id"] = eos_token_id

        outputs = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
        response = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
        response = response.split(sep)[0].strip()

        history.append((question, response))
        if verbose:
            printable = query.replace(IMG_CONTEXT_TOKEN, "").replace(
                f"{IMG_START_TOKEN}{IMG_END_TOKEN}", "<image>"
            )
            print(printable + response)
        return (response, history) if return_history else response

    # ------------------------------------------------------------------ #
    # Embedding plumbing
    # ------------------------------------------------------------------ #
    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, value):
        return self.language_model.set_output_embeddings(value)


# Name used by the original research codebase; kept so older configs still load.
CVIUVLChatModel = QuPAINTModel
