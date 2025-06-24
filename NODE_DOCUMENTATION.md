# ComfyUI Advanced Guidance Nodes: Documentation

This document provides detailed information about various advanced guidance nodes available for ComfyUI, including their parameters and practical applications in image generation. These nodes offer sophisticated ways to control and refine the output of diffusion models.

## General Concepts

Many of these nodes share common parameters that control their behavior:

*   **`model`**: The input diffusion model that will be patched or affected by the node.
*   **`scale`**: Generally controls the strength or intensity of the guidance effect. Higher values mean a stronger effect.
*   **`sigma_start`**: The diffusion timestep (sigma value) at which the node's effect begins. Higher sigma values correspond to earlier, noisier stages of diffusion. A value of `-1.0` (or any negative) usually means the effect is active from the very beginning (highest sigma/noise).
*   **`sigma_end`**: The diffusion timestep (sigma value) at which the node's effect stops. Lower sigma values correspond to later, less noisy stages. A value of `-1.0` (or any negative) usually means the effect is active until the very end (lowest sigma/noise). If `sigma_end` is a higher sigma value than `sigma_start`, the effect will likely not be active.
*   **`unet_block_list`**: A comma-separated string allowing precise selection of U-Net blocks where the guidance should be applied. This offers fine-grained control for advanced users. The specific format can vary (e.g., "middle.0,output.2,output.3" or "d2.2-9,d3"). If empty, behavior often defaults to simpler `unet_block` and `unet_block_id` parameters or applies to all relevant blocks.
*   **`rescale` & `rescale_mode`**: Parameters used in some guidance techniques (like PAG, SEG, TPG) to adjust how the guidance signal is combined with the main denoising prediction. This can help stabilize the output and prevent artifacts.
    *   `rescale`: A float value (often 0.0 to 1.0) controlling the amount of rescaling.
    *   `rescale_mode`: Can be "full", "partial", or "snf" (Signal-to-Noise First), each offering a different strategy for integrating the guidance.

---

## Node: Normalized Attention Guidance (NAG)

*   **Category:** `model_patches/unet`
*   **Description:**
    *   An additional way to apply negative prompts to the image by modifying cross-attention (attn2) layers.
    *   It's compatible with CFG, PAG, and other guidances, and can be used with guidance- and step-distilled models as well.
    *   It's also compatible with other attn2 replacers (such as `IPAdapter`) - but make sure to place NAG node **after** other model patches!
*   **Parameters:**
    *   **Required:**
        *   `model`:
            *   Type: `IO.MODEL`
            *   Tooltip/Description: The diffusion model. If you are using any other attn2 replacer (such as `IPAdapter`), you should place this node after it.
            *   Practical Use: This is the input model that will be patched with Normalized Attention Guidance. It's crucial to pipe in the model you intend to use for generation. If other attention modifications are used (like IPAdapter), NAG should generally be applied last to ensure it can correctly access and modify the attention mechanism.
        *   `negative`:
            *   Type: `IO.CONDITIONING`
            *   Tooltip/Description: Negative conditioning: either the one you use for CFG or a completely different one.
            *   Practical Use: This conditioning defines what concepts the model should steer *away* from. It can be the same negative prompt used for standard CFG or a specialized one tailored for NAG's effect. For example, if generating a "beautiful landscape," a negative prompt here like "ugly, deformed, pollution" would be actively pushed against by NAG.
        *   `scale`:
            *   Type: `IO.FLOAT`
            *   Default: `2.0`, Min: `0.0`, Max: `100.0`, Step: `0.1`, Round: `0.01`
            *   Tooltip/Description: Scale of NAG, does nothing when `tau=0`.
            *   Practical Use: Controls the strength of the NAG effect. Higher values mean a stronger push away from the concepts defined in the `negative` conditioning. If `tau` is 0, this parameter has no effect. It's the primary knob for tuning how aggressively NAG influences the image.
        *   `tau`:
            *   Type: `IO.FLOAT`
            *   Default: `2.5`, Min: `0.0`, Max: `100.0`, Step: `0.1`, Round: `0.01`
            *   Tooltip/Description: Normalization threshold, larger value should increase `scale` impact.
            *   Practical Use: `tau` is a threshold for the normalization process in NAG. It determines how much the attention scores are allowed to deviate. A higher `tau` generally allows the `scale` parameter to have a more pronounced effect, potentially leading to stronger guidance but also risking overly aggressive changes if not balanced with `scale`. If `tau` is 0, NAG is effectively disabled.
        *   `alpha`:
            *   Type: `IO.FLOAT`
            *   Default: `0.5`, Min: `0.0`, Max: `1.0`, Step: `0.001`, Round: `0.001`
            *   Tooltip/Description: Linear interpolation between original (at `alpha=0`) and NAG (at `alpha=1`) results.
            *   Practical Use: This parameter blends the NAG-modified attention output with the original attention output. An `alpha` of 0 means no NAG effect is applied (original attention is used), while an `alpha` of 1 means the full NAG effect is applied. Values in between allow for a partial application, which can be useful for fine-tuning the intensity of the guidance and preventing artifacts that might arise from too strong a NAG signal.
        *   `sigma_start`:
            *   Type: `IO.FLOAT`
            *   Default: `-1.0`, Min: `-1.0`, Max: `10000.0`, Step: `0.01`
            *   Tooltip/Description: Specifies the diffusion timestep (sigma value) at which NAG starts being active. A value of -1.0 (or any negative) means it's active from the very beginning (inf sigma).
            *   Practical Use: Controls when NAG begins to apply its guidance during the diffusion process. Higher sigma values correspond to earlier, noisier stages of diffusion. Setting this allows NAG to influence the image from a specific point onwards.
        *   `sigma_end`:
            *   Type: `IO.FLOAT`
            *   Default: `-1.0`, Min: `-1.0`, Max: `10000.0`, Step: `0.01`
            *   Tooltip/Description: Specifies the diffusion timestep (sigma value) at which NAG stops being active. A value of -1.0 (or any negative) means it's active until the very end (0 sigma).
            *   Practical Use: Controls when NAG stops applying its guidance. Lower sigma values correspond to later, less noisy stages.
    *   **Optional:**
        *   `unet_block_list`:
            *   Type: `IO.STRING`
            *   Default: `""` (empty string)
            *   Tooltip/Description: Comma-separated blocks to which NAG is being applied to. When the list is empty, NAG is being applied to all block. Read README from sd-perturbed-attention for more details.
            *   Practical Use: Allows fine-grained control over which parts of the U-Net model's cross-attention (attn2) layers NAG affects. An empty string applies it to all. Example: "middle.0,output.2,output.3,output.4,output.5".

---

## Node: Perturbed Attention Guidance (PAG)

*   **Category:** `model_patches/unet`
*   **Description:** This node applies Perturbed Attention Guidance, a technique that modifies the self-attention mechanism (attn1) in specified U-Net blocks to enhance or alter image features, often leading to increased detail or different interpretations of the prompt. It works alongside CFG.
*   **Parameters:**
    *   **Required:**
        *   `model`:
            *   Type: `MODEL`
            *   Practical Use: The input diffusion model to be patched with PAG.
        *   `scale`:
            *   Type: `FLOAT`
            *   Default: `3.0`, Min: `0.0`, Max: `100.0`, Step: `0.1`, Round: `0.01`
            *   Practical Use: Strength of the PAG effect. Higher values lead to a stronger perturbation of the self-attention, which can increase details or make features more prominent.
        *   `adaptive_scale`:
            *   Type: `FLOAT`
            *   Default: `0.0`, Min: `0.0`, Max: `1.0`, Step: `0.001`, Round: `0.0001`
            *   Practical Use: Modulates the `scale` based on the diffusion timestep. A non-zero value will reduce the `scale` as the timestep `t` increases (i.e., less perturbation in earlier, noisier steps).
        *   `unet_block`:
            *   Type: `COMBO` (Options: `["input", "middle", "output"]`)
            *   Default: `"middle"`
            *   Practical Use: Specifies the U-Net block type (input, middle, or output) where PAG will be applied if `unet_block_list` is empty.
        *   `unet_block_id`:
            *   Type: `INT`
            *   Default: `0`
            *   Practical Use: Specifies the index of the U-Net block within the `unet_block` type (e.g., `middle.0`) if `unet_block_list` is empty.
        *   `sigma_start`:
            *   Type: `FLOAT`
            *   Default: `-1.0`, Min: `-1.0`, Max: `10000.0`, Step: `0.01`
            *   Practical Use: The noise level (sigma) at which PAG starts being active. `-1.0` means active from the beginning.
        *   `sigma_end`:
            *   Type: `FLOAT`
            *   Default: `-1.0`, Min: `-1.0`, Max: `10000.0`, Step: `0.01`
            *   Practical Use: The noise level (sigma) at which PAG stops being active. `-1.0` means active until the end.
        *   `rescale`:
            *   Type: `FLOAT`
            *   Default: `0.0`, Min: `0.0`, Max: `1.0`, Step: `0.01`
            *   Practical Use: Factor for rescaling the guidance. Helps normalize the PAG effect.
        *   `rescale_mode`:
            *   Type: `COMBO` (Options: `["full", "partial", "snf"]`)
            *   Default: `"full"`
            *   Practical Use: Mode for applying rescale (`full`, `partial`, `snf`).
    *   **Optional:**
        *   `unet_block_list`:
            *   Type: `STRING`
            *   Default: `""`
            *   Practical Use: Comma-separated list of specific U-Net self-attention blocks (e.g., "middle.0.attn1,output.2.attn1") to apply PAG to. Overrides `unet_block` and `unet_block_id`.

---

## Node: Smoothed Energy Guidance Advanced (SEG)

*   **Category:** `model_patches/unet`
*   **Description:** This node applies Smoothed Energy Guidance by modifying self-attention (attn1) layers. It involves blurring the attention maps, which can lead to smoother results, emphasize larger structures, or create stylized effects.
*   **Parameters:**
    *   **Required:**
        *   `model`:
            *   Type: `MODEL`
            *   Practical Use: The input diffusion model to be patched with SEG.
        *   `scale`:
            *   Type: `FLOAT`
            *   Default: `3.0`, Min: `0.0`, Max: `100.0`, Step: `0.1`, Round: `0.01`
            *   Practical Use: Strength of the SEG effect. Higher values increase the influence of the smoothed attention.
        *   `blur_sigma`:
            *   Type: `FLOAT`
            *   Default: `-1.0`, Min: `-1.0`, Max: `9999.0`, Step: `0.01`, Round: `0.001`
            *   Practical Use: Sigma for Gaussian blur applied to attention maps. Higher value means more blur. `-1.0` might imply a default or no blur.
        *   `unet_block`:
            *   Type: `COMBO` (Options: `["input", "middle", "output"]`)
            *   Default: `"middle"`
            *   Practical Use: U-Net block type if `unet_block_list` is empty.
        *   `unet_block_id`:
            *   Type: `INT`
            *   Default: `0`
            *   Practical Use: U-Net block index if `unet_block_list` is empty.
        *   `sigma_start`:
            *   Type: `FLOAT`
            *   Default: `-1.0`, Min: `-1.0`, Max: `10000.0`, Step: `0.01`
            *   Practical Use: Sigma level to start applying SEG.
        *   `sigma_end`:
            *   Type: `FLOAT`
            *   Default: `-1.0`, Min: `-1.0`, Max: `10000.0`, Step: `0.01`
            *   Practical Use: Sigma level to stop applying SEG.
        *   `rescale`:
            *   Type: `FLOAT`
            *   Default: `0.0`, Min: `0.0`, Max: `1.0`, Step: `0.01`
            *   Practical Use: Rescaling factor for SEG guidance.
        *   `rescale_mode`:
            *   Type: `COMBO` (Options: `["full", "partial", "snf"]`)
            *   Default: `"full"`
            *   Practical Use: Mode for applying rescale.
    *   **Optional:**
        *   `unet_block_list`:
            *   Type: `STRING`
            *   Default: `""`
            *   Practical Use: Comma-separated list of specific U-Net self-attention blocks for SEG. Overrides `unet_block` and `unet_block_id`.

---

## Node: Sliding Window Guidance Advanced (SWG)

*   **Category:** `model_patches/unet`
*   **Description:** This node implements Sliding Window Guidance, likely for improving coherence or detail in larger images by processing in overlapping tiles. Guidance is derived by comparing full image prediction with tiled predictions.
*   **Parameters:**
    *   **Required:**
        *   `model`:
            *   Type: `MODEL`
            *   Practical Use: Input model for SWG.
        *   `scale`:
            *   Type: `FLOAT`
            *   Default: `5.0`, Min: `0.0`, Max: `100.0`, Step: `0.1`, Round: `0.01`
            *   Practical Use: Strength of SWG. Controls influence of global vs. tiled prediction differences.
        *   `tile_width`:
            *   Type: `INT`
            *   Default: `768`, Min: `16`, Max: `16384`, Step: `8`
            *   Practical Use: Width of individual tiles (pixels). Internally divided by 8.
        *   `tile_height`:
            *   Type: `INT`
            *   Default: `768`, Min: `16`, Max: `16384`, Step: `8`
            *   Practical Use: Height of individual tiles (pixels). Internally divided by 8.
        *   `tile_overlap`:
            *   Type: `INT`
            *   Default: `256`, Min: `16`, Max: `16384`, Step: `8`
            *   Practical Use: Overlap between tiles (pixels). Crucial for smooth transitions. Internally divided by 8.
        *   `sigma_start`:
            *   Type: `FLOAT`
            *   Default: `-1.0`, Min: `-1.0`, Max: `10000.0`, Step: `0.01`
            *   Practical Use: Sigma level to start SWG.
        *   `sigma_end`:
            *   Type: `FLOAT`
            *   Default: `5.42`, Min: `-1.0`, Max: `10000.0`, Step: `0.01`
            *   Practical Use: Sigma level to stop SWG. Default suggests activity in early to middle diffusion stages.

---

## Node: TRT Attach Pag

*   **Category:** `TensorRT`
*   **Description:** Prepares a TensorRT model for Perturbed Attention Guidance (PAG) by patching specified U-Net blocks to use `perturbed_attention`. This output model is typically used with `TRTPerturbedAttention`.
*   **Parameters:**
    *   **Required:**
        *   `model`:
            *   Type: `MODEL`
            *   Practical Use: Input TensorRT-compatible model to be prepared for PAG.
        *   `unet_block`:
            *   Type: `COMBO` (Options: `["input", "middle", "output"]`)
            *   Default: `"middle"`
            *   Practical Use: U-Net block type for applying `perturbed_attention` if `unet_block_list` is empty.
        *   `unet_block_id`:
            *   Type: `INT`
            *   Default: `0`
            *   Practical Use: U-Net block index if `unet_block_list` is empty.
    *   **Optional:**
        *   `unet_block_list`:
            *   Type: `STRING`
            *   Default: `""`
            *   Practical Use: Comma-separated list of specific U-Net self-attention blocks for `perturbed_attention`. Overrides `unet_block` and `unet_block_id`.

---

## Node: TRT Perturbed Attention

*   **Category:** `TensorRT`
*   **Description:** Applies PAG using a base TensorRT model and a PAG-prepared TensorRT model (from `TRTAttachPag`). It calculates PAG by comparing their outputs. Specific to TensorRT accelerated workflows.
*   **Parameters:**
    *   **Required:**
        *   `model_base`:
            *   Type: `MODEL`
            *   Practical Use: Standard, unpatched TensorRT model for regular conditional prediction.
        *   `model_pag`:
            *   Type: `MODEL`
            *   Practical Use: TensorRT model patched with `perturbed_attention` (e.g., via `TRTAttachPag`).
        *   `scale`:
            *   Type: `FLOAT`
            *   Default: `3.0`, Min: `0.0`, Max: `100.0`, Step: `0.1`, Round: `0.01`
            *   Practical Use: Strength of PAG. Amplifies difference between `model_base` and `model_pag`.
        *   `adaptive_scale`:
            *   Type: `FLOAT`
            *   Default: `0.0`, Min: `0.0`, Max: `1.0`, Step: `0.001`, Round: `0.0001`
            *   Practical Use: Modulates `scale` based on timestep; reduces scale in earlier, noisier steps.
        *   `sigma_start`:
            *   Type: `FLOAT`
            *   Default: `-1.0`, Min: `-1.0`, Max: `10000.0`, Step: `0.01`
            *   Practical Use: Sigma level to start PAG.
        *   `sigma_end`:
            *   Type: `FLOAT`
            *   Default: `-1.0`, Min: `-1.0`, Max: `10000.0`, Step: `0.01`
            *   Practical Use: Sigma level to stop PAG.
        *   `rescale`:
            *   Type: `FLOAT`
            *   Default: `0.0`, Min: `0.0`, Max: `1.0`, Step: `0.01`
            *   Practical Use: Factor for rescaling PAG guidance.
        *   `rescale_mode`:
            *   Type: `COMBO` (Options: `["full", "partial"]`)
            *   Default: `"full"`
            *   Practical Use: Mode for applying rescale. ("snf" not available here).

---

## Node: Pladis

*   **Category:** `model_patches/unet`
*   **Experimental:** Yes
*   **Description:** Implements PLADIS (PLAtent DIffusion Sparsification), modifying cross-attention (attn2) layers using sparse activation functions (Entmax 1.5 or Sparsemax). This may focus attention or create cleaner details.
*   **Parameters:**
    *   **Required:**
        *   `model`:
            *   Type: `MODEL`
            *   Practical Use: Input model for PLADIS; its cross-attention layers will be modified.
        *   `scale`:
            *   Type: `FLOAT`
            *   Default: `2.0`, Min: `0.0`, Max: `100.0`, Step: `0.1`, Round: `0.01`
            *   Practical Use: Strength of PLADIS modification. Higher scale means more pronounced sparse attention.
        *   `sparse_func`:
            *   Type: `COMBO` (Options: `["entmax15", "sparsemax"]` - inferred)
            *   Default: `"entmax15"` (inferred)
            *   Practical Use: Determines sparse activation function:
                *   `entmax15`: Smoother sparsity, zeros out some attention scores.
                *   `sparsemax`: Produces highly sparse outputs, forcing many scores to zero.
                *   Choice affects how aggressively attention is sparsified.

---

## Node: Token Perturbation Guidance (TPG)

*   **Display Name:** `Token Perturbation Guidance`
*   **Category:** `model_patches/unet`
*   **Description:** Implements Token Perturbation Guidance. It shuffles tokens within specified U-Net transformer blocks. Guidance is derived from the difference between predictions with normal and perturbed tokens. Can improve prompt alignment or vary interpretations.
*   **Parameters:**
    *   **Required:**
        *   `model`:
            *   Type: `IO.MODEL`
            *   Practical Use: Input model for TPG. Specified transformer blocks are wrapped for token shuffling.
        *   `scale`:
            *   Type: `IO.FLOAT`
            *   Default: `3.0`, Min: `0.0`, Max: `100.0`, Step: `0.1`, Round: `0.01`
            *   Practical Use: Strength of TPG. Higher scale amplifies guidance from token perturbation.
        *   `sigma_start`:
            *   Type: `IO.FLOAT`
            *   Default: `-1.0`, Min: `-1.0`, Max: `10000.0`, Step: `0.01`
            *   Practical Use: Sigma level to start TPG.
        *   `sigma_end`:
            *   Type: `IO.FLOAT`
            *   Default: `-1.0`, Min: `-1.0`, Max: `10000.0`, Step: `0.01`
            *   Practical Use: Sigma level to stop TPG.
        *   `rescale`:
            *   Type: `IO.FLOAT`
            *   Default: `0.0`, Min: `0.0`, Max: `1.0`, Step: `0.01`
            *   Practical Use: Rescaling factor for TPG guidance.
        *   `rescale_mode`:
            *   Type: `IO.COMBO` (Options: `["full", "partial", "snf"]`)
            *   Default: `"full"`
            *   Practical Use: Mode for applying rescale.
    *   **Optional:**
        *   `unet_block_list`:
            *   Type: `IO.STRING`
            *   Default: `"d2.2-9,d3"`
            *   Tooltip/Description: Blocks to which TPG is applied.
            *   Practical Use: Comma-separated string specifying which U-Net transformer blocks are patched. Format like "d2.2-9,d3" targets specific block groups/indices. TPG wraps entire `BasicTransformerBlock` modules. The default targets specific blocks.

---
## General Practical Use, Tips, and Examples

Understanding how to use these advanced guidance nodes effectively can significantly enhance your image generation capabilities. Here are some general tips and example scenarios:

### Controlling Effect Timing with `sigma_start` and `sigma_end`

The `sigma_start` and `sigma_end` parameters are crucial for controlling *when* a guidance effect is active during the diffusion process.
*   **Sigma values range from high (early, noisy steps) to low (late, refined steps).** Most schedulers represent this as e.g. ~14.0 (start) down to 0.0 (end).
*   **`sigma_start = -1.0` (or any negative value):** Typically means the effect starts from the very beginning of the diffusion process (highest sigma).
*   **`sigma_end = -1.0` (or any negative value):** Typically means the effect lasts until the very end of the diffusion process (lowest sigma).
*   **Early Stage Application (High `sigma_start`, relatively High `sigma_end`):** Applying guidance like PAG or TPG primarily in the early stages can influence the fundamental composition and structure of the image. For example, `sigma_start = 14.0`, `sigma_end = 7.0`.
*   **Late Stage Application (Low `sigma_start`, `sigma_end = -1.0` or lower):** Applying guidance in later stages can refine details, textures, or clean up artifacts without drastically altering the core subject. For example, `sigma_start = 5.0`, `sigma_end = -1.0`.
*   **Mid-Stage Application:** Targeting the middle stages can be a balance, influencing both structure and some level of detail.
*   **Full Duration:** `sigma_start = -1.0`, `sigma_end = -1.0` applies the effect throughout the entire generation.

Experiment with these values. Sometimes, an effect that's too strong throughout can be made more subtle and effective by limiting its application to specific phases of diffusion.

### Fine-Grained Control with `unet_block_list`

The `unet_block_list` parameter (available in NAG, PAG, SEG, TPG, TRTAttachPag) offers powerful, advanced control over *where* in the U-Net architecture the guidance is applied.
*   The U-Net has an encoder (input blocks), a middle block, and a decoder (output blocks). Each block typically contains attention layers and ResNet blocks.
*   **Syntax:** The exact syntax (e.g., "middle.0,output.1.attn1", or "d2.2-9,d3" for TPG) depends on how the node parses this string. Refer to specific node documentation or examples if available.
*   **Why use it?**
    *   **Targeting specific features:** Different U-Net blocks contribute to different aspects of the image. Early blocks (input) might handle broader strokes and composition, while later blocks (output) refine details.
    *   **Avoiding conflicts:** If one guidance interferes negatively with another or with the base model's behavior in certain layers, you can exclude those layers.
    *   **Efficiency/Performance:** Applying guidance only where needed might slightly reduce computational overhead, though this is usually minor.
    *   **Experimental effects:** Advanced users can achieve unique styles by selectively applying guidance. For example, applying PAG only to output blocks might sharpen details without changing the overall composition formed by earlier blocks.

If unsure, starting with an empty `unet_block_list` (which often defaults to all relevant blocks or a sensible subset) or the simpler `unet_block`/`unet_block_id` parameters is recommended.

### Example Scenarios:

1.  **Enhancing Detail with PAG (Perturbed Attention Guidance):**
    *   **Goal:** Generate an image with very crisp details, for example, intricate patterns on fabric or sharp facial features.
    *   **How:** Use PAG with a moderate `scale` (e.g., 2.0-5.0). Consider applying it primarily in the mid to late stages (`sigma_start` around 7.0-10.0, `sigma_end` around 0.0-3.0 or -1.0) to refine details developed by the standard diffusion process.
    *   **Tip:** If PAG makes the image too "busy" or introduces artifacts, try reducing the `scale`, increasing `adaptive_scale`, or using `rescale` with "partial" or "snf" mode.

2.  **Refining Negatives with NAG (Normalized Attention Guidance):**
    *   **Goal:** Ensure an image *strongly* avoids certain unwanted concepts, more so than standard negative prompts with CFG. For example, generating a character without "extra limbs" or "distorted hands."
    *   **How:** Use NAG with a specific `negative` conditioning for the unwanted elements. Adjust `scale` and `tau` to control intensity. `alpha` can blend the effect smoothly.
    *   **Tip:** Place the NAG node *after* any other attention-modifying nodes like IPAdapters for compatibility. Start with `alpha = 0.5` and adjust.

3.  **Smoother, Stylized Looks with SEG (Smoothed Energy Guidance):**
    *   **Goal:** Create images with a softer, painterly, or more abstract feel by smoothing attention details.
    *   **How:** Use SEG with a positive `blur_sigma` (e.g., 1.0-3.0). The `scale` will determine how much this smoothed attention influences the image.
    *   **Tip:** Applying SEG to earlier U-Net blocks might result in broader smoothing effects, while applying to later blocks could smooth out finer textures.

4.  **Improving Coherence in Large Images with SWG (Sliding Window Guidance):**
    *   **Goal:** When generating very large images that might suffer from incoherence or repeating patterns if upscaled naively, SWG can help.
    *   **How:** Set `tile_width`, `tile_height`, and `tile_overlap` appropriately for your target resolution. The `scale` determines how strongly the tiled information corrects the global generation.
    *   **Tip:** SWG often works best when active in the earlier to middle stages of diffusion (e.g., default `sigma_end = 5.42`). Ensure `tile_overlap` is sufficient (e.g., 1/4 to 1/3 of tile dimension) to avoid visible seams.

5.  **Targeted Prompt Adherence with TPG (Token Perturbation Guidance):**
    *   **Goal:** Improve the model's focus on specific elements of a complex prompt or explore variations by emphasizing token-level information.
    *   **How:** Use TPG with a moderate `scale`. The default `unet_block_list` ("d2.2-9,d3") is a good starting point, as it targets specific transformer blocks.
    *   **Tip:** TPG can be subtle. Experiment with `scale` and `sigma` ranges. Combining with other guidance methods might yield interesting results.

6.  **TensorRT Acceleration with PAG (TRTAttachPag + TRTPerturbedAttention):**
    *   **Goal:** Utilize PAG for detail enhancement while benefiting from TensorRT's speed.
    *   **How:** First, use `TRTAttachPag` to create a specialized PAG-enabled TensorRT model. Then, feed this model along with your base TensorRT model into `TRTPerturbedAttention`.
    *   **Tip:** Ensure the `unet_block_list` or `unet_block`/`id` settings are consistent between `TRTAttachPag` and your intended PAG application strategy.

### Combining Guidance Methods:

Multiple guidance nodes can often be chained together. The order can matter. For example:
*   `Model -> PAG -> NAG -> KSampler`
*   `Model -> TPG -> KSampler`

When combining, start with lower `scale` values for each guidance method and gradually increase them to find a good balance. Too many strong guidance signals can lead to conflicting instructions and undesirable image artifacts or overly slow generation. Pay attention to whether a node modifies `attn1` (self-attention, like PAG/SEG) or `attn2` (cross-attention, like NAG/Pladis), as this can influence how they interact.

Always experiment and iterate. What works best will depend on the specific model, prompt, and desired artistic outcome.
