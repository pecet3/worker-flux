import os
from typing import List

import torch
from diffusers import (
    FluxPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
# from diffusers.pipelines.stable_diffusion.safety_checker import (
#     StableDiffusionSafetyChecker,
# )


MODEL_ID = "black-forest-labs/FLUX.1-schnell"
MODEL_CACHE = "diffusers-cache"


class Predictor:
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        self.pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            safety_checker=None,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")

        self.pipe.enable_xformers_memory_efficient_attention()

    @torch.inference_mode()
    def predict(self, prompt, num_outputs, num_inference_steps, guidance_scale, scheduler, seed):
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        self.pipe.scheduler = make_scheduler(scheduler, self.pipe.scheduler.config)

        generator = torch.Generator("cuda").manual_seed(seed)
        output = self.pipe(
            prompt=[prompt] * num_outputs if prompt is not None else None,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            if output.nsfw_content_detected and output.nsfw_content_detected[i] and self.NSFW:
                continue

            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(output_path)

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]
