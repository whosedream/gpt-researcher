"""CLIP-based image relevance filtering.

Uses OpenAI's CLIP model to score image-text relevance via cosine similarity
between image and text embeddings.
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

try:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False
    logger.info(
        "CLIP dependencies not installed. "
        "Install with: pip install 'gpt-researcher[clip]'"
    )


class CLIPImageFilter:
    """Filters images by semantic relevance to a text query using CLIP.

    Falls back gracefully when CLIP dependencies are unavailable.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ):
        """Initialize CLIP model and processor.

        Args:
            model_name: HuggingFace model identifier.
            device: Compute device ("cuda", "cpu", or None for auto-detect).

        Attributes:
            available: Whether CLIP loaded successfully.
        """
        self.available = False
        self.model = None
        self.processor = None
        self.device = device

        if not _CLIP_AVAILABLE:
            logger.warning("CLIP dependencies not available, filter disabled.")
            return

        try:
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self.available = True
            logger.info(f"CLIP model loaded: {model_name} on {self.device}")
        except Exception as e:
            logger.warning(f"Failed to load CLIP model: {e}")
            self.available = False

    def score_images(self, images: list[dict], query: str) -> list[dict]:
        """Score images by relevance to a text query.

        Args:
            images: List of image dicts, each with at least a "url" key.
            query: Text query to measure relevance against.

        Returns:
            List of image dicts with added "clip_score" key, sorted descending.
        """
        if not self.available:
            logger.warning("CLIP not available, returning images with score 0.")
            return [{**img, "clip_score": 0.0} for img in images]

        scored = []
        text_emb = self._encode_text(query)

        for img_data in images:
            url = img_data.get("url", "")
            if not url:
                continue

            image = self._load_image(url)
            if image is None:
                scored.append({**img_data, "clip_score": 0.0})
                continue

            image_emb = self._encode_image(image)
            if image_emb is None:
                scored.append({**img_data, "clip_score": 0.0})
                continue

            similarity = self._cosine_similarity(image_emb, text_emb)
            scored.append({**img_data, "clip_score": float(similarity)})

        scored.sort(key=lambda x: x["clip_score"], reverse=True)
        return scored

    def filter_relevant(
        self,
        images: list[dict],
        query: str,
        threshold: float = 0.25,
    ) -> list[dict]:
        """Filter images by CLIP relevance threshold.

        Args:
            images: List of image dicts with "url" key.
            query: Text query for relevance scoring.
            threshold: Minimum cosine similarity to keep (0-1).

        Returns:
            Filtered and sorted list of image dicts.
        """
        scored = self.score_images(images, query)
        return [img for img in scored if img.get("clip_score", 0.0) >= threshold]

    def _load_image(self, image_url: str) -> Optional["Image.Image"]:
        """Load an image from a URL or local file path.

        Args:
            image_url: HTTP(S) URL or local file path of the image.

        Returns:
            PIL Image or None if load fails.
        """
        try:
            # Support local file paths
            if not image_url.startswith(("http://", "https://")):
                return Image.open(image_url).convert("RGB")
            # Remote URL: download via httpx
            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                response = client.get(image_url)
                response.raise_for_status()
                return Image.open(response.content).convert("RGB")
        except Exception as e:
            logger.debug(f"Failed to load image {image_url}: {e}")
            return None

    def _encode_image(self, image: "Image.Image") -> Optional["torch.Tensor"]:
        """Encode a PIL image into a CLIP embedding tensor.

        Args:
            image: PIL Image to encode.

        Returns:
            Normalized embedding tensor or None on failure.
        """
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                output = self.model.get_image_features(**inputs)
            # get_image_features returns BaseModelOutputWithPooling
            embeddings = output.pooler_output if hasattr(output, "pooler_output") else output
            return embeddings / embeddings.norm(dim=-1, keepdim=True)
        except Exception as e:
            logger.debug(f"Failed to encode image: {e}")
            return None

    def _encode_text(self, text: str) -> Optional["torch.Tensor"]:
        """Encode text into a CLIP embedding tensor.

        Args:
            text: Text string to encode.

        Returns:
            Normalized embedding tensor or None on failure.
        """
        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                output = self.model.get_text_features(**inputs)
            # get_text_features returns BaseModelOutputWithPooling
            embeddings = output.pooler_output if hasattr(output, "pooler_output") else output
            return embeddings / embeddings.norm(dim=-1, keepdim=True)
        except Exception as e:
            logger.debug(f"Failed to encode text: {e}")
            return None

    def _cosine_similarity(
        self, img_emb: "torch.Tensor", txt_emb: "torch.Tensor"
    ) -> "torch.Tensor":
        """Compute cosine similarity between image and text embeddings.

        Args:
            img_emb: Image embedding tensor (1, dim).
            txt_emb: Text embedding tensor (1, dim).

        Returns:
            Scalar similarity score tensor.
        """
        return torch.nn.functional.cosine_similarity(img_emb, txt_emb)
