"""Tests for CLIP image filtering module.

Uses importlib to load clip_filter directly from its file path,
completely bypassing the gpt_researcher package import chain.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _load_clip_filter():
    """Load clip_filter module directly from file, avoiding package imports."""
    module_name = "gpt_researcher.multimodal.clip_filter"
    sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(
        module_name,
        Path(__file__).resolve().parent.parent / "gpt_researcher" / "multimodal" / "clip_filter.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestCLIPImageFilterInit:
    """Test CLIPImageFilter initialization behavior."""

    def test_init_success(self):
        mod = _load_clip_filter()
        mod._CLIP_AVAILABLE = True

        mock_model_instance = MagicMock()
        mock_model_cls = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance

        mock_proc_cls = MagicMock()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.object(mod, "CLIPModel", mock_model_cls, create=True), \
             patch.object(mod, "CLIPProcessor", mock_proc_cls, create=True), \
             patch.object(mod, "torch", mock_torch, create=True):
            f = mod.CLIPImageFilter(model_name="test-model", device="cuda")

        assert f.available is True
        assert f.device == "cuda"
        mock_model_cls.from_pretrained.assert_called_once_with("test-model")
        mock_model_instance.to.assert_called_once_with("cuda")

    def test_init_fallback_no_deps(self):
        mod = _load_clip_filter()
        mod._CLIP_AVAILABLE = False
        f = mod.CLIPImageFilter()
        assert f.available is False
        assert f.model is None

    def test_init_model_load_failure(self):
        mod = _load_clip_filter()
        mod._CLIP_AVAILABLE = True

        mock_model_cls = MagicMock()
        mock_model_cls.from_pretrained.side_effect = RuntimeError("load failed")
        mock_proc_cls = MagicMock()

        with patch.object(mod, "CLIPModel", mock_model_cls, create=True), \
             patch.object(mod, "CLIPProcessor", mock_proc_cls, create=True):
            f = mod.CLIPImageFilter()

        assert f.available is False

    def test_init_auto_device(self):
        mod = _load_clip_filter()
        mod._CLIP_AVAILABLE = True

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_model_instance = MagicMock()
        mock_model_cls = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        mock_proc_cls = MagicMock()

        with patch.object(mod, "CLIPModel", mock_model_cls, create=True), \
             patch.object(mod, "CLIPProcessor", mock_proc_cls, create=True), \
             patch.object(mod, "torch", mock_torch, create=True):
            f = mod.CLIPImageFilter()

        assert f.device == "cpu"


class TestScoreImages:
    """Test score_images behavior."""

    def test_score_images_unavailable(self):
        mod = _load_clip_filter()
        mod._CLIP_AVAILABLE = False
        f = mod.CLIPImageFilter()
        images = [{"url": "http://a.com/img.jpg", "score": 1}]
        result = f.score_images(images, "cat")
        assert len(result) == 1
        assert result[0]["clip_score"] == 0.0

    def test_score_images_with_mocks(self):
        mod = _load_clip_filter()
        mod._CLIP_AVAILABLE = True
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        mock_model_instance = MagicMock()
        mock_model_cls = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        mock_proc_cls = MagicMock()

        with patch.object(mod, "CLIPModel", mock_model_cls, create=True), \
             patch.object(mod, "CLIPProcessor", mock_proc_cls, create=True), \
             patch.object(mod, "torch", mock_torch, create=True):
            f = mod.CLIPImageFilter(device="cpu")
            f.available = True

            # Patch _cosine_similarity to return controlled values
            # so we don't need real torch tensors
            with patch.object(f, "_encode_text", return_value=MagicMock()), \
                 patch.object(f, "_load_image", return_value=MagicMock()), \
                 patch.object(f, "_encode_image", side_effect=[MagicMock(), MagicMock()]), \
                 patch.object(f, "_cosine_similarity", side_effect=[0.95, 0.30]):
                result = f.score_images(
                    [{"url": "http://a.com/1.jpg"}, {"url": "http://b.com/2.jpg"}],
                    "cat",
                )

        assert len(result) == 2
        assert all("clip_score" in r for r in result)
        # Sorted descending by clip_score
        assert result[0]["clip_score"] >= result[1]["clip_score"]
        assert result[0]["clip_score"] == 0.95
        assert result[1]["clip_score"] == 0.30

    def test_score_images_empty_url(self):
        mod = _load_clip_filter()
        mod._CLIP_AVAILABLE = True
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        mock_model_instance = MagicMock()
        mock_model_cls = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        mock_proc_cls = MagicMock()

        with patch.object(mod, "CLIPModel", mock_model_cls, create=True), \
             patch.object(mod, "CLIPProcessor", mock_proc_cls, create=True), \
             patch.object(mod, "torch", mock_torch, create=True):
            f = mod.CLIPImageFilter(device="cpu")
            f.available = True

            with patch.object(f, "_load_image", return_value=None), \
                 patch.object(f, "_encode_text", return_value=MagicMock()):
                result = f.score_images([{"url": ""}], "cat")
        # Empty url is skipped (not appended to result)
        assert len(result) == 0


class TestFilterRelevant:
    """Test filter_relevant threshold behavior."""

    def test_filter_relevant_unavailable(self):
        mod = _load_clip_filter()
        mod._CLIP_AVAILABLE = False
        f = mod.CLIPImageFilter()
        result = f.filter_relevant(
            [{"url": "http://a.com/img.jpg"}], "cat", threshold=0.25
        )
        # Score is 0.0, below threshold
        assert len(result) == 0

    def test_filter_relevant_threshold(self):
        mod = _load_clip_filter()
        mod._CLIP_AVAILABLE = True
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        mock_model_instance = MagicMock()
        mock_model_cls = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        mock_proc_cls = MagicMock()

        with patch.object(mod, "CLIPModel", mock_model_cls, create=True), \
             patch.object(mod, "CLIPProcessor", mock_proc_cls, create=True), \
             patch.object(mod, "torch", mock_torch, create=True):
            f = mod.CLIPImageFilter(device="cpu")
            f.available = True

            # Patch _cosine_similarity to return controlled values
            with patch.object(f, "_encode_text", return_value=MagicMock()), \
                 patch.object(f, "_load_image", return_value=MagicMock()), \
                 patch.object(f, "_encode_image", side_effect=[MagicMock(), MagicMock()]), \
                 patch.object(f, "_cosine_similarity", side_effect=[0.95, -0.10]):
                result = f.filter_relevant(
                    [{"url": "http://a.com/1.jpg"}, {"url": "http://b.com/2.jpg"}],
                    "cat",
                    threshold=0.25,
                )

        # Only the high-similarity image (0.95) passes threshold; -0.10 is below
        assert len(result) == 1
        assert result[0]["clip_score"] == 0.95


class TestLoadImage:
    """Test _load_image HTTP handling."""

    def test_load_image_success(self):
        mod = _load_clip_filter()
        mod._CLIP_AVAILABLE = True
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        mock_model_instance = MagicMock()
        mock_model_cls = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        mock_proc_cls = MagicMock()

        with patch.object(mod, "CLIPModel", mock_model_cls, create=True), \
             patch.object(mod, "CLIPProcessor", mock_proc_cls, create=True), \
             patch.object(mod, "torch", mock_torch, create=True):
            f = mod.CLIPImageFilter(device="cpu")

        mock_response = MagicMock()
        mock_response.content = b"\x89PNG\r\n"
        mock_response.raise_for_status = MagicMock()

        with patch.object(mod, "httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response

            with patch.object(mod, "Image", create=True) as mock_image:
                mock_image.open.return_value = "fake_image"
                result = f._load_image("http://example.com/img.png")
                assert result == "fake_image"

    def test_load_image_failure(self):
        mod = _load_clip_filter()
        mod._CLIP_AVAILABLE = True
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        mock_model_instance = MagicMock()
        mock_model_cls = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        mock_proc_cls = MagicMock()

        with patch.object(mod, "CLIPModel", mock_model_cls, create=True), \
             patch.object(mod, "CLIPProcessor", mock_proc_cls, create=True), \
             patch.object(mod, "torch", mock_torch, create=True):
            f = mod.CLIPImageFilter(device="cpu")

        with patch.object(mod, "httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = Exception("network error")

            result = f._load_image("http://example.com/bad.png")
            assert result is None
