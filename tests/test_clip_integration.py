"""Integration tests for CLIP scoring in get_relevant_images.

Tests the CSS-only path, CLIP-enabled path, and fallback behavior.
Mocks heavy dependencies to avoid importing gpt_researcher's full chain.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


TEST_HTML = """
<html>
<body>
    <img src="http://example.com/hero.jpg" class="hero" width="1920" height="1080">
    <img src="http://example.com/small.png" width="100" height="100">
    <img src="http://example.com/content.jpg" class="content" width="800" height="600">
    <img src="http://example.com/banner.png" width="1600" height="800">
    <img src="http://example.com/thumb.jpg" class="thumbnail">
</body>
</html>
"""


def _load_utils_mod():
    """Load utils module with mocked dependencies in sys.modules."""
    mod_name = "gpt_researcher.scraper.utils"
    sys.modules.pop(mod_name, None)

    # Mock parent packages
    mock_pkg = MagicMock()
    mock_pkg.__path__ = [str(Path(__file__).resolve().parent.parent / "gpt_researcher" / "scraper")]
    sys.modules["gpt_researcher"] = mock_pkg
    sys.modules["gpt_researcher.scraper"] = mock_pkg

    # Mock bs4
    sys.modules["bs4"] = MagicMock()
    sys.modules["bs4.BeautifulSoup"] = MagicMock()

    # Mock gpt_researcher.config so _get_clip_max_images can import Config
    mock_config_module = MagicMock()
    sys.modules["gpt_researcher.config"] = mock_config_module

    # Mock gpt_researcher.multimodal so _apply_clip_scoring can import
    sys.modules["gpt_researcher.multimodal"] = MagicMock()
    sys.modules["gpt_researcher.multimodal.clip_filter"] = MagicMock()

    spec = importlib.util.spec_from_file_location(
        mod_name,
        Path(__file__).resolve().parent.parent / "gpt_researcher" / "scraper" / "utils.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod, mock_config_module


def _cleanup_modules():
    """Remove injected mock modules."""
    for key in list(sys.modules.keys()):
        if key.startswith("gpt_researcher"):
            del sys.modules[key]


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    _cleanup_modules()


def _make_soup(html: str = ""):
    """Create a mock BeautifulSoup that returns test img tags from find_all."""
    soup = MagicMock()
    # Simulate 5 images matching TEST_HTML structure
    imgs = []
    for src, cls, w, h in [
        ("http://example.com/hero.jpg", ["hero"], "1920", "1080"),
        ("http://example.com/small.png", [], "100", "100"),
        ("http://example.com/content.jpg", ["content"], "800", "600"),
        ("http://example.com/banner.png", [], "1600", "800"),
        ("http://example.com/thumb.jpg", ["thumbnail"], None, None),
    ]:
        img = MagicMock()
        img.__getitem__ = lambda self, key: {"src": src, "class": cls, "width": w, "height": h}[key]
        img.get = lambda key, default=None: {"src": src, "class": cls, "width": w, "height": h}.get(key, default)
        img.__contains__ = lambda self, key: key in {"src", "class", "width", "height"}
        imgs.append(img)
    soup.find_all.return_value = imgs
    return soup


class TestCSSOnlyMode:
    """Test get_relevant_images with CLIP disabled (default behavior)."""

    def test_no_query_returns_css_sorted(self):
        mod, mock_config = _load_utils_mod()
        # Default: clip not enabled, max_images=10
        mock_config.Config.return_value.clip_enabled = False
        mock_config.Config.return_value.clip_max_images = 10

        soup = _make_soup(TEST_HTML)
        result = mod.get_relevant_images(soup, "http://example.com")

        assert len(result) > 0
        assert all("url" in img for img in result)
        assert all("score" in img for img in result)
        # Hero (class=hero, score=4) should be first
        assert result[0]["score"] >= result[-1]["score"]

    def test_query_none_no_clip(self):
        mod, mock_config = _load_utils_mod()
        mock_config.Config.return_value.clip_enabled = False
        mock_config.Config.return_value.clip_max_images = 10

        soup = _make_soup(TEST_HTML)
        result = mod.get_relevant_images(soup, "http://example.com", query=None)
        assert len(result) > 0


class TestCLIPEnabledMode:
    """Test get_relevant_images with CLIP enabled."""

    def test_clip_enabled_re_ranking(self):
        mod, mock_config = _load_utils_mod()
        mock_config.Config.return_value.clip_enabled = True
        mock_config.Config.return_value.clip_model = "test-model"
        mock_config.Config.return_value.clip_device = "cpu"
        mock_config.Config.return_value.clip_relevance_threshold = 0.25
        mock_config.Config.return_value.clip_max_images = 10

        mock_filter = MagicMock()
        mock_filter.available = True
        mock_filter.filter_relevant.return_value = [
            {"url": "http://example.com/hero.jpg", "score": 4, "clip_score": 0.85},
            {"url": "http://example.com/content.jpg", "score": 4, "clip_score": 0.72},
        ]

        # Patch CLIPImageFilter on the mocked clip_filter module
        sys.modules["gpt_researcher.multimodal.clip_filter"].CLIPImageFilter.return_value = mock_filter

        soup = _make_soup(TEST_HTML)
        result = mod.get_relevant_images(soup, "http://example.com", query="cat photo")

        # CLIP filter should have been called
        mock_filter.filter_relevant.assert_called_once()
        assert len(result) == 2
        assert all("clip_score" in img for img in result)

    def test_clip_fallback_on_failure(self):
        mod, mock_config = _load_utils_mod()
        mock_config.Config.return_value.clip_enabled = True
        mock_config.Config.return_value.clip_model = "test-model"
        mock_config.Config.return_value.clip_device = "cpu"
        mock_config.Config.return_value.clip_max_images = 10

        # Make CLIPImageFilter raise on construction
        sys.modules["gpt_researcher.multimodal.clip_filter"].CLIPImageFilter.side_effect = Exception("model fail")

        result = mod.get_relevant_images(
            _make_soup(TEST_HTML), "http://example.com", query="cat"
        )
        # Should still return results (CSS scores fallback)
        assert len(result) > 0
        assert all("url" in img for img in result)

    def test_clip_disabled_skips_scoring(self):
        mod, mock_config = _load_utils_mod()
        mock_config.Config.return_value.clip_enabled = False
        mock_config.Config.return_value.clip_max_images = 10

        soup = _make_soup(TEST_HTML)
        result = mod.get_relevant_images(soup, "http://example.com", query="cat photo")

        assert len(result) > 0
        assert all("clip_score" not in img for img in result)


class TestApplyClipScoring:
    """Test the _apply_clip_scoring helper directly."""

    def test_clip_disabled_returns_unchanged(self):
        mod, mock_config = _load_utils_mod()
        mock_config.Config.return_value.clip_enabled = False
        images = [{"url": "http://a.com/1.jpg", "score": 2}]

        result = mod._apply_clip_scoring(images, "cat")
        assert len(result) == 1
        assert result[0]["url"] == "http://a.com/1.jpg"

    def test_clip_unavailable_returns_with_zero_score(self):
        mod, mock_config = _load_utils_mod()
        mock_config.Config.return_value.clip_enabled = True
        mock_config.Config.return_value.clip_model = "test"
        mock_config.Config.return_value.clip_device = "cpu"

        mock_filter = MagicMock()
        mock_filter.available = False
        sys.modules["gpt_researcher.multimodal.clip_filter"].CLIPImageFilter.return_value = mock_filter

        images = [{"url": "http://a.com/1.jpg", "score": 2}]
        result = mod._apply_clip_scoring(images, "cat")

        assert len(result) == 1
        assert result[0]["clip_score"] == 0.0
