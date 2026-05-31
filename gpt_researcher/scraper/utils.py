"""Utility functions for web scraping.

This module provides helper functions for extracting content, images,
and processing HTML from web pages.
"""

import hashlib
import logging
import re
from urllib.parse import parse_qs, urljoin, urlparse

import bs4
from bs4 import BeautifulSoup


def get_relevant_images(soup: BeautifulSoup, url: str, query: str = None) -> list:
    """Extract relevant images from the page.

    Uses CSS-based scoring by default. When a query is provided and CLIP is
    enabled in config, images are additionally scored by semantic relevance
    to the query text via CLIP cosine similarity.

    Args:
        soup: Parsed HTML document.
        url: Source URL (used to resolve relative image paths).
        query: Optional search query for CLIP-based semantic filtering.

    Returns:
        List of image dicts sorted by relevance, limited to CLIP_MAX_IMAGES.
    """
    image_urls = []

    try:
        # Find all img tags with src attribute
        all_images = soup.find_all('img', src=True)

        for img in all_images:
            img_src = urljoin(url, img['src'])
            if img_src.startswith(('http://', 'https://')):
                score = 0
                # Check for relevant classes
                if any(cls in img.get('class', []) for cls in ['header', 'featured', 'hero', 'thumbnail', 'main', 'content']):
                    score = 4  # Higher score
                # Check for size attributes
                elif img.get('width') and img.get('height'):
                    width = parse_dimension(img['width'])
                    height = parse_dimension(img['height'])
                    if width and height:
                        if width >= 2000 and height >= 1000:
                            score = 3  # Medium score (very large images)
                        elif width >= 1600 or height >= 800:
                            score = 2  # Lower score
                        elif width >= 800 or height >= 500:
                            score = 1  # Lowest score
                        elif width >= 500 or height >= 300:
                            score = 0  # Lowest score
                        else:
                            continue  # Skip small images

                image_urls.append({'url': img_src, 'score': score})

        # Sort images by CSS score (highest first)
        sorted_images = sorted(image_urls, key=lambda x: x['score'], reverse=True)

        # Keep a larger candidate pool for CLIP re-ranking
        candidate_limit = 20 if query else 10
        candidates = sorted_images[:candidate_limit]

        # Apply CLIP semantic re-ranking when a query is provided
        if query:
            candidates = _apply_clip_scoring(candidates, query)

        max_images = _get_clip_max_images()
        return candidates[:max_images]

    except Exception as e:
        logging.error(f"Error in get_relevant_images: {e}")
        return []


def _apply_clip_scoring(images: list, query: str) -> list:
    """Apply CLIP-based semantic re-ranking to candidate images.

    If CLIP is disabled or unavailable, returns images unchanged with
    clip_score set to 0.0 for consistent downstream handling.
    """
    try:
        from gpt_researcher.config import Config
        config = Config()

        if not getattr(config, 'clip_enabled', False):
            return images

        from gpt_researcher.multimodal.clip_filter import CLIPImageFilter
        clip_filter = CLIPImageFilter(
            model_name=getattr(config, 'clip_model', 'openai/clip-vit-base-patch32'),
            device=getattr(config, 'clip_device', 'cuda'),
        )

        if not clip_filter.available:
            logging.info("CLIP model not available, using CSS scores only.")
            return [{**img, 'clip_score': 0.0} for img in images]

        threshold = getattr(config, 'clip_relevance_threshold', 0.25)
        return clip_filter.filter_relevant(images, query, threshold=threshold)

    except Exception as e:
        logging.warning(f"CLIP scoring failed, falling back to CSS scores: {e}")
        return [{**img, 'clip_score': 0.0} for img in images]


def _get_clip_max_images() -> int:
    """Get the configured max images limit."""
    try:
        from gpt_researcher.config import Config
        config = Config()
        return getattr(config, 'clip_max_images', 10)
    except Exception:
        return 10

def parse_dimension(value: str) -> int:
    """Parse dimension value, handling px units"""
    if value.lower().endswith('px'):
        value = value[:-2]  # Remove 'px' suffix
    try:
        # Convert to float first to handle decimal values like '409.12'
        return int(float(value))
    except (ValueError, TypeError) as e:
        print(f"Error parsing dimension value {value}: {e}")
        return None

def extract_title(soup: BeautifulSoup) -> str:
    """Extract the title from the BeautifulSoup object"""
    return soup.title.string if soup.title else ""

def get_image_hash(image_url: str) -> str:
    """Calculate a simple hash based on the image filename and essential query parameters"""
    try:
        parsed_url = urlparse(image_url)
        
        # Extract the filename
        filename = parsed_url.path.split('/')[-1]
        
        # Extract essential query parameters (e.g., 'url' for CDN-served images)
        query_params = parse_qs(parsed_url.query)
        essential_params = query_params.get('url', [])
        
        # Combine filename and essential parameters
        image_identifier = filename + ''.join(essential_params)
        
        # Calculate hash
        return hashlib.md5(image_identifier.encode()).hexdigest()
    except Exception as e:
        logging.error(f"Error calculating image hash for {image_url}: {e}")
        return None


def clean_soup(soup: BeautifulSoup) -> BeautifulSoup:
    """Clean the soup by removing unwanted tags"""
    for tag in soup.find_all(
        [
            "script",
            "style",
            "footer",
            "header",
            "nav",
            "menu",
            "sidebar",
            "svg",
        ]
    ):
        tag.decompose()

    disallowed_class_set = {"nav", "menu", "sidebar", "footer"}

    # clean tags with certain classes
    def does_tag_have_disallowed_class(elem) -> bool:
        if not isinstance(elem, bs4.Tag):
            return False

        return any(
            cls_name in disallowed_class_set for cls_name in elem.get("class", [])
        )

    for tag in soup.find_all(does_tag_have_disallowed_class):
        tag.decompose()

    return soup


def get_text_from_soup(soup: BeautifulSoup) -> str:
    """Get the relevant text from the soup with improved filtering"""
    text = soup.get_text(strip=True, separator="\n")
    # Remove excess whitespace
    text = re.sub(r"\s{2,}", " ", text)
    return text