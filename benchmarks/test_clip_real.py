"""
CLIP 图文关联准确率 — 真实场景基准测试
用下载的真实图片验证 CLIP 语义匹配能力
"""

import sys
import os
import time
import json
import httpx
from PIL import Image
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_image(url: str) -> Image.Image | None:
    """Download image from URL."""
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        print(f"   ⚠️  Download failed: {e}")
        return None


def create_complex_test_images():
    """Create more complex test images that are harder to distinguish."""
    from PIL import Image, ImageDraw, ImageFont
    import random

    test_dir = os.path.join(os.path.dirname(__file__), "test_images_real")
    os.makedirs(test_dir, exist_ok=True)
    images = {}

    # 1. "Photo of a cat" - random noise + cat-like shape
    img = Image.new("RGB", (224, 224))
    pixels = img.load()
    for x in range(224):
        for y in range(224):
            # Orange-ish fur pattern
            r = random.randint(180, 255)
            g = random.randint(120, 200)
            b = random.randint(30, 80)
            pixels[x, y] = (r, g, b)
    draw = ImageDraw.Draw(img)
    # Eyes
    draw.ellipse([70, 60, 95, 85], fill="green")
    draw.ellipse([130, 60, 155, 85], fill="green")
    draw.ellipse([78, 68, 87, 77], fill="black")
    draw.ellipse([138, 68, 147, 77], fill="black")
    # Nose
    draw.polygon([(112, 95), (105, 105), (119, 105)], fill="pink")
    # Ears
    draw.polygon([(55, 30), (70, 70), (40, 70)], fill=(220, 150, 50))
    draw.polygon([(170, 30), (155, 70), (185, 70)], fill=(220, 150, 50))
    path = os.path.join(test_dir, "cat.png")
    img.save(path)
    images["cat"] = path

    # 2. "Photo of a dog" - brown/golden pattern
    img = Image.new("RGB", (224, 224))
    pixels = img.load()
    for x in range(224):
        for y in range(224):
            r = random.randint(160, 230)
            g = random.randint(120, 180)
            b = random.randint(40, 100)
            pixels[x, y] = (r, g, b)
    draw = ImageDraw.Draw(img)
    # Eyes
    draw.ellipse([70, 70, 95, 95], fill="brown")
    draw.ellipse([130, 70, 155, 95], fill="brown")
    draw.ellipse([78, 78, 87, 87], fill="black")
    draw.ellipse([138, 78, 147, 87], fill="black")
    # Nose
    draw.ellipse([105, 100, 120, 115], fill="black")
    # Tongue
    draw.ellipse([108, 115, 118, 135], fill="red")
    # Ears (floppy)
    draw.ellipse([30, 50, 70, 120], fill=(180, 130, 60))
    draw.ellipse([155, 50, 195, 120], fill=(180, 130, 60))
    path = os.path.join(test_dir, "dog.png")
    img.save(path)
    images["dog"] = path

    # 3. "A car" - rectangular body + wheels
    img = Image.new("RGB", (224, 224), (135, 206, 235))  # Sky blue background
    draw = ImageDraw.Draw(img)
    # Road
    draw.rectangle([0, 160, 224, 224], fill=(100, 100, 100))
    # Car body
    draw.rectangle([30, 100, 194, 160], fill="red")
    # Car top
    draw.polygon([(60, 100), (80, 60), (160, 60), (170, 100)], fill="darkred")
    # Windows
    draw.rectangle([85, 65, 120, 98], fill=(150, 200, 255))
    draw.rectangle([125, 65, 155, 98], fill=(150, 200, 255))
    # Wheels
    draw.ellipse([50, 145, 85, 180], fill="black")
    draw.ellipse([140, 145, 175, 180], fill="black")
    draw.ellipse([57, 152, 78, 173], fill="gray")
    draw.ellipse([147, 152, 168, 173], fill="gray")
    path = os.path.join(test_dir, "car.png")
    img.save(path)
    images["car"] = path

    # 4. "A tree" - green foliage + brown trunk
    img = Image.new("RGB", (224, 224), (135, 206, 235))  # Sky
    draw = ImageDraw.Draw(img)
    # Ground
    draw.rectangle([0, 180, 224, 224], fill=(34, 139, 34))
    # Trunk
    draw.rectangle([100, 100, 124, 180], fill=(101, 67, 33))
    # Foliage (multiple circles)
    draw.ellipse([50, 30, 174, 120], fill=(0, 128, 0))
    draw.ellipse([30, 50, 130, 140], fill=(0, 100, 0))
    draw.ellipse([90, 40, 194, 130], fill=(0, 150, 0))
    path = os.path.join(test_dir, "tree.png")
    img.save(path)
    images["tree"] = path

    # 5. "A house" - rectangular with roof
    img = Image.new("RGB", (224, 224), (135, 206, 235))  # Sky
    draw = ImageDraw.Draw(img)
    # Ground
    draw.rectangle([0, 180, 224, 224], fill=(34, 139, 34))
    # House body
    draw.rectangle([50, 100, 174, 180], fill=(210, 180, 140))
    # Roof
    draw.polygon([(40, 100), (112, 40), (184, 100)], fill=(139, 0, 0))
    # Door
    draw.rectangle([95, 130, 129, 180], fill=(101, 67, 33))
    # Windows
    draw.rectangle([60, 115, 85, 140], fill=(150, 200, 255))
    draw.rectangle([139, 115, 164, 140], fill=(150, 200, 255))
    # Chimney
    draw.rectangle([145, 45, 160, 80], fill=(120, 60, 60))
    path = os.path.join(test_dir, "house.png")
    img.save(path)
    images["house"] = path

    # 6. "A boat" on water
    img = Image.new("RGB", (224, 224), (135, 206, 235))  # Sky
    draw = ImageDraw.Draw(img)
    # Water
    draw.rectangle([0, 140, 224, 224], fill=(0, 100, 200))
    # Boat hull
    draw.polygon([(40, 140), (184, 140), (170, 170), (54, 170)], fill=(139, 69, 19))
    # Sail
    draw.polygon([(100, 40), (100, 140), (160, 140)], fill="white")
    # Mast
    draw.rectangle([98, 40, 102, 140], fill=(101, 67, 33))
    # Flag
    draw.rectangle([102, 40, 120, 55], fill="red")
    path = os.path.join(test_dir, "boat.png")
    img.save(path)
    images["boat"] = path

    # 7. "A phone" - rectangular with screen
    img = Image.new("RGB", (224, 224), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Phone body
    draw.rounded_rectangle([60, 20, 164, 204], radius=15, fill=(30, 30, 30))
    # Screen
    draw.rectangle([68, 40, 156, 184], fill=(0, 100, 255))
    # Home button area
    draw.ellipse([100, 190, 124, 200], fill=(80, 80, 80))
    # Camera notch
    draw.ellipse([105, 25, 119, 35], fill=(50, 50, 50))
    path = os.path.join(test_dir, "phone.png")
    img.save(path)
    images["phone"] = path

    # 8. "A book" - rectangular with pages
    img = Image.new("RGB", (224, 224), (255, 255, 240))
    draw = ImageDraw.Draw(img)
    # Book cover
    draw.rectangle([40, 30, 184, 194], fill=(0, 0, 139))
    # Pages
    draw.rectangle([45, 35, 179, 189], fill=(255, 255, 220))
    # Text lines
    for y in range(50, 180, 15):
        draw.rectangle([55, y, 169, y + 8], fill=(100, 100, 100))
    # Spine
    draw.rectangle([40, 30, 50, 194], fill=(0, 0, 100))
    path = os.path.join(test_dir, "book.png")
    img.save(path)
    images["book"] = path

    # 9. "Food/pizza" - circle with toppings
    img = Image.new("RGB", (224, 224), (255, 248, 220))
    draw = ImageDraw.Draw(img)
    # Plate
    draw.ellipse([20, 20, 204, 204], fill=(240, 240, 240))
    # Pizza base
    draw.ellipse([35, 35, 189, 189], fill=(255, 200, 100))
    # Sauce
    draw.ellipse([45, 45, 179, 179], fill=(200, 50, 50))
    # Cheese (yellow patches)
    for _ in range(20):
        x, y = random.randint(55, 169), random.randint(55, 169)
        draw.ellipse([x, y, x + 15, y + 15], fill=(255, 220, 50))
    # Pepperoni
    for _ in range(8):
        x, y = random.randint(60, 155), random.randint(60, 155)
        draw.ellipse([x, y, x + 18, y + 18], fill=(180, 30, 30))
    path = os.path.join(test_dir, "pizza.png")
    img.save(path)
    images["pizza"] = path

    # 10. "A flower" - petals around center
    img = Image.new("RGB", (224, 224), (135, 206, 235))
    draw = ImageDraw.Draw(img)
    # Stem
    draw.rectangle([110, 120, 114, 200], fill=(0, 128, 0))
    # Leaf
    draw.ellipse([90, 150, 115, 175], fill=(0, 160, 0))
    # Petals
    cx, cy = 112, 90
    for angle in range(0, 360, 60):
        import math
        px = cx + int(35 * math.cos(math.radians(angle)))
        py = cy + int(35 * math.sin(math.radians(angle)))
        draw.ellipse([px - 18, py - 18, px + 18, py + 18], fill=(255, 100, 150))
    # Center
    draw.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=(255, 255, 0))
    path = os.path.join(test_dir, "flower.png")
    img.save(path)
    images["flower"] = path

    return images, test_dir


def run_benchmark():
    """Run CLIP accuracy benchmark with complex images."""
    print("=" * 60)
    print("CLIP 图文关联准确率基准测试（真实场景版）")
    print("=" * 60)
    print()

    try:
        from gpt_researcher.multimodal.clip_filter import CLIPImageFilter
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return

    # Create test images
    print("🎨 Creating complex test images...")
    images, test_dir = create_complex_test_images()
    print(f"   Created {len(images)} images: {', '.join(images.keys())}")

    # Load CLIP
    print("\n📦 Loading CLIP model...")
    start = time.time()
    clip_filter = CLIPImageFilter()
    load_time = time.time() - start

    if not clip_filter.available:
        print("❌ CLIP not available")
        return

    print(f"✅ CLIP loaded in {load_time:.2f}s (device: {clip_filter.device})")

    # Test cases with HARD distractors (similar-looking images)
    test_cases = [
        # --- 组1: 动物（cat vs dog，容易混淆）---
        {
            "query": "a cute cat with green eyes",
            "relevant": "cat",
            "distractors": ["dog", "phone"],  # dog 是难区分项
        },
        {
            "query": "a golden retriever dog with tongue out",
            "relevant": "dog",
            "distractors": ["cat", "tree"],  # cat 是难区分项
        },
        # --- 组2: 交通工具 vs 建筑 ---
        {
            "query": "a red sports car driving fast",
            "relevant": "car",
            "distractors": ["house", "boat"],
        },
        {
            "query": "a sailboat on the ocean",
            "relevant": "boat",
            "distractors": ["car", "tree"],
        },
        # --- 组3: 自然（tree vs flower）---
        {
            "query": "a large green tree with thick trunk",
            "relevant": "tree",
            "distractors": ["flower", "house"],
        },
        {
            "query": "a beautiful pink flower with petals",
            "relevant": "flower",
            "distractors": ["tree", "cat"],
        },
        # --- 组4: 物品 ---
        {
            "query": "a smartphone with blue screen",
            "relevant": "phone",
            "distractors": ["book", "car"],
        },
        {
            "query": "a book with text on pages",
            "relevant": "book",
            "distractors": ["phone", "house"],
        },
        # --- 组5: 食物 ---
        {
            "query": "a delicious pizza with pepperoni toppings",
            "relevant": "pizza",
            "distractors": ["flower", "cat"],
        },
        # --- 组6: 建筑 ---
        {
            "query": "a house with red roof and chimney",
            "relevant": "house",
            "distractors": ["car", "book"],
        },
    ]

    # Run
    print(f"\n{'='*60}")
    print(f"Running {len(test_cases)} test cases...")
    print(f"{'='*60}")

    results = []
    total_r1, total_r3 = 0, 0

    for i, case in enumerate(test_cases):
        query = case["query"]
        relevant_name = case["relevant"]
        distractor_names = case["distractors"]

        # Build image list
        all_images = [{"url": images[relevant_name], "score": 5, "name": relevant_name}]
        for d in distractor_names:
            all_images.append({"url": images[d], "score": 3, "name": d})

        print(f"\nTest {i+1}/{len(test_cases)}: \"{query}\"")
        print(f"   Candidates: {relevant_name} (✓) vs {', '.join(distractor_names)}")

        # Score
        scored = clip_filter.score_images(all_images, query)

        # Find rank
        relevant_rank = -1
        for rank, img in enumerate(scored):
            if img.get("name") == relevant_name:
                relevant_rank = rank + 1
                break

        r1 = 1 if relevant_rank == 1 else 0
        r3 = 1 if relevant_rank <= 3 else 0
        total_r1 += r1
        total_r3 += r3

        # Show results
        for rank, img in enumerate(scored):
            name = img.get("name", "?")
            score = img.get("clip_score", 0)
            marker = " ✓" if name == relevant_name else ""
            print(f"     #{rank+1} {name:10s} {score:.4f}{marker}")

        relevant_score = next((img.get("clip_score", 0) for img in scored if img.get("name") == relevant_name), 0)
        print(f"   → R@1: {'✅' if r1 else '❌'} | Score: {relevant_score:.4f}")

        results.append({
            "query": query, "relevant": relevant_name,
            "relevant_rank": relevant_rank, "relevant_score": relevant_score,
            "recall_at_1": r1, "recall_at_3": r3,
        })

    # Summary
    total = len(test_cases)
    acc_r1 = total_r1 / total * 100
    acc_r3 = total_r3 / total * 100

    print(f"\n{'='*60}")
    print(f"📊 基准测试结果")
    print(f"{'='*60}")
    print(f"\n测试用例数: {total}")
    print(f"Recall@1: {total_r1}/{total} = {acc_r1:.1f}%")
    print(f"Recall@3: {total_r3}/{total} = {acc_r3:.1f}%")
    print(f"\n设备: {clip_filter.device}")
    print(f"模型加载: {load_time:.2f}s")

    target = 85.0
    if acc_r1 >= target:
        print(f"\n✅ 准确率达标！Recall@1 = {acc_r1:.1f}% ≥ {target}%")
    else:
        print(f"\n⚠️  Recall@1 = {acc_r1:.1f}% < {target}%")

    # Save
    output_path = os.path.join(os.path.dirname(__file__), "clip_benchmark_real.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "openai/clip-vit-base-patch32",
            "device": str(clip_filter.device),
            "load_time": load_time,
            "total": total,
            "recall_at_1": acc_r1,
            "recall_at_3": acc_r3,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n📁 结果: {output_path}")

    # Cleanup
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)
    print("🧹 已清理测试图片")


if __name__ == "__main__":
    run_benchmark()
