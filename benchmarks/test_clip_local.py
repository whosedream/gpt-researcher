"""
CLIP 图文关联准确率基准测试（本地图片版）
用本地图片文件验证 CLIP 语义匹配能力
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 测试方法：生成本地测试图片，然后用 CLIP 做图文匹配
# 每个测试用例：1 个查询文本 + 3 张图片（1 张相关 + 2 张不相关）


def create_test_images():
    """Create simple colored test images locally."""
    from PIL import Image, ImageDraw, ImageFont

    test_dir = os.path.join(os.path.dirname(__file__), "test_images")
    os.makedirs(test_dir, exist_ok=True)

    images = {}

    # Image 1: Red circle on white background (represents "red apple")
    img = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 50, 174, 174], fill="red")
    path = os.path.join(test_dir, "red_circle.png")
    img.save(path)
    images["red_circle"] = path

    # Image 2: Blue rectangle on white (represents "blue sky")
    img = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 204, 204], fill="blue")
    path = os.path.join(test_dir, "blue_square.png")
    img.save(path)
    images["blue_square"] = path

    # Image 3: Green triangle (represents "nature/leaf")
    img = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(img)
    draw.polygon([(112, 20), (20, 200), (204, 200)], fill="green")
    path = os.path.join(test_dir, "green_triangle.png")
    img.save(path)
    images["green_triangle"] = path

    # Image 4: Yellow star (represents "sun/light")
    img = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(img)
    draw.regular_polygon((112, 112, 80), 5, fill="yellow")
    path = os.path.join(test_dir, "yellow_star.png")
    img.save(path)
    images["yellow_star"] = path

    # Image 5: Black circle (represents "dark/night")
    img = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 50, 174, 174], fill="black")
    path = os.path.join(test_dir, "black_circle.png")
    img.save(path)
    images["black_circle"] = path

    return images


def run_benchmark():
    """Run CLIP accuracy benchmark with local images."""
    print("=" * 60)
    print("CLIP 图文关联准确率基准测试（本地图片版）")
    print("=" * 60)
    print()

    try:
        from gpt_researcher.multimodal.clip_filter import CLIPImageFilter
        print("✅ CLIPImageFilter imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import CLIPImageFilter: {e}")
        return

    # Create test images
    print("🎨 Creating test images...")
    images = create_test_images()
    print(f"   Created {len(images)} test images")
    for name, path in images.items():
        print(f"   - {name}: {os.path.basename(path)}")

    # Initialize CLIP
    print("\n📦 Loading CLIP model...")
    start = time.time()
    clip_filter = CLIPImageFilter()
    load_time = time.time() - start

    if not clip_filter.available:
        print("❌ CLIP model not available.")
        return

    print(f"✅ CLIP model loaded in {load_time:.2f}s (device: {clip_filter.device})")

    # Test cases: query -> expected relevant image
    test_cases = [
        {
            "query": "a red circle",
            "relevant": "red_circle",
            "distractors": ["blue_square", "green_triangle"],
        },
        {
            "query": "a blue square shape",
            "relevant": "blue_square",
            "distractors": ["red_circle", "yellow_star"],
        },
        {
            "query": "a green triangle in nature",
            "relevant": "green_triangle",
            "distractors": ["black_circle", "red_circle"],
        },
        {
            "query": "a yellow star shining bright",
            "relevant": "yellow_star",
            "distractors": ["black_circle", "blue_square"],
        },
        {
            "query": "a dark black circle",
            "relevant": "black_circle",
            "distractors": ["yellow_star", "green_triangle"],
        },
    ]

    # Run tests
    print("\n" + "-" * 60)
    results = []
    total_r1 = 0
    total_r3 = 0

    for i, case in enumerate(test_cases):
        query = case["query"]
        relevant_name = case["relevant"]
        relevant_path = images[relevant_name]
        distractor_paths = [images[d] for d in case["distractors"]]

        # Build image list
        all_images = [{"url": relevant_path, "score": 5, "name": relevant_name}]
        for d_name in case["distractors"]:
            all_images.append({"url": images[d_name], "score": 3, "name": d_name})

        print(f"\nTest {i+1}/{len(test_cases)}: \"{query}\"")
        print(f"   Expected relevant: {relevant_name}")

        # Score with CLIP
        try:
            scored = clip_filter.score_images(all_images, query)
        except Exception as e:
            print(f"   ⚠️  Scoring failed: {e}")
            results.append({"query": query, "error": str(e)})
            continue

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

        top_score = scored[0].get("clip_score", 0) if scored else 0
        relevant_score = next((img.get("clip_score", 0) for img in scored if img.get("name") == relevant_name), 0)

        print(f"   Results:")
        for rank, img in enumerate(scored):
            name = img.get("name", "?")
            score = img.get("clip_score", 0)
            marker = " ← relevant" if name == relevant_name else ""
            print(f"     #{rank+1} {name}: {score:.4f}{marker}")

        print(f"   R@1: {'✅' if r1 else '❌'} | R@3: {'✅' if r3 else '❌'} | Relevant score: {relevant_score:.4f}")

        results.append({
            "query": query,
            "relevant": relevant_name,
            "relevant_rank": relevant_rank,
            "relevant_score": relevant_score,
            "top_score": top_score,
            "recall_at_1": r1,
            "recall_at_3": r3,
        })

    # Summary
    total = len(test_cases)
    acc_r1 = total_r1 / total * 100
    acc_r3 = total_r3 / total * 100

    print("\n" + "=" * 60)
    print("📊 基准测试结果")
    print("=" * 60)
    print(f"\n测试用例数: {total}")
    print(f"Recall@1: {total_r1}/{total} = {acc_r1:.1f}%")
    print(f"Recall@3: {total_r3}/{total} = {acc_r3:.1f}%")
    print(f"\nCLIP 模型: openai/clip-vit-base-patch32")
    print(f"设备: {clip_filter.device}")
    print(f"模型加载时间: {load_time:.2f}s")

    target = 85.0
    if acc_r1 >= target:
        print(f"\n✅ 准确率达标！Recall@1 = {acc_r1:.1f}% ≥ {target}%")
    else:
        print(f"\n⚠️  Recall@1 = {acc_r1:.1f}% < {target}% (目标)")

    # Save
    output_path = os.path.join(os.path.dirname(__file__), "clip_benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "openai/clip-vit-base-patch32",
            "device": str(clip_filter.device),
            "load_time_seconds": load_time,
            "total_cases": total,
            "recall_at_1": acc_r1,
            "recall_at_3": acc_r3,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n📁 结果已保存: {output_path}")

    # Cleanup
    test_dir = os.path.join(os.path.dirname(__file__), "test_images")
    for f in os.listdir(test_dir):
        os.remove(os.path.join(test_dir, f))
    os.rmdir(test_dir)
    print("🧹 测试图片已清理")


if __name__ == "__main__":
    run_benchmark()
