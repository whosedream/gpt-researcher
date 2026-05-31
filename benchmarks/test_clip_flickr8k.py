"""
CLIP 图文关联准确率 — Flickr30k 真实数据集基准测试

测试方法（Image-to-Text Retrieval）：
- 每张图片有 5 个 caption（人工标注）
- 取 1 个 caption 作为 query，用 CLIP 从 N 张图片中检索出正确图片
- 衡量 Recall@1 / Recall@5 / Recall@10
- 标准 benchmark，可与论文结果对比

参考：CLIP 原论文在 Flickr30k 上的 zero-shot Recall@1 = 88.0%
"""

import sys
import os
import time
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global Python path
GLOBAL_PYTHON = "/c/Users/hzj/AppData/Local/Programs/Python/Python312/python.exe"


def run_benchmark():
    print("=" * 60)
    print("CLIP 图文关联基准测试 — Flickr30k 真实数据集")
    print("=" * 60)
    print()

    # Import
    try:
        from gpt_researcher.multimodal.clip_filter import CLIPImageFilter
        import torch
        from datasets import load_dataset
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return

    # Load CLIP
    print("📦 Loading CLIP model...")
    start = time.time()
    clip_filter = CLIPImageFilter(device="cuda")
    load_time = time.time() - start

    if not clip_filter.available:
        print("❌ CLIP not available")
        return
    print(f"✅ CLIP loaded in {load_time:.2f}s (device: {clip_filter.device})")

    # Load Flickr30k test split
    print("\n📥 Loading Flickr30k test split...")
    ds = load_dataset("lmms-lab/flickr30k", split="test", streaming=True)

    # Collect image-caption pairs
    # Flickr30k format: each row has image + caption (list of 5 strings)
    print("📊 Collecting image-caption pairs...")
    image_captions = {}  # img_idx -> {image, captions[]}
    count = 0
    for example in ds:
        count += 1
        if count > 1000:  # Limit for speed
            break

        img = example.get("image")
        if img is None:
            continue

        # caption is a list of 5 strings
        captions = example.get("caption", [])
        if isinstance(captions, str):
            captions = [captions]
        if not captions:
            continue

        img_id = f"img_{count}"
        image_captions[img_id] = {"image": img, "captions": captions}

    print(f"   Collected {len(image_captions)} unique images with captions")

    # Filter to images with at least 1 caption
    valid_images = {k: v for k, v in image_captions.items() if len(v["captions"]) >= 1}
    print(f"   Valid images: {len(valid_images)}")

    if len(valid_images) < 10:
        print("❌ Not enough images for benchmark")
        return

    # --- Multi-scale benchmark: 10 / 100 / 1000 candidates ---
    img_ids = list(valid_images.keys())
    random.seed(42)
    N_QUERIES = 50

    # Pre-encode ALL images for efficiency
    print(f"\n⏳ Pre-encoding {len(img_ids)} images...")
    encode_start = time.time()
    all_img_embeddings = {}  # img_id -> tensor
    for idx, img_id in enumerate(img_ids):
        img = valid_images[img_id]["image"]
        img = img.convert("RGB") if img.mode != "RGB" else img
        emb = clip_filter._encode_image(img)
        if emb is not None:
            all_img_embeddings[img_id] = emb
        if (idx + 1) % 200 == 0:
            print(f"   Encoded {idx+1}/{len(img_ids)} images...")
    encode_time = time.time() - encode_start
    print(f"   ✅ Encoded {len(all_img_embeddings)} images in {encode_time:.1f}s")

    all_results = {}

    for N_CANDIDATES in [10, 100, 1000]:
        if N_CANDIDATES > len(all_img_embeddings):
            print(f"\n⚠️  Skipping {N_CANDIDATES} candidates (only {len(all_img_embeddings)} images available)")
            continue

        print(f"\n{'='*60}")
        print(f"🔬 Test: Image Retrieval (1 vs {N_CANDIDATES-1})")
        print(f"   {N_QUERIES} queries, {N_CANDIDATES} candidates per query")
        print(f"{'='*60}")

        r1_count = 0
        r3_count = 0
        r5_count = 0
        r10_count = 0
        scores_list = []
        usable_ids = list(all_img_embeddings.keys())

        for qi in range(N_QUERIES):
            target_id = random.choice(usable_ids)
            target_caption = random.choice(valid_images[target_id]["captions"])

            # Pick N-1 distractors
            distractor_ids = [x for x in usable_ids if x != target_id]
            distractor_ids = random.sample(distractor_ids, min(N_CANDIDATES - 1, len(distractor_ids)))

            # All candidates
            candidate_ids = [target_id] + distractor_ids
            random.shuffle(candidate_ids)

            # Encode query
            txt_emb = clip_filter._encode_text(target_caption)
            if txt_emb is None:
                continue

            # Score all candidates using pre-computed embeddings
            scored = []
            for cid in candidate_ids:
                emb = all_img_embeddings.get(cid)
                if emb is None:
                    scored.append((float("-inf"), cid == target_id))
                else:
                    sim = torch.nn.functional.cosine_similarity(emb, txt_emb).item()
                    scored.append((sim, cid == target_id))

            scored.sort(key=lambda x: x[0], reverse=True)

            # Find rank
            correct_rank = -1
            for rank, (score, is_correct) in enumerate(scored):
                if is_correct:
                    correct_rank = rank + 1
                    break

            if correct_rank == 1: r1_count += 1
            if correct_rank <= 3: r3_count += 1
            if correct_rank <= 5: r5_count += 1
            if correct_rank <= 10: r10_count += 1

            correct_score = next((s for s, c in scored if c), 0)
            scores_list.append(correct_score)

            if (qi + 1) % 10 == 0:
                print(f"   {qi+1}/{N_QUERIES} | R@1: {r1_count}/{qi+1}={r1_count/(qi+1)*100:.0f}%")

        r1_acc = r1_count / N_QUERIES * 100
        r3_acc = r3_count / N_QUERIES * 100
        r5_acc = r5_count / N_QUERIES * 100
        r10_acc = r10_count / N_QUERIES * 100
        avg_score = sum(scores_list) / len(scores_list) if scores_list else 0

        print(f"\n   Results ({N_CANDIDATES} candidates):")
        print(f"   Recall@1:  {r1_count}/{N_QUERIES} = {r1_acc:.1f}%")
        print(f"   Recall@3:  {r3_count}/{N_QUERIES} = {r3_acc:.1f}%")
        print(f"   Recall@5:  {r5_count}/{N_QUERIES} = {r5_acc:.1f}%")
        print(f"   Recall@10: {r10_count}/{N_QUERIES} = {r10_acc:.1f}%")
        print(f"   Avg score: {avg_score:.4f}")

        random.seed(42)  # Reset seed for fair comparison

        all_results[f"n{N_CANDIDATES}"] = {
            "n_candidates": N_CANDIDATES,
            "n_queries": N_QUERIES,
            "recall_at_1": r1_acc,
            "recall_at_3": r3_acc,
            "recall_at_5": r5_acc,
            "recall_at_10": r10_acc,
            "avg_score": avg_score,
        }

    # Summary
    print(f"\n{'='*60}")
    print(f"📊 最终结果 — Flickr30k 真实数据集")
    print(f"{'='*60}")
    print(f"\n数据集: Flickr30k (test split)")
    print(f"图片总数: {len(all_img_embeddings)}")
    print(f"查询数: {N_QUERIES}")
    print(f"设备: {clip_filter.device} | 模型加载: {load_time:.1f}s | 图片编码: {encode_time:.1f}s")
    print(f"\n参考: CLIP 论文 Flickr30k zero-shot R@1 = 88.0% (1000 选 1)")
    print()
    print(f"{'候选数':>8} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8}")
    print("-" * 48)
    for key, r in sorted(all_results.items()):
        print(f"{r['n_candidates']:>8} {r['recall_at_1']:>7.1f}% {r['recall_at_3']:>7.1f}% {r['recall_at_5']:>7.1f}% {r['recall_at_10']:>7.1f}%")

    # Save
    output_path = os.path.join(os.path.dirname(__file__), "clip_flickr30k_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "Flickr30k",
            "split": "test",
            "model": "openai/clip-vit-base-patch32",
            "device": str(clip_filter.device),
            "load_time": load_time,
            "encode_time": encode_time,
            "total_images": len(all_img_embeddings),
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n📁 结果: {output_path}")


if __name__ == "__main__":
    run_benchmark()
