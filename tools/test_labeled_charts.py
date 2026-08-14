import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
r"""测试 C:\Users\NaNK\Downloads 中有标定数的谱面（v7.3模型）"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from app import predict_one_chart
from unified_parser import load_chart_from_bytes

DOWNLOADS = r'C:\Users\NaNK\Downloads'

# 文件名 → 标定数（从文件名提取）
labeled = {
    'スタートリップ(12.2).json': 12.2,
    'ふたりのスタートボタン(13.4).json': 13.4,
    'トキラキメキ(14.6)(1).json': 14.6,
    '茉子の日常(15.5).json': 15.5,
    'ニャンだふる♡サマー!!(15.8).json': 15.8,
    'Runengon(16.2~16.4).json': 16.3,
    'Breakcore革命前夜(16.3~16.5).json': 16.4,
    'おぎゃりないざー(16.4~16.6).json': 16.5,
    'Lemegeton -little key of solomon-(16.6).json': 16.6,
    '恋ひ恋ふ縁(16.8)(1).json': 16.8,
    'Cheerio!(17.0).json': 17.0,
    'silly-willy-nilly(17.7)(1).json': 17.7,
    'Waking Shadows (feat. Eili)(17.8).json': 17.8,
    'Submerged City(17.8).json': 17.8,
    'Apollo(17.8).json': 17.8,
    'Xaleid◆scopiX(18.2)(1).json': 18.2,
    'ギザバ怪文書(18.3).json': 18.3,
    'The Final EndGame(18.4).json': 18.4,
    '朧月(18.4)(1).json': 18.4,
}

print(f"{'谱面':35s} {'标定':>5s} {'GB':>6s} {'Boost':>6s} {'预测':>6s} {'误差':>6s} {'格式'}")
print("-" * 85)

results = []
for filename, label in labeled.items():
    path = os.path.join(DOWNLOADS, filename)
    if not os.path.exists(path):
        print(f"  [跳过] 文件不存在: {filename}")
        continue

    with open(path, 'rb') as f:
        raw = f.read()

    chart_data, _ = load_chart_from_bytes(raw)
    if chart_data is None:
        # 尝试直接JSON
        chart_data = json.loads(raw.decode('utf-8', errors='replace'))

    result, err = predict_one_chart(chart_data, speed=1.0)
    if err:
        print(f"  [错误] {filename}: {err}")
        continue

    name = result['song_name'] or filename.split('(')[0]
    gb = result['gb']
    boost = result['boost_adj']
    pred = result['prediction']
    err_val = pred - label
    fmt = result['format']
    results.append((label, pred, err_val))

    print(f"{name:35s} {label:>5.1f} {gb:>6.2f} {boost:>6.2f} {pred:>6.2f} {err_val:>+6.2f} {fmt}")

# 统计
if results:
    abs_errs = [abs(r[2]) for r in results]
    mae = sum(abs_errs) / len(abs_errs)
    print(f"\n--- 统计 ({len(results)} 首) ---")
    print(f"MAE: {mae:.3f}")
    print(f"最大误差: {max(abs_errs):.3f}")
    above = sum(1 for r in results if r[1] > r[0])
    below = sum(1 for r in results if r[1] < r[0])
    print(f"偏高: {above}, 偏低: {below}")
