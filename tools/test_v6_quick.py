"""用 v6 模型测试 Downloads 目录中的所有 JSON 谱面"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os, sys, json, pickle, numpy as np
sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor import extract_features
from predict_rpe import convert_rpe_to_standard
from data_loader import load_chart_json

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', '6dim_model_v6.pkl')
with open(MODEL_PATH, 'rb') as f:
    m = pickle.load(f)
gb = m['gb']; scaler = m['scaler']
FN = m['feature_names']; P95 = m['p95_vals']; P99 = m['p99_vals']
FLAT_FEATURES = m['FLAT_FEATURES']
DC = m.get('dynamic_cap', {'knee': 2.5, 'power': 0.9})

def _dynamic_cap(raw):
    KNEE = DC['knee']; POWER = DC['power']
    if raw <= KNEE:
        return raw
    excess = raw - KNEE
    return KNEE + excess ** POWER

def compute_boost(feats):
    total = 0.0
    for fname, bl, co in FLAT_FEATURES:
        v = feats.get(fname, 0)
        pv = P95.get(fname, 0)
        t = max(pv * 0.55, bl * 0.5)
        if v <= t:
            continue
        e = v / t - 1.0
        x = co * (e ** 0.55)
        if v > max(P99.get(fname, 0), bl * 0.5):
            pe = v / max(P99.get(fname, 0), bl * 0.5) - 1.0
            x += co * max(0, pe) ** 0.55 * 0.5
        total += x
    return _dynamic_cap(total)

def load_chart_any(fp):
    """自动识别标准/RPE格式并加载"""
    with open(fp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        if 'META' in data and 'RPEVersion' in data.get('META', {}):
            data = convert_rpe_to_standard(data)
        return data
    return data

DOWNLOADS = r'C:\Users\NaNK\Downloads'
json_files = sorted([f for f in os.listdir(DOWNLOADS) if f.endswith('.json')])

print('='*70)
print(f'  v6 模型测试 — {len(json_files)} 张测试谱面')
print('='*70)

# 从文件名解析预估定数
import re
def parse_est_diff(filename):
    """从文件名中解析预估定数，如 (17.1), (18.0), (16.3~16.5)"""
    # 取第一个匹配的括号数字（排除(1)这种后缀）
    matches = re.findall(r'\((\d+\.?\d*(?:~\d+\.?\d*)?)\)', filename)
    for m in matches:
        parts = m.split('~')
        nums = [float(p) for p in parts]
        avg = sum(nums) / len(nums)
        if avg > 1.5:  # 排除后缀(1)
            return avg
    return None

results = []
for fn in json_files:
    fp = os.path.join(DOWNLOADS, fn)
    try:
        cd = load_chart_any(fp)
    except Exception as e:
        # 尝试PE格式
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                raw = f.read()
            from app import parse_pe_format
            cd = parse_pe_format(raw)
        except Exception as e2:
            print(f'  解析失败: {fn} - {e2}')
            continue

    feats = extract_features(cd)
    if not feats:
        print(f'  特征提取失败: {fn}')
        continue

    x = np.array([[feats.get(n, 0) for n in FN]])
    xs = scaler.transform(x)
    p_gb = float(gb.predict(xs)[0])
    p_b = compute_boost(feats)
    p_f = p_gb + p_b

    est_diff = parse_est_diff(fn)
    results.append({
        'file': fn,
        'prediction': p_f,
        'gb': p_gb,
        'boost': p_b,
        'est_diff': est_diff,
        'total_notes': feats.get('total_notes', 0),
        'duration_sec': feats.get('duration_sec', 0),
        'real_core_tps': feats.get('real_core_notes_per_second', 0),
        'stair_density': feats.get('stair_density', 0),
        'trill_density': feats.get('trill_density', 0),
        'jack_density': feats.get('jack_density', 0),
        'pattern_switch_rate': feats.get('pattern_switch_rate', 0),
        'position_cluster_count': feats.get('position_cluster_count', 0),
        'track_deviation_score': feats.get('track_deviation_score', 0),
        'chord_size_entropy': feats.get('chord_size_entropy', 0),
    })

# 按预测难度降序
results.sort(key=lambda r: -r['prediction'])

# 打印结果
print(f'\n{"文件名":<45} {"预测定数":>8} {"GB":>7} {"Boost":>7} {"预估":>7} {"误差":>7} {"物量":>5} {"时长":>6}')
print('-'*100)
for r in results:
    est_str = f'{r["est_diff"]:.1f}' if r['est_diff'] is not None else '-'
    diff_str = '' if r['est_diff'] is None else f'{r["prediction"]-r["est_diff"]:+.2f}'
    short_name = r['file'][:43] if len(r['file']) > 44 else r['file']
    print(f'{short_name:<45} {r["prediction"]:>8.2f} {r["gb"]:>7.2f} {r["boost"]:>7.2f} {est_str:>7} {diff_str:>7} {r["total_notes"]:>5} {r["duration_sec"]:>6.1f}')

# 打印配置维度详情
print('\n' + '='*70)
print('  配置维度详情')
print('='*70)
print(f'\n{"文件名":<45} {"楼梯密度":>8} {"颤音密度":>8} {"纵连密度":>8} {"和弦熵":>7} {"聚类数":>6} {"离轨度":>7} {"型切换":>7}')
print('-'*105)
for r in results:
    short_name = r['file'][:43] if len(r['file']) > 44 else r['file']
    print(f'{short_name:<45} {r["stair_density"]:>8.2f} {r["trill_density"]:>8.2f} {r["jack_density"]:>8.2f} {r["chord_size_entropy"]:>7.2f} {r["position_cluster_count"]:>6.0f} {r["track_deviation_score"]:>7.3f} {r["pattern_switch_rate"]:>7.3f}')

# 如果有预估定数，计算统计
valid = [r for r in results if r['est_diff'] is not None]
if valid:
    errors = [abs(r['prediction'] - r['est_diff']) for r in valid]
    print(f'\n与文件名预估定数对比 (n={len(valid)})')
    print(f'  平均误差: {np.mean(errors):.2f}')
    print(f'  最大误差: {np.max(errors):.2f}')
    print(f'  误差标准差: {np.std(errors):.2f}')

print(f'\n总计: {len(results)} 张测试谱面')
print('='*70)