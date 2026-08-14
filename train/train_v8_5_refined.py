"""
v8.5 STAT-REFINED: 基于r值验证 + manual curation + hold-out
  删: pattern_switch_rate, drag_flick_ratio, rest_ratio, speed_volatility, above_below_cross,
       stair_speed_avg, position_range_used, fast_note_64th
  保留: fast_note_24th/48th 锁定(极端谱检测), 其余29个r>0.1特征
"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sys, os, pickle, numpy as np, math, re, random
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, '.')
from feature_extractor import extract_features
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from unified_parser import load_chart_from_bytes
import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
random.seed(42); np.random.seed(42)
print('='*70); print('  v8.5 STAT-REFINED — 31特征 + hold-out验证'); print('='*70)

# ===== 1. 加载 =====
CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
diff_map = load_difficulty_tsv(TSV)
chart_files = find_chart_files(CHART_DIR)
all_official = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in diff_map: continue
    for lv in ['IN','AT']:
        if lv not in info.get('levels',{}): continue
        if lv not in diff_map[sid]: continue
        try:
            cd = load_chart_json(info['levels'][lv]); f = extract_features(cd)
            if f: f['_difficulty'] = diff_map[sid][lv]; f['_name'] = fn[:30]; all_official.append(f)
        except: pass
# 排除新谱
exclude = ['chartnekockLK','snow dance','Snow']
all_official = [f for f in all_official if not any(p.lower() in f['_name'].lower() for p in exclude)]
print(f'官谱: {len(all_official)}')

# ===== 2. STAT-REFINED FLAT (r>0.1, 无负相关) =====
FLAT = [
    # 密度 (r>0.5) 
    ('density_dimension',              1.0,    0.35),
    ('above_avg_density_mean',         4.0,    0.12),
    ('above_avg_duration_sec',         30.0,   0.04),
    ('real_core_notes_per_second',     2.0,    0.10),
    # 快音符 (16th/32nd locked, 24th/48th locked for extreme detection)
    ('fast_note_density_16th',         4.0,    0.08),   # r=0.52 LOCKED
    ('fast_note_density_32nd',         2.0,    0.15),   # r=0.18 LOCKED
    ('fast_note_density_24th',         1.0,    0.10),   # r=0.09 LOCKED (极端谱)
    ('fast_note_density_48th',         0.5,    0.12),   # r=0.09 LOCKED (极端谱)
    ('rhythm_type_count',              3.0,    0.08),   # r=0.42
    # 配置 (r>0.2)
    ('stair_rate_per_sec',             5.0,    0.04),   # r=0.65
    ('stair_complexity',               0.2,    0.02),
    ('stair_chord_ratio',              0.3,    0.02),
    ('trill_density',                  2.0,    0.02),
    ('chord_size_entropy',             0.5,    0.02),
    ('chord_alternation_rate',         0.5,    0.08),   # r=0.67
    ('weighted_mf_score_per_sec',      10.0,   0.06),   # r=0.53
    ('position_entropy',               2.0,    0.02),
    ('avg_chord_size_poly',            2.0,    0.04),
    ('jack_density',                   4.0,    0.02),
    # 耐力 (r>0.15)
    ('tap_per_second',                 2.5,    0.25),   # r=0.66
    ('total_notes',                    400.0,  0.15),   # r=0.77
    ('duration_sec',                   100.0,  0.03),   # r=0.15
    ('tap_burst_top5',                 0.5,    0.04),   # r=0.45
    # 读谱 (r>0.1)
    ('tempo_change_count',             50.0,   0.04),   # r=0.73
    ('type_switch_per_sec',            0.4,    0.06),   # r=0.53
    ('density_transition_mean',        0.15,   0.03),   # r=0.50
    ('density_transition_std',         0.2,    0.04),   # r=0.54
    ('note_clutter_ratio',             0.05,   0.04),   # r=0.40
    ('rhythm_entropy',                 2.5,    0.03),   # r=0.17
    ('hold_interference_index',        0.3,    0.03),   # r=0.10
    ('jline_movement_density',         50.0,   0.04),   # r=0.40
    ('jline_rotate_density',           20.0,   0.03),   # r=0.23
    ('jline_disappear_density',        20.0,   0.03),   # r=0.28
]

PINNED = {'fast_note_density_16th':0.08,'fast_note_density_32nd':0.15,
          'fast_note_density_24th':0.10,'fast_note_density_48th':0.12}
pinned_indices = [i for i,(n,_,_) in enumerate(FLAT) if n in PINNED]
free_indices = [i for i in range(len(FLAT)) if i not in pinned_indices]
feat_names_boost = [n for n,_,_ in FLAT]
print(f'FLAT: {len(FLAT)} ({len(pinned_indices)} locked, {len(free_indices)} free)')

# P95/P99
P95={}; P99={}
for fn in feat_names_boost:
    vals=[f.get(fn,0) for f in all_official]
    P95[fn]=float(np.percentile(vals,95)) if vals else 0
    P99[fn]=float(np.percentile(vals,99)) if vals else 0

DC={'knee':2.5,'power':0.9}

def compute_excess(feats,fname,bl):
    val=feats.get(fname,0); th=max(P95.get(fname,0)*0.55,bl*0.5)
    if val<=th: return 0
    e=(val/th-1)**0.7
    if val>max(P99.get(fname,0),bl*0.5):
        pe=(val/max(P99.get(fname,0),bl*0.5)-1)**0.7; e+=0.5*max(0,pe)**0.7
    return e

def _dc(r):
    if r<=DC['knee']: return r
    return DC['knee']+(r-DC['knee'])**DC['power']

co_current=np.array([c for _,_,c in FLAT])

def compute_raw_boost(feats,co_arr):
    raw=0.0
    for j,(fname,bl,_) in enumerate(FLAT):
        raw+=co_arr[j]*compute_excess(feats,fname,bl)
    return raw

def adjust_boost(boost,gb_val,target=0.28,thresh=0.22,power=0.75):
    if boost<2.0 or gb_val<=0: return boost
    r=boost/gb_val; e=target*gb_val; a=e*((boost/e)**power)
    w=1/(1+math.exp(-25*(r-thresh)))
    return (1-w)*boost+w*a

# ===== 3. hold-out分层抽样 13-17 =====
diffs_all=np.array([f['_difficulty'] for f in all_official])
hold_out_mask=np.zeros(len(all_official),dtype=bool)
bins=np.digitize(diffs_all,bins=[13,14,15,16,17])
for b in range(1,6):
    idx=np.where(bins==b)[0]
    if len(idx)<3: continue
    n_hold=max(1,int(len(idx)*0.25))
    chosen=np.random.choice(idx,size=n_hold,replace=False)
    hold_out_mask[chosen]=True

print(f'Hold-out: {hold_out_mask.sum()}/{len(all_official)} (13-17 range)')

# ===== 4. 训练 =====
feats_train=[all_official[i] for i in range(len(all_official)) if not hold_out_mask[i]]
feats_test=[all_official[i] for i in range(len(all_official)) if hold_out_mask[i]]
y_train=diffs_all[~hold_out_mask]; y_test=diffs_all[hold_out_mask]

FN=list(all_official[0].keys()); FN=[k for k in FN if k not in ('_difficulty','_name')]
X_gb_train=np.array([[f.get(k,0) for k in FN] for f in feats_train])

X_excess_train=np.zeros((len(feats_train),len(FLAT)))
for i in range(len(feats_train)):
    for j,(fname,bl,_) in enumerate(FLAT):
        X_excess_train[i,j]=compute_excess(feats_train[i],fname,bl)

for it in range(3):
    print(f'\n--- Iter {it+1}/3 ---')
    all_boosts=np.array([_dc(compute_raw_boost(feats_train[i],co_current)) for i in range(len(feats_train))])
    sc=StandardScaler()
    gbm=GradientBoostingRegressor(n_estimators=500,max_depth=5,min_samples_leaf=3,
                                   learning_rate=0.05,subsample=0.8,random_state=42)
    gbm.fit(sc.fit_transform(X_gb_train),y_train-all_boosts)
    preds=gbm.predict(sc.transform(X_gb_train))+all_boosts
    print(f'  GB: R²={r2_score(y_train,preds):.4f} MAE={mean_absolute_error(y_train,preds):.4f}')
    
    sca=StandardScaler()
    gb_full=GradientBoostingRegressor(n_estimators=500,max_depth=5,min_samples_leaf=3,
                                       learning_rate=0.05,subsample=0.8,random_state=42)
    gb_full.fit(sca.fit_transform(X_gb_train),y_train-all_boosts)
    y_res=y_train-gb_full.predict(sca.transform(X_gb_train))
    
    X_excess_free=X_excess_train[:,free_indices]
    pinned_contrib=np.zeros(len(feats_train))
    for pi in pinned_indices:
        pinned_contrib+=PINNED[FLAT[pi][0]]*X_excess_train[:,pi]
    y_res_free=y_res-pinned_contrib
    
    best_a=5.0; best_cv=float('inf')
    for a in [0.1,1,5,10,50]:
        ridge=Ridge(alpha=a,fit_intercept=False,positive=True)
        s=-cross_val_score(ridge,X_excess_free,y_res_free,cv=3,scoring='neg_mean_absolute_error')
        if s.mean()<best_cv: best_cv=s.mean(); best_a=a
    ridge=Ridge(alpha=best_a,fit_intercept=False,positive=True)
    ridge.fit(X_excess_free,y_res_free)
    
    co_new=np.zeros(len(FLAT))
    for fi,pi in enumerate(free_indices): co_new[pi]=ridge.coef_[fi]
    for pi in pinned_indices: co_new[pi]=PINNED[FLAT[pi][0]]
    
    for k in ['density_dimension','above_avg_density_mean','tap_per_second','total_notes',
              'fast_note_density_16th','fast_note_density_32nd']:
        if k in feat_names_boost:
            print(f'  {k:<35} co={co_new[feat_names_boost.index(k)]:.4f}')
    co_current=0.3*co_current+0.7*co_new

# ===== 5. 全量 + 评估 =====
all_boosts_f=np.array([_dc(compute_raw_boost(f,co_current)) for f in feats_train])
sc_f=StandardScaler()
gb_f=GradientBoostingRegressor(n_estimators=700,max_depth=5,min_samples_leaf=3,
                                learning_rate=0.05,subsample=0.8,random_state=42)
gb_f.fit(sc_f.fit_transform(X_gb_train),y_train-all_boosts_f)

# Hold-out eval
X_gb_test_ex=np.array([[f.get(k,0) for k in FN] for f in feats_test])
pg_test=gb_f.predict(sc_f.transform(X_gb_test_ex))
pb_test=np.array([_dc(compute_raw_boost(f,co_current)) for f in feats_test])
pa_test=np.array([adjust_boost(pb_test[i],pg_test[i]) for i in range(len(feats_test))])
preds_hold=pg_test+pa_test
mae_hold=mean_absolute_error(y_test,preds_hold)
print(f'\n=== Hold-out MAE: {mae_hold:.4f} ({len(feats_test)} charts) ===')

# ===== 6. 保存 =====
FLAT_F=[(n,b,float(co_current[j])) for j,(n,b,_) in enumerate(FLAT)]
out={'gb':gb_f,'scaler':sc_f,'feature_names':list(FN),
     'p95_vals':P95,'p99_vals':P99,'FLAT_FEATURES':FLAT_F,'dynamic_cap':DC,
     'stat_refined':True,'holdout_mae':mae_hold}
os.makedirs('models',exist_ok=True)
with open('models/6dim_model_v8_5.pkl','wb') as f: pickle.dump(out,f)
print(f'Saved: models/6dim_model_v8_5.pkl')

# ===== 7. 全测试 =====
test_dir=r'C:\Users\NaNK\Downloads'
all_preds=[]
for fn in os.listdir(test_dir):
    if not fn.endswith('.json') or '_2xBPM' in fn: continue
    fp=os.path.join(test_dir,fn)
    if os.path.getsize(fp)<100: continue
    try:
        rating=None
        for m in re.finditer(r'\((\d+\.?\d*)\)',fn):
            v=float(m.group(1))
            if 5<=v<=25: rating=v; break
        with open(fp,'rb') as f: raw=f.read()
        data,_=load_chart_from_bytes(raw); feats=extract_features(data)
        if feats:
            Xx=np.array([[feats.get(k,0) for k in FN]])
            pg=float(gb_f.predict(sc_f.transform(Xx))[0])
            pr=compute_raw_boost(feats,co_current); pb=_dc(pr)
            pa=adjust_boost(pb,pg)
            all_preds.append((fn[:30],rating,pg+pa,feats.get('density_dimension',0)))
    except: pass

all_preds.sort(key=lambda x:-x[2])
print(f'\n{"谱面":<30} {"定数":>5} {"预测":>7} {"密度":>6}')
for name,rating,pred,dd in all_preds:
    rs=f'{rating:.1f}' if rating else ' ?? '
    print(f'{name:<30} {rs:>5} {pred:>7.2f} {dd:>6.1f}')
