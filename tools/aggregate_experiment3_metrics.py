#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math, statistics, struct
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT=Path('/root/project')
DATA_ROOT=ROOT/'data'
TOOLS_ROOT=Path(__file__).resolve().parent
OUT=TOOLS_ROOT/'experiment3_tables'
TAG='recon_major_3_double_single_gpu'
DATASET_NAMES={'360_v2':'Mip-NeRF 360','tank_temples':'Tanks and Temples'}
METHOD_ALIASES={'original':'3DGS (original frames)','ratio_0.50':'REAKS-3DGS'}
BASE='original'; REAKS='ratio_0.50'


def num(v):
    if v in (None,'','NA'): return None
    try:
        x=float(v)
        return None if math.isnan(x) else x
    except Exception:
        return None

def read_json(p:Path):
    if not p.exists(): return {}
    try: return json.loads(p.read_text())
    except Exception: return {}

def read_csv(p:Path):
    if not p.exists(): return []
    with p.open(newline='') as f: return list(csv.DictReader(f))

def write_json(p:Path,obj): p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

def latest_val(stats:Path):
    files=sorted(stats.glob('val_step*.json'))
    return read_json(files[-1]) if files else {}

def stage_sec(rows, stage):
    vals=[]
    for r in rows:
        if r.get('stage')==stage and r.get('status') in {'ok','skipped','not_applicable','dry_run'}:
            x=num(r.get('elapsed_sec'))
            if x is not None: vals.append(x)
    return sum(vals) if vals else None

def timing_prefix(t,prefix):
    vals=[]
    for k,v in t.items():
        if k.startswith(prefix) and isinstance(v,dict):
            x=num(v.get('elapsed_sec'))
            if x is not None: vals.append(x)
    return sum(vals) if vals else None

def reaks_sec(run, stage):
    vals=[num(r.get('elapsed_sec')) for r in read_csv(run/'reaks'/'reaks_stage_timings.csv') if r.get('stage')==stage]
    vals=[v for v in vals if v is not None]
    return sum(vals) if vals else None

def points3d(p:Path):
    if not p.exists(): return {'sfm_points':None,'sfm_points_k':None,'sfm_reproj_error_px':None}
    errs=[]
    try:
        with p.open('rb') as f:
            raw=f.read(8)
            if len(raw)!=8: return {'sfm_points':0,'sfm_points_k':0.0,'sfm_reproj_error_px':None}
            n=struct.unpack('<Q',raw)[0]
            for _ in range(n):
                h=f.read(43)
                if len(h)!=43: break
                errs.append(float(struct.unpack('<QdddBBBd',h)[-1]))
                tl=f.read(8)
                if len(tl)!=8: break
                f.seek(struct.unpack('<Q',tl)[0]*8,1)
    except Exception:
        return {'sfm_points':None,'sfm_points_k':None,'sfm_reproj_error_px':None}
    return {'sfm_points':len(errs),'sfm_points_k':len(errs)/1000.0,'sfm_reproj_error_px':statistics.mean(errs) if errs else None}

def qvec2rotmat(qvec):
    q0,q1,q2,q3=qvec
    return (
        (1-2*q2*q2-2*q3*q3, 2*q1*q2-2*q0*q3, 2*q3*q1+2*q0*q2),
        (2*q1*q2+2*q0*q3, 1-2*q1*q1-2*q3*q3, 2*q2*q3-2*q0*q1),
        (2*q3*q1-2*q0*q2, 2*q2*q3+2*q0*q1, 1-2*q1*q1-2*q2*q2),
    )

def camera_centers_from_images_bin(p:Path):
    centers=[]
    if not p.exists(): return centers
    try:
        with p.open('rb') as f:
            raw=f.read(8)
            if len(raw)!=8: return centers
            n=struct.unpack('<Q',raw)[0]
            for _ in range(n):
                h=f.read(64)
                if len(h)!=64: break
                vals=struct.unpack('<idddddddi',h)
                qvec=vals[1:5]; tvec=vals[5:8]
                R=qvec2rotmat(qvec)
                center=[]
                for i in range(3):
                    center.append(-(R[0][i]*tvec[0]+R[1][i]*tvec[1]+R[2][i]*tvec[2]))
                centers.append(tuple(center))
                while True:
                    c=f.read(1)
                    if not c or c==b'\x00': break
                n2_raw=f.read(8)
                if len(n2_raw)!=8: break
                f.seek(struct.unpack('<Q',n2_raw)[0]*24,1)
    except Exception:
        return []
    return centers

def normalized_entropy_from_images_bin(p:Path, sectors:int=24):
    centers=camera_centers_from_images_bin(p)
    if not centers:
        return {'ne':None,'ne_sectors':sectors,'ne_registered_images':0}
    cx=sum(c[0] for c in centers)/len(centers)
    cy=sum(c[1] for c in centers)/len(centers)
    counts=[0]*sectors
    used=0
    for x,y,_ in centers:
        dx=x-cx; dy=y-cy
        if abs(dx)<1e-12 and abs(dy)<1e-12:
            continue
        angle=math.atan2(dy,dx)%(2*math.pi)
        idx=min(int(angle/(2*math.pi)*sectors),sectors-1)
        counts[idx]+=1; used+=1
    if used==0:
        return {'ne':0.0,'ne_sectors':sectors,'ne_registered_images':len(centers)}
    ent=0.0
    for c in counts:
        if c:
            prob=c/used
            ent-=prob*math.log(prob)
    return {'ne':ent/math.log(sectors),'ne_sectors':sectors,'ne_registered_images':len(centers)}

def count_images(run):
    d=run/'images'
    return len([p for p in d.iterdir() if p.is_file() or p.is_symlink()]) if d.exists() else 0

def status(rows, has_metrics):
    if any(r.get('status')=='failed' for r in rows): return 'failed'
    if has_metrics: return 'ok'
    return 'partial' if rows else 'missing'

def run_dirs(): return sorted(DATA_ROOT.glob(f'*/*/{TAG}/*/seed_*'))

def parse(run:Path):
    method_raw=run.parent.name; seed=run.name.replace('seed_','')
    scene_dir=run.parents[2]; dataset_raw=scene_dir.parent.name; scene=scene_dir.name
    dataset=DATASET_NAMES.get(dataset_raw,dataset_raw); method=METHOD_ALIASES.get(method_raw,method_raw)
    ratio=method_raw.replace('ratio_','') if method_raw.startswith('ratio_') else ('0.50' if method_raw=='original' else None)
    rows=read_csv(run/'stage_timings.csv'); timings=read_json(run/'results'/'stats'/'pipeline_timings.json'); metrics=latest_val(run/'results'/'stats')
    meta=read_json(run/'reaks'/'reaks_timing_metadata.json') or read_json(run/'selection_metadata.json')
    is_reaks=method_raw.startswith('ratio_')
    rf=rg=rc=rk=None
    if is_reaks:
        rf=reaks_sec(run,'feature_extraction'); rg=reaks_sec(run,'similarity_graph'); rc=reaks_sec(run,'spectral_clustering'); rk=reaks_sec(run,'keyframe_selection')
    prep=sum(v for v in [rf,rg,rc,rk] if v is not None) if is_reaks else None
    col_f=stage_sec(rows,'colmap_feature_extraction'); col_m=stage_sec(rows,'colmap_feature_matching'); col_map=stage_sec(rows,'colmap_mapping_sfm')
    col_vals=[v for v in [col_f,col_m,col_map] if v is not None]; col_total=sum(col_vals) if col_vals else None
    train=timing_prefix(timings,'3dgs_training') or stage_sec(rows,'3dgs_training_command')
    render_parts=[timing_prefix(timings,'rendering_all_step'), timing_prefix(timings,'trajectory_rendering_step')]
    rendering=sum(v for v in render_parts if v is not None) if any(v is not None for v in render_parts) else None
    eval_t=timing_prefix(timings,'evaluation_val_step')
    total_parts=([prep] if is_reaks else [])+[col_total,train,rendering,eval_t]
    total=sum(v for v in total_parts if v is not None) if any(v is not None for v in total_parts) else None
    inp=meta.get('input_count'); sel=meta.get('selected_count')
    if sel in (None,''): sel=count_images(run)
    if inp in (None,''):
        try: inp=round(float(sel)/float(ratio)) if ratio else sel
        except Exception: inp=sel
    try: retention=float(sel)/float(inp) if float(inp) else None
    except Exception: retention=None
    setup=timings.get('setup') or {}; tt=timings.get('3dgs_training') or {}; sfm=points3d(run/'sparse'/'0'/'points3D.bin'); ne=normalized_entropy_from_images_bin(run/'sparse'/'0'/'images.bin')
    return {'dataset':dataset,'dataset_raw':dataset_raw,'scene':scene,'method':method,'method_raw':method_raw,'seed':seed,'status':status(rows,bool(metrics)),'run_dir':str(run),
            'input_images':inp,'selected_images':sel,'retention_ratio':retention,'psnr':metrics.get('psnr'),'ssim':metrics.get('ssim'),'lpips':metrics.get('lpips'),'render_time_per_image_sec':metrics.get('ellipse_time'),'num_gaussians':metrics.get('num_GS'),**sfm,**ne,
            'reaks_feature_extraction_sec':rf,'reaks_similarity_graph_sec':rg,'reaks_clustering_sec':rc,'reaks_keyframe_selection_sec':rk,'reaks_preprocessing_time_sec':prep,
            'colmap_feature_extraction_sec':col_f,'colmap_matching_sec':col_m,'colmap_mapping_sec':col_map,'colmap_sfm_time_sec':col_total,'training_time_sec':train,'rendering_time_sec':rendering,'evaluation_time_sec':eval_t,'total_pipeline_time_sec':total,
            'max_gpu_mem_gb':tt.get('max_gpu_mem_gb'),'hardware':setup.get('hardware'),'max_steps':setup.get('max_steps'),'data_factor':setup.get('data_factor'),'random_seed':tt.get('seed') or setup.get('seed') or seed,
            'colmap_config':'single_camera=1; camera_model=OPENCV; SiftExtraction.use_gpu=1; SiftMatching.use_gpu=1; Mapper.ba_global_function_tolerance=1e-6'}

def write_csv(path, rows, fields=None):
    if fields is None:
        fields=[]
        for r in rows:
            for k in r:
                if k not in fields: fields.append(k)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def stats(vals):
    xs=[num(v) for v in vals]; xs=[x for x in xs if x is not None]
    if not xs: return None,None,0,'--'
    m=statistics.mean(xs); s=statistics.stdev(xs) if len(xs)>1 else None
    return m,s,len(xs),(f'{m:.6g} ± {s:.6g}' if s is not None else f'{m:.6g} (n=1)')

def scene_means(rows, metrics):
    groups=defaultdict(list)
    for r in rows: groups[(r['dataset'],r['scene'],r['method'],r['method_raw'])].append(r)
    out=[]
    for key,items in groups.items():
        row={'dataset':key[0],'scene':key[1],'method':key[2],'method_raw':key[3],'scene_run_count':len(items)}
        for m in metrics: row[m]=stats([i.get(m) for i in items])[0]
        out.append(row)
    return out

def group_summary(rows, group_keys, metrics):
    groups=defaultdict(list)
    for r in rows: groups[tuple(r[k] for k in group_keys)].append(r)
    out=[]
    for key,items in groups.items():
        row={k:v for k,v in zip(group_keys,key)}; row['num_runs']=len(items); row['num_scenes']=len({i.get('scene') for i in items if i.get('scene')})
        for m in metrics:
            mean,std,n,text=stats([i.get(m) for i in items])
            row[m+'_mean']=mean; row[m+'_std']=std; row[m+'_n']=n; row[m+'_mean_std']=text
        out.append(row)
    return out

def percentage(rows):
    groups=defaultdict(dict)
    for r in rows: groups[(r['dataset'],r['scene'],r['seed'])][r['method_raw']]=r
    out=[]
    for (dataset,scene,seed),g in groups.items():
        if BASE not in g or REAKS not in g: continue
        b=g[BASE]; rr=g[REAKS]; row={'dataset':dataset,'scene':scene,'seed':seed}
        for m in ['psnr','ssim','lpips','ne','sfm_reproj_error_px','sfm_points_k','colmap_sfm_time_sec','training_time_sec','rendering_time_sec','evaluation_time_sec','total_pipeline_time_sec','max_gpu_mem_gb']:
            bv=num(b.get(m)); rv=num(rr.get(m)); row[m+'_baseline']=bv; row[m+'_reaks']=rv
            if bv is None or rv is None or bv==0: row[m+'_percent_change']=None
            elif m in {'psnr','ssim','ne'}: row[m+'_percent_change']=100*(rv-bv)/bv
            else: row[m+'_percent_change']=100*(bv-rv)/bv
        row['delta_PSNR_dB']=None if row.get('psnr_baseline') is None or row.get('psnr_reaks') is None else row['psnr_reaks']-row['psnr_baseline']
        out.append(row)
    return out

def missing_report(rows, fields):
    out=[]
    for r in rows:
        for f in fields:
            if r.get(f) in (None,''):
                out.append({'dataset':r['dataset'],'scene':r['scene'],'method':r['method'],'seed':r['seed'],'field':f,'run_dir':r['run_dir']})
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    runs=run_dirs(); rows=[parse(r) for r in runs]
    ok=[r for r in rows if r['status']=='ok']
    fields=list(rows[0].keys()) if rows else []
    metrics=['psnr','ssim','lpips','ne','sfm_reproj_error_px','sfm_points_k','selected_images','retention_ratio','max_gpu_mem_gb','reaks_preprocessing_time_sec','colmap_sfm_time_sec','training_time_sec','rendering_time_sec','evaluation_time_sec','total_pipeline_time_sec']
    pipeline_metrics=['reaks_feature_extraction_sec','reaks_similarity_graph_sec','reaks_clustering_sec','reaks_keyframe_selection_sec','reaks_preprocessing_time_sec','colmap_sfm_time_sec','training_time_sec','rendering_time_sec','evaluation_time_sec','total_pipeline_time_sec','max_gpu_mem_gb']
    scenes=scene_means(ok,sorted(set(metrics+pipeline_metrics)))
    ds=group_summary(scenes,['dataset','method','method_raw'],metrics)
    pipe=group_summary(scenes,['dataset','method','method_raw'],pipeline_metrics)
    ne_summary=group_summary(scenes,['dataset','method','method_raw'],['ne'])
    ch=percentage(ok)
    ne_rows=[{k:r.get(k) for k in ['dataset','dataset_raw','scene','method','method_raw','seed','status','input_images','selected_images','retention_ratio','ne','ne_sectors','ne_registered_images','run_dir']} for r in rows]
    miss=missing_report(rows,['psnr','ssim','lpips','ne','sfm_reproj_error_px','sfm_points_k','selected_images','retention_ratio','colmap_sfm_time_sec','training_time_sec','rendering_time_sec','evaluation_time_sec','total_pipeline_time_sec','max_gpu_mem_gb','max_steps','random_seed','hardware'])
    write_csv(OUT/'experiment3_per_scene_metrics.csv',rows,fields); write_json(OUT/'experiment3_per_scene_metrics.json',rows)
    write_csv(OUT/'experiment3_dataset_summary.csv',ds); write_json(OUT/'experiment3_dataset_summary.json',ds)
    write_csv(OUT/'experiment3_pipeline_time.csv',pipe); write_json(OUT/'experiment3_pipeline_time.json',pipe)
    write_csv(OUT/'experiment3_percentage_changes.csv',ch); write_json(OUT/'experiment3_percentage_changes.json',ch)
    write_csv(OUT/'experiment3_ne_metrics.csv',ne_rows); write_json(OUT/'experiment3_ne_metrics.json',ne_rows)
    write_csv(OUT/'experiment3_ne_summary.csv',ne_summary); write_json(OUT/'experiment3_ne_summary.json',ne_summary)
    write_csv(OUT/'missing_fields_report.csv',miss); (OUT/'missing_fields_report.md').write_text('# Missing fields report\n\nMissing entries: '+str(len(miss))+'\n')
    notes=['# Summary notes','',f'Parsed run rows: {len(rows)}',f'Completed rows: {len(ok)}']
    for r in rows:
        if r['status']!='ok': notes.append(f"Failed/partial: {r['dataset']} / {r['scene']} / {r['method']} / seed {r['seed']}: {r['status']}")
    (OUT/'summary_notes.md').write_text('\n'.join(notes)+'\n')
    (OUT/'README_experiment3_tables.md').write_text('Per-scene metrics and third-reviewer timing tables. Main file: experiment3_per_scene_metrics.csv\n')
    print('Parsed run rows:',len(rows)); print('Completed rows:',len(ok)); print('Output:',OUT)
    print('Per-scene CSV:',OUT/'experiment3_per_scene_metrics.csv')
    print('Dataset summary CSV:',OUT/'experiment3_dataset_summary.csv')
    print('Pipeline time CSV:',OUT/'experiment3_pipeline_time.csv')
    print('NE CSV:',OUT/'experiment3_ne_metrics.csv')
    print('NE summary CSV:',OUT/'experiment3_ne_summary.csv')

if __name__=='__main__': main()
