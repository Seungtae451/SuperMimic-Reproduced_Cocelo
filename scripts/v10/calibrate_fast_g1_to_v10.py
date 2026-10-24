import json,sqlite3,sys
from pathlib import Path
import numpy as np
import mujoco

root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'src'))
from scripts.v10.pilot_fast_g1_to_v10 import G1_XML,V10_XML,joint_mapping,load_g1_csv
from supermimic.data.motion_convert import resample_motion

G=mujoco.MjModel.from_xml_path(str(G1_XML));V=mujoco.MjModel.from_xml_path(str(V10_XML))
mapping=joint_mapping(G,V)
db=root/'data/v10/manifests/v10_process.sqlite';stream=root/'data/v10/library/accepted_stream.f16'
with sqlite3.connect(db) as con:
    rows={r[1]:(r[2],r[3],r[4]) for r in con.execute('SELECT source_path,name,start,length,width FROM compact_clips')}
rawstream=np.memmap(stream,dtype=np.float16,mode='r')
records=[]
for src in sorted((root/'data/v10/fast_g1/calibration_csv').glob('*.csv')):
    key=src.stem
    direct=root/'data/v10/accepted'/f'{key}.npz'
    s=resample_motion(load_g1_csv(src.read_bytes()),120,50).joint_pos
    if direct.exists():
        with np.load(direct) as z:t=np.asarray(z['joint_pos'],dtype=np.float64)
    elif key in rows:
        start,length,width=rows[key]
        block=rawstream[start*width:(start+length)*width].reshape(length,width)
        t=np.asarray(block[:,7:7+26],dtype=np.float64)
    else:continue
    if len(s)!=len(t):print('Mismatch',key,len(s),len(t),flush=True);continue
    x=np.zeros_like(t)
    for i,j,sign in mapping:x[:,i]=sign*s[:,j]
    # Equal weight for short and long clips.
    ix=np.linspace(0,len(x)-1,min(200,len(x)),dtype=int)
    records.append((key,x[ix],t[ix]))

rng=np.random.default_rng(42)
order=rng.permutation(len(records))
split=max(1,int(.8*len(order)))
train=[records[i] for i in order[:split]]
test=[records[i] for i in order[split:]]
bounds=V.jnt_range[1:]
def fit(group):
    x=np.concatenate([r[1] for r in group]);y=np.concatenate([r[2] for r in group])
    meanx=x.mean(0);meany=y.mean(0)
    ridge=0.03
    a=(np.mean((x-meanx)*(y-meany),axis=0)+ridge)/(np.var(x,axis=0)+ridge)
    a=np.clip(a,0,1.5)
    b=meany-a*meanx
    return a,b
def stats(group,a,b):
    x=np.concatenate([r[1] for r in group]);y=np.concatenate([r[2] for r in group]);p=x*a+b
    raw=np.sqrt(np.mean((x-y)**2))
    calibrated=np.sqrt(np.mean((np.clip(p,bounds[:,0],bounds[:,1])-y)**2))
    excess=np.maximum(bounds[:,0]-p,p-bounds[:,1]).clip(min=0)
    return {'clips':len(group),'frames':len(x),'raw_joint_rms_rad':float(raw),'calibrated_joint_rms_rad':float(calibrated),'clipped_frame_fraction':float(np.mean(np.any(excess>1e-6,axis=1))),'excess_over_0_1rad_frame_fraction':float(np.mean(np.any(excess>.1,axis=1))),'maximum_excess_rad':float(excess.max())}
a,b=fit(train)
report={'source_clips':len(records),'train':stats(train,a,b),'holdout':stats(test,a,b)}
print(json.dumps(report,indent=2),flush=True)
a,b=fit(records)
mapping_json={
    'method':'linear least-squares joint calibration, equal clip-frame cap, ridge=0.03',
    'source':'BONES-SEED pre-retargeted G1 CSV',
    'target':'Humanoid V10 direct SOMA PyRoki retargets',
    'matched_clips':len(records),
    'validation':report,
    'joints':[{'v10':V.joint(i+1).name,'g1':G.joint(j+1).name,'sign':sign,'scale':float(a[i]),'offset':float(b[i])} for i,j,sign in mapping],
}
out=root/'data/v10/fast_g1/calibration.json'
out.write_text(json.dumps(mapping_json,indent=2)+'\n')
print('Saved',out,flush=True)
