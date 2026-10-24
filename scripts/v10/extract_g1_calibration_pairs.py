from pathlib import Path
import sqlite3,tarfile,time

root=Path(__file__).resolve().parents[2]
db=root/'data/v10/manifests/v10_process.sqlite'
with sqlite3.connect(db) as con:
    source={r[0].replace('soma_uniform/bvh/','g1/csv/').replace('.bvh','.csv')
            for r in con.execute("SELECT source_path FROM compact_clips")}
for f in (root/'data/v10/accepted').glob('*.npz'):
    day,stem=f.stem.split('__',1)
    source.add(f'g1/csv/{day}/{stem}.csv')
out=root/'data/v10/fast_g1/calibration_csv'
out.mkdir(parents=True,exist_ok=True)
start=time.monotonic();found=0
with tarfile.open(root/'data/v10/downloads/g1.tar.gz',mode='r|gz') as tar:
    for member in tar:
        if not member.isfile() or member.name not in source:continue
        payload=tar.extractfile(member)
        if payload is None:continue
        target=out/(member.name.split('/')[-2]+'__'+Path(member.name).name)
        target.write_bytes(payload.read());found+=1
        if found%20==0:print(found,len(source),round(time.monotonic()-start,1),flush=True)
        if found==len(source):break
print('found',found,'requested',len(source),'elapsed',time.monotonic()-start,flush=True)
