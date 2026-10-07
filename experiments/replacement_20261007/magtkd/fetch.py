"""Download released archive or its six feature members with ZIP CRC verification."""
import argparse,hashlib,json,struct,zlib
from pathlib import Path
import requests
URL='https://drive.usercontent.google.com/download?id=19g3hTaBEKF5wXI0DHdvRYbu0BD3XZa3d&export=download&confirm=t'
SIZE=6192026726

def entries():
 r=requests.get(URL+'&range=directory',headers={'Range':'bytes=-65536'},timeout=90);r.raise_for_status()
 if r.status_code!=206: raise RuntimeError('Range unsupported')
 result=[]
 for entry in r.content.split(b'PK\x01\x02')[1:]:
  h=struct.unpack_from('<6H3L5H2L',entry);n,e,c=h[9:12];name=entry[42:42+n].decode();comp,uncomp,offset=h[7],h[8],h[-1]
  extra=entry[42+n:42+n+e];i=0
  while i+4<=len(extra):
   tag,length=struct.unpack_from('<HH',extra,i);p=i+4
   if tag==1:
    if uncomp==0xffffffff:uncomp=struct.unpack_from('<Q',extra,p)[0];p+=8
    if comp==0xffffffff:comp=struct.unpack_from('<Q',extra,p)[0];p+=8
    if offset==0xffffffff:offset=struct.unpack_from('<Q',extra,p)[0]
   i+=4+length
  result.append(dict(name=name,compressed_bytes=comp,uncompressed_bytes=uncomp,offset=offset,compression=h[3],crc=h[6]))
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--archive',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 if a.archive:
  dest=a.output/'MAGTKD.zip';part=dest.with_suffix('.zip.part')
  r=requests.get(URL+'&download=full',stream=True,timeout=(30,120));r.raise_for_status();h=hashlib.sha256();size=0
  with part.open('wb') as f:
   for chunk in r.iter_content(8*1024*1024):f.write(chunk);h.update(chunk);size+=len(chunk);print(size,flush=True)
  if size!=SIZE:raise RuntimeError(f'Archive length {size} != {SIZE}')
  part.rename(dest);(a.output/'archive_manifest.json').write_text(json.dumps({'url':URL,'bytes':size,'sha256':h.hexdigest()},indent=2)+'\n');return
 records=entries();(a.output/'zip_directory.json').write_text(json.dumps(records,indent=2)+'\n')
 for rec in records:
  if 'features.pkl' not in rec['name']:continue
  ds='IEMOCAP' if '/IEMOCAP/' in rec['name'] else 'MELD';dest=a.output/ds/Path(rec['name']).name;dest.parent.mkdir(exist_ok=True)
  if dest.exists() and dest.stat().st_size==rec['uncompressed_bytes']:continue
  start=rec['offset'];end=start+30+len(rec['name'].encode())+65536+rec['compressed_bytes']
  r=requests.get(URL+f'&member={start}',headers={'Range':f'bytes={start}-{end}'},timeout=(30,180));r.raise_for_status()
  if r.status_code!=206 or r.content[:4]!=b'PK\x03\x04':raise RuntimeError('Invalid ZIP range')
  n,e=struct.unpack_from('<HH',r.content,26);body=r.content[30+n+e:30+n+e+rec['compressed_bytes']]
  data=zlib.decompress(body,-15) if rec['compression']==8 else body
  if len(data)!=rec['uncompressed_bytes'] or zlib.crc32(data)!=rec['crc']:raise RuntimeError('ZIP integrity failed')
  dest.write_bytes(data);rec['sha256']=hashlib.sha256(data).hexdigest();print(ds,dest.name,len(data),flush=True)
 (a.output/'feature_manifest.json').write_text(json.dumps([r for r in records if 'features.pkl' in r['name']],indent=2)+'\n')
if __name__=='__main__':main()
