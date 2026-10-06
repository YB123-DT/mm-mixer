"""Download the four original ECERC feature files; no feature substitution."""
import argparse, hashlib, json, re
from html import unescape
from pathlib import Path
import requests
FILES = {
 'iemocap/iemocap_emotion_features_roberta.pkl': '1vbgQ78Pzil6trugnPmgqP8q7j-fAweCj',
 'iemocap/iemocap_emotion_semantic_features_roberta.pkl': '1CQ32t7FPvQU7i0S7h6fGUClPxwvQW9Qv',
 'iemocap/IEMOCAP_features.pkl': '1RV-V6aiUzENUy97XskKqgm_bCJATpapS',
 'meld/meld_emotion_semantic_features_roberta.pkl': '1IQokygWgjaZuSXvCsfIcUzj1Gu69hDCb',
}
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 manifest={}
 for name,file_id in FILES.items():
  dest=args.output/name;dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():
   session=requests.Session();r=session.get('https://drive.google.com/uc',params={'export':'download','id':file_id},stream=True,timeout=90)
   r.raise_for_status()
   if 'text/html' in r.headers.get('content-type',''):
    html=r.text
    action=re.search(r'<form[^>]*action="([^"]+)"',html)
    if not action: raise RuntimeError(html[:500])
    params=dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"',html))
    r=session.get(unescape(action.group(1)),params={k:unescape(v) for k,v in params.items()},stream=True,timeout=90);r.raise_for_status()
   if 'text/html' in r.headers.get('content-type',''):raise RuntimeError(r.text[:500])
   tmp=dest.with_suffix('.part')
   with tmp.open('wb') as out:
    for chunk in r.iter_content(8*1024*1024):out.write(chunk)
   tmp.rename(dest)
  digest=hashlib.sha256()
  with dest.open('rb') as src:
   for chunk in iter(lambda:src.read(8*1024*1024),b''):digest.update(chunk)
  manifest[name]={'google_drive_id':file_id,'size':dest.stat().st_size,'sha256':digest.hexdigest()}
  print(json.dumps({name:manifest[name]}),flush=True)
 (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
