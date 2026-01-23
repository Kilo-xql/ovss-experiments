import os,pickle,lmdb
import numpy as np
from PIL import Image

if __name__=="__main__":
 import argparse
 ap=argparse.ArgumentParser()
 ap.add_argument("--lmdb",required=True)
 ap.add_argument("--out",required=True)
 ap.add_argument("--start",type=int,default=0)
 ap.add_argument("--end",type=int,default=200)
 a=ap.parse_args()
 os.makedirs(a.out+"/images",exist_ok=True)
 os.makedirs(a.out+"/labels",exist_ok=True)
 env=lmdb.open(a.lmdb,readonly=True,lock=False,readahead=False)
 n=0
 with env.begin(write=False) as txn:
  cur=txn.cursor()
  i=0
  for k,v in cur:
   if i>=a.end: break
   if i<a.start: i+=1; continue
   obj=pickle.loads(v)
   img_b, img_s, m_b, m_s = obj[0], obj[1], obj[2], obj[3]
   img=np.frombuffer(img_b,dtype=np.uint8).reshape(img_s).transpose(1,2,0)
   img=img[..., ::-1]
   m=np.frombuffer(m_b,dtype=np.uint8).reshape(m_s)
   stem=f"{i:05d}"
   Image.fromarray(img,"RGB").save(f"{a.out}/images/{stem}.png")
   Image.fromarray(m,"L").save(f"{a.out}/labels/{stem}.png")
   n+=1; i+=1
 print("Exported",n,"to",a.out)
