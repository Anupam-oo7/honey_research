import fs from 'fs';
import {fileURLToPath} from 'node:url';
const dir=process.argv[2]||'/home/claude/honey_project/results/loop';
const MODELS=JSON.parse(fs.readFileSync(dir+'/models_bundle.json','utf8')), V=JSON.parse(fs.readFileSync(dir+'/parity_vectors.json','utf8'));
const f=new Function('MODELS',fs.readFileSync(fileURLToPath(new URL('core.js',import.meta.url)),'utf8')+';return {buildModel,l1,nbScore,MODELS,seal,newSession,step,metrics,dictionary,randomInstance}');
const C=f(MODELS);let bad=0,n=0;
for(const k of ['card','cred'])for(const ver of ['v1','v2']){const v=V[k][ver],m=C.buildModel({klass:k,ver});
 v.seeds.forEach((s,i)=>{n++;if(m.dec(BigInt(s))!==v.decoded[i]){bad++;if(bad<4)console.log('DECODE MISMATCH',k,ver,m.dec(BigInt(s)),v.decoded[i])}});
 const nb=MODELS[k].nb[ver];v.sample.forEach((s,i)=>{n+=2;if((C.l1(k,s)==='')!==v.l1[i]){bad++;if(bad<4)console.log('L1 MISMATCH',k,s)}
  if(Math.abs(C.nbScore(nb,k,s)-v.nb[i])>1e-9){bad++;if(bad<4)console.log('NB MISMATCH',k,ver,s,C.nbScore(nb,k,s),v.nb[i])}})}
console.log('parity checks',n,'mismatches',bad);
// end-to-end smoke: every scheme, both workloads, both attacker levels
const dict=C.dictionary(150);
for(const k of ['card','cred'])for(const id of ['PBE-GCM','PBE-CBC','PBE-CTR','HE-v1','HE-v2'])for(const lvl of ['l1','nb']){
 const pt=C.randomInstance(k),pw=dict[5],s=C.newSession(await C.seal(id,pt,pw,1000),dict,150,'all',lvl);while(!s.done)await C.step(s);const m=C.metrics(s);
 if(lvl==='nb')console.log(k,id.padEnd(8),'cand',m.candidates,'fpr',(m.fpr??0).toFixed(2),'found',m.found)}
