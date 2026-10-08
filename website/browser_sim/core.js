// ---------- core (no DOM): real crypto via WebCrypto, DTE with BigInt ----------
const TE=new TextEncoder(),L1=new TextDecoder('latin1'),S_=crypto.subtle;
const hex=u=>[...u].map(b=>b.toString(16).padStart(2,'0')).join('');
const rnd=n=>crypto.getRandomValues(new Uint8Array(n));
const cat=(...a)=>{const o=new Uint8Array(a.reduce((x,y)=>x+y.length,0));let p=0;for(const y of a){o.set(y,p);p+=y.length}return o};
const rbig=()=>{let v=0n;for(const b of rnd(9))v=v<<8n|BigInt(b);return v};
// models bundle produced by the Python experiment (models_bundle.json); public lists + HE-v1/HE-v2 specs + naive-Bayes tables
const PUB=MODELS.public,USERS=PUB.users,BINS=PUB.bins,WORDS=PUB.words,SUF2=PUB.suf,SYM=new Set(PUB.sym);
const BASE="password iloveyou princess sunshine monkey dragon football baseball welcome master shadow superman letmein qwerty abc123 admin login hello freedom whatever trustno1 batman starwars michael jennifer jordan ashley charlie thomas hunter soccer harley ranger buster killer pepper summer winter cookie flower tigger secret ginger orange banana computer internet samsung manipal udupi india cricket mumbai delhi krishna ganesh".split(' ');
const SUF=["","1","12","123","1234","!","@123","2020","2023","2024","2025","007","786","99","11","69","21","*","#1","321","12345","1990","2000","01","@"];
function dictionary(n){const c=[];BASE.forEach((b,i)=>SUF.forEach((s,j)=>c.push([(i+1)*(j+2),b+s])));c.sort((a,b)=>a[0]-b[0]||(a[1]<b[1]?-1:a[1]>b[1]?1:0));return[...new Set(c.map(x=>x[1]))].slice(0,n)}

const entropy=u=>{const c={};u.forEach(b=>c[b]=(c[b]||0)+1);return-Object.values(c).reduce((a,v)=>a+v/u.length*Math.log2(v/u.length),0)};
const luhn=s=>{let t=0;[...s].reverse().forEach((c,i)=>{let d=+c;if(i%2){d*=2;if(d>9)d-=9}t+=d});return t%10===0};
const luhnC=s=>{for(let c=0;c<10;c++)if(luhn(s+c))return''+c};
const zipfPick=n=>{let h=0;for(let i=1;i<=n;i++)h+=1/i;let u=Math.random()*h;for(let i=0;i<n;i++){u-=1/(i+1);if(u<=0)return i}return n-1};
const wpick=w=>{let u=Math.random()*w.reduce((a,b)=>a+b,0);for(let i=0;i<w.length;i++){u-=w[i];if(u<=0)return i}return w.length-1};
const CRED_RE=/^[a-z]{3,10}:[\x21-\x7e]{1,16}$/;
const classify=pt=>/^\d{16}$/.test(pt)?'card':CRED_RE.test(pt)?'cred':'text';
const cap=w=>w.slice(0,1).toUpperCase()+w.slice(1);
function parsePw(pw){let best=null;SUF2.forEach((s,si)=>{if(s&&!pw.endsWith(s))return;const stem=s?pw.slice(0,pw.length-s.length):pw,w=stem.toLowerCase(),wi=WORDS.indexOf(w);if(wi>=0&&(stem===w||stem===cap(w))){const c=[+(stem!==w),wi,si];if(!best||s.length>SUF2[best[2]].length)best=c}});return best}
const bucket=r=>r<0?-1:r<3?0:r<10?1:r<25?2:3;
// KDF + PBE baselines
async function kdf(pw,salt,it){const k=await S_.importKey('raw',TE.encode(pw),'PBKDF2',false,['deriveBits']);return new Uint8Array(await S_.deriveBits({name:'PBKDF2',hash:'SHA-256',salt,iterations:it},k,256))}
const mkPBE=(name,ivLen,why,par)=>({why,
 enc:async(k,pt)=>{const iv=rnd(ivLen);const ck=await S_.importKey('raw',k,name,false,['encrypt']);return cat(iv,new Uint8Array(await S_.encrypt(par(iv),ck,pt)))},
 dec:async(k,b)=>{const iv=b.slice(0,ivLen);const ck=await S_.importKey('raw',k,name,false,['decrypt']);return new Uint8Array(await S_.decrypt(par(iv),ck,b.slice(ivLen)))}});
const PBE={'PBE-GCM':mkPBE('AES-GCM',12,'authentication tag mismatch',iv=>({name:'AES-GCM',iv})),
 'PBE-CBC':mkPBE('AES-CBC',16,'invalid PKCS7 padding',iv=>({name:'AES-CBC',iv})),
 'PBE-CTR':mkPBE('AES-CTR',16,'',iv=>({name:'AES-CTR',counter:iv,length:64}))};
// Two-level distribution-transforming encoder over a 64-bit seed space
const T64=1n<<64n;
function dte(weights,sub){const w=weights.map(x=>BigInt(Math.max(1,Math.round(x*1e9)))),tot=w.reduce((a,b)=>a+b,0n);let acc=0n;const st=[];for(const x of w){st.push(T64*acc/tot);acc+=x}st.push(T64);
 return{enc(i,j){const a=st[i],sz=st[i+1]-a,lo=a+(j*sz+sub-1n)/sub,hi=a+((j+1n)*sz+sub-1n)/sub-1n;return lo+rbig()%(hi-lo+1n)},
  dec(seed){let l=0,h=st.length-2;while(l<h){const m=(l+h+1)>>1;st[m]<=seed?l=m:h=m-1}const sz=st[l+1]-st[l];return[l,(seed-st[l])*sub/sz]}}}
const MC={};
function buildModel(m){const key=m.klass+m.ver;if(MC[key])return MC[key];const sp=MODELS[m.klass][m.ver];let r;
 if(m.klass==='card'){const sub=sp.luhn?10n**9n:10n**10n,d=dte(sp.bin_w,sub),n=sp.luhn?15:16;
  r={enc:s=>{const i=sp.bins.indexOf(s.slice(0,6));if(i<0||(sp.luhn&&!luhn(s)))throw Error('HE card model needs a known BIN'+(sp.luhn?' and a Luhn-valid number':''));return d.enc(i,BigInt(s.slice(6,n)))},
   dec:v=>{const[i,j]=d.dec(v);const s=sp.bins[i]+String(j).padStart(n-6,'0');return sp.luhn?s+luhnC(s):s}}}
 else if(m.klass==='cred'){const d=dte(sp.pw_w,BigInt(sp.users.length)),ix=new Map(sp.vocab.map((w,i)=>[w,i]));
  r={enc:s=>{const k=s.indexOf(':'),i=ix.get(s.slice(k+1)),j=sp.users.indexOf(s.slice(0,k));if(i===undefined||j<0)throw Error('HE credential model covers the 20 demo users and the public password list (word, optional capital, suffix)');return d.enc(i,BigInt(j))},
   dec:v=>{const[i,j]=d.dec(v);return sp.users[Number(j)]+':'+sp.vocab[i]}}}
 else throw Error('Honey Encryption prototype supports 16-digit card numbers and user:password credentials');
 return MC[key]=r}
const b8=v=>Uint8Array.from(v.toString(16).padStart(16,'0').match(/../g).map(x=>parseInt(x,16)));
const n8=u=>BigInt('0x'+hex(u));
async function seal(id,pt,pw,it,salt){const klass=classify(pt);salt=salt||rnd(16);const t0=performance.now(),key=await kdf(pw,salt,it);let blob,model=null;
 if(id.startsWith('HE')){model={klass,ver:id.slice(3)};const m=buildModel(model),seed=m.enc(pt);blob=b8(seed^n8(key.slice(0,8)));if(m.dec(n8(blob)^n8(key.slice(0,8)))!==pt)throw Error('Plaintext not representable in the HE message space')}
 else blob=await PBE[id].enc(key,TE.encode(pt));
 return{pub:{id,kind:id.startsWith('HE')?'HE':id,salt,iters:it,blob,model,klass},secret:{pt,pw},encMs:performance.now()-t0}}
async function guess(pub,pw){const t0=performance.now(),key=await kdf(pw,pub.salt,pub.iters),t1=performance.now();let out=null,why='';
 if(pub.kind==='HE')out=TE.encode(buildModel(pub.model).dec(n8(pub.blob)^n8(key.slice(0,8))));
 else try{out=await PBE[pub.id].dec(key,pub.blob)}catch(e){why=PBE[pub.id].why}
 return{out,why,kdf:t1-t0,dec:performance.now()-t1}}
// ---- attacker analysis: uses ONLY the output and public information ----
function l1(k,s){if(k==='card'){if(!/^[0-9]{16}$/.test(s))return'not 16 digits';if(!luhn(s))return'Luhn checksum fails';if(!BINS.includes(s.slice(0,6)))return'BIN not in public list';return''}
 if(k==='cred'){const i=s.indexOf(':'),u=i<0?s:s.slice(0,i),p=i<0?'':s.slice(i+1);if(!CRED_RE.test(s))return'not user:password';if(!USERS.includes(u))return'unknown user';if(!parsePw(p))return'password is not word+suffix';return''}
 return[...s].every(c=>c>=' '&&c<='~')&&s.length?'':'non-printable bytes'}
function nbFeats(k,s){if(k==='card')return{luhn_valid:+(/^[0-9]+$/.test(s)&&luhn(s)),bin_id:BINS.indexOf(s.slice(0,6))};
 const i=s.indexOf(':'),p=i<0?'':s.slice(i+1),pr=parsePw(p),[c,w,si]=pr||[-1,-1,-1],sym=si>=0&&SYM.has(SUF2[si]);return{has_upper:c,suffix_id:si,word_bucket:bucket(w),cap_x_sym:+(c===1&&sym)}}
function nbScore(m,k,s){const f=nbFeats(k,s);return m.feats.reduce((a,n)=>a+(m.llr[n][String(f[n])]||0),0)}
function judge(pub,out,why,lvl){if(!out)return{shown:'(decryption failed)',f:null,ok:false,reason:why};
 const s=L1.decode(out),pr=out.filter(b=>b>=32&&b<127).length/Math.max(out.length,1);let reason=l1(pub.klass,s),f={len:out.length,pr,ent:entropy(out)};
 if(!reason&&lvl==='nb'&&pub.kind==='HE'&&pub.klass!=='text'){const nb=MODELS[pub.klass].nb[pub.model.ver],sc=nbScore(nb,pub.klass,s);f.nb=sc;if(sc<nb.tau)reason='naive-Bayes score below threshold'}
 return{shown:[...s].map(c=>c.charCodeAt(0)>=32&&c.charCodeAt(0)<127?c:'\\x'+c.charCodeAt(0).toString(16).padStart(2,'0')).join(''),f,ok:!reason,reason}}
function newSession(sealed,dict,max,policy,lvl){return{...sealed,dict,max:Math.min(max,dict.length),policy,lvl,i:0,rows:[],gt:[],kdfMs:0,decMs:0,done:false}}
async function step(s){if(s.done)return null;const pw=s.dict[s.i],g=await guess(s.pub,pw),j=judge(s.pub,g.out,g.why,s.lvl);
 const row={pw,...j};s.rows.push(row);s.gt.push({correct:pw===s.secret.pw});s.kdfMs+=g.kdf;s.decMs+=g.dec;s.i++;
 if(s.i>=s.max||(s.policy==='first'&&j.ok))s.done=true;return row}
function metrics(s){const made=s.i,ci=s.gt.findIndex(g=>g.correct),cand=s.rows.filter(r=>r.ok).length,wrongs=made-(ci>=0?1:0),pw=s.rows.filter((r,i)=>r.ok&&!s.gt[i].correct).length,fc=s.rows.findIndex(r=>r.ok),found=ci>=0&&s.rows[ci].ok;
 const rank=s.dict.indexOf(s.secret.pw),candPos=found?s.rows.slice(0,ci).filter(r=>r.ok).length:-1,fpr=wrongs?pw/wrongs:null,tpr=ci>=0?+found:null;
 return{guesses:made,candidates:cand,plausible_wrong:pw,wrong_guesses:wrongs,plausible_wrong_rate:fpr,distinguishability:fpr==null?null:1-fpr,tpr,fpr,advantage:tpr!=null&&fpr!=null?tpr-fpr:null,
  correct_tried:ci>=0,found,top1:fc>=0&&s.gt[fc].correct,cand_pos:candPos,top10:candPos>=0&&candPos<10,rank,in_dict:rank>=0,
  kdf_ms:s.kdfMs,dec_ms:s.decMs,total_ms:s.kdfMs+s.decMs,dec_us_per_guess:made?s.decMs/made*1000:0,enc_ms:s.encMs,blob_bytes:s.pub.blob.length+s.pub.salt.length}}
// researcher-side generator (same synthetic distributions as groundtruth.py)
function randomInstance(klass){const G=MODELS.gen;if(klass==='card'){const s=BINS[wpick(G.bin_w)]+String(Math.floor(Math.random()*1e9)).padStart(9,'0');return s+luhnC(s)}
 const c=+(Math.random()<G.p_cap),w=WORDS[zipfPick(WORDS.length)],sf=SUF2[wpick(G.suf_w[c])];return USERS[Math.floor(Math.random()*20)]+':'+(c?cap(w):w)+sf}
