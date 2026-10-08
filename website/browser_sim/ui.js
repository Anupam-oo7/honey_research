// ---------- UI ----------
const $=s=>document.querySelector(s),esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const IDS=[['PBE-GCM','PBE: AES-GCM (authenticated)'],['PBE-CBC','PBE: AES-CBC (PKCS7)'],['PBE-CTR','PBE: AES-CTR (no integrity)'],['HE-v1','HE-v1: initial decoy model'],['HE-v2','HE-v2: model revised from analysis']];
const NAME=Object.fromEntries(IDS),sleep=ms=>new Promise(r=>setTimeout(r,ms)),pc=p=>p==null?'n/a':(p*100).toFixed(1)+'%',ms=v=>v.toFixed(2)+' ms';
$('#sa').innerHTML=IDS.map(([k,v])=>`<option value="${k}">${v}</option>`).join('');
$('#sb').innerHTML='<option value="">None (single run)</option>'+IDS.map(([k,v])=>`<option value="${k}">${v}</option>`).join('');
$('#sc').innerHTML=$('#sb').innerHTML;$('#sa').value='PBE-GCM';$('#sb').value='HE-v1';$('#sc').value='HE-v2';
$('#wfb').innerHTML=['card','cred'].map(k=>`<p><b>${k==='card'?'Card numbers':'Vault credentials'}</b>: held-out classifier AUC ${MODELS[k].auc.v1.toFixed(3)} (v1) to ${MODELS[k].auc.v2.toFixed(3)} (v2), computed in the Python experiment.</p><ul>`+MODELS[k].changes.map(c=>`<li>${esc(c.component)}: ${esc(c.action)} (triggered by ${c.triggered_by.map(esc).join(', ')})</li>`).join('')+'</ul>').join('');
$('#xsch').innerHTML=IDS.map(([k,v])=>`<label style="flex-direction:row;align-items:center;gap:6px"><input type="checkbox" style="width:auto" value="${k}" checked>${v}</label>`).join('');
let S=[],view='a',running=false,EXP=[],runNo=0,stopX=false,chart;
const msg=t=>$('#msg').textContent=t||'';
$('#tabs').onclick=e=>{const t=e.target.dataset.t;if(!t)return;document.querySelectorAll('#tabs button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.t===t));$('#t-sim').classList.toggle('hide',t!=='sim');$('#t-exp').classList.toggle('hide',t!=='exp')};
$('#pre').onchange=()=>$('#pt').value=$('#pre').value;
$('#ds').onchange=()=>$('#cw').classList.toggle('hide',$('#ds').value!=='custom');
$('#oc').onchange=()=>$('#panels').classList.toggle('oc',$('#oc').checked);
$('#vw').onclick=e=>{const v=e.target.dataset.v;if(!v)return;view=v;document.querySelectorAll('#vw .seg button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.v===v));rebuild()};
const getDict=()=>$('#ds').value==='custom'?[...new Set($('#cu').value.split('\n').map(x=>x.trim()).filter(Boolean))]:dictionary(+$('#ds').value);
const feat=r=>r.f?`len ${r.f.len} · printable ${Math.round(r.f.pr*100)}% · H ${r.f.ent.toFixed(2)}${r.f.nb!==undefined?' · NB '+r.f.nb.toFixed(2):''}`:'no output';
function rowHTML(s,k){const r=s.rows[k],g=s.gt[k];
 if(view==='a')return`<tr class="${r.ok?'cand':'rj'}"><td>${k+1}</td><td class="mono">${esc(r.pw)}</td><td class="mono">${esc(r.shown)}</td><td class="mono">${feat(r)}</td><td>${r.ok?'Candidate':'Reject: '+esc(r.reason)}</td></tr>`;
 const cls=g.correct?'hit':r.ok?'fp':'rj';
 return`<tr class="${cls}"><td>${k+1}</td><td class="mono">${esc(r.pw)}</td><td>${g.correct?'<b class="good">correct</b>':'wrong'}</td><td class="mono">${esc(r.shown)}</td><td>${g.correct?'genuine plaintext':r.ok?'plausible decoy (false positive)':'ruled out'}</td></tr>`}
function inference(s){const c=s.rows.filter(r=>r.ok),first=c[0];
 if(!s.rows.length)return'No guesses yet.';if(!c.length)return s.done?'No candidate found in the guess budget.':'No candidates so far.';
 if(s.pub.id==='PBE-GCM')return`Candidate "${esc(first.pw)}" passed AES-GCM authentication. The tag verifies, so the attacker is certain this is the password.`;
 if(c.length===1)return`One candidate so far: "${esc(first.pw)}". Nothing else passes the checks, so it is very likely genuine.`;
 return`${c.length} candidates pass the checks; the first is "${esc(first.pw)}". The outputs alone do not reveal which is genuine, so the attacker must verify each candidate elsewhere.`}
function panel(s,i){const pub=s.pub,a=view==='a',m=metrics(s);
 let h=`<div class="panel ${a?'pa':'pe'}"><h3>${NAME[pub.id]}</h3>`;
 if(a)h+=`<div class="box"><b>The attacker holds</b><br>Ciphertext: <span class="mono">${hex(pub.blob)}</span><br>Salt: <span class="mono">${hex(pub.salt)}</span><br>KDF: PBKDF2-SHA256, ${pub.iters} iterations · Plaintext class: ${pub.klass}${pub.model?' · HE message model is public':''}<br>Dictionary: ${s.dict.length} passwords · Guess budget: ${s.max}</div>`;
 else h+=`<div class="box"><b>Sealed ground truth</b> (never given to the attacker)<br>Plaintext: <span class="mono">${esc(s.secret.pt)}</span><br>Correct password: <span class="mono">${esc(s.secret.pw)}</span> · ${m.in_dict?`rank ${m.rank+1} in the dictionary`:'<span class="bad">not in the dictionary</span>'}</div>`;
 h+=`<div class="bar"><i style="width:${s.i/s.max*100}%"></i></div><div class="mute" id="pg${i}"></div><div class="tw"><table><thead><tr>${a?'<th>#</th><th>Candidate password</th><th>Resulting output</th><th>Observable features</th><th>Attacker decision</th>':'<th>#</th><th>Candidate password</th><th>Ground truth</th><th>Output</th><th>Meaning</th>'}</tr></thead><tbody id="tb${i}">${s.rows.map((_,k)=>rowHTML(s,k)).join('')}</tbody></table></div>`;
 return h+`<div class="box" id="in${i}"></div></div>`}
function stats(s,i){const m=metrics(s);$('#pg'+i).textContent=`Guess ${s.i} of ${s.max} · candidates ${m.candidates} · rejected ${m.guesses-m.candidates}${s.done?' · attack finished':''}`;
 $('#in'+i).innerHTML=view==='a'?`<b>Attacker's inference</b><br>${inference(s)}`:`<b>Evaluation</b><br>Genuine password ${m.correct_tried?(m.found?'tried and accepted as a candidate':'tried but rejected'):'not tried yet'} · plausible wrong outputs ${m.plausible_wrong} of ${m.wrong_guesses} wrong guesses · distinguishability ${pc(m.distinguishability)}`;
 const t=$('#tb'+i).parentElement.parentElement;t.scrollTop=t.scrollHeight}
function rebuild(){if(!S.length)return;$('#panels').innerHTML=S.map(panel).join('');S.forEach(stats);summary()}
function summary(){const M=S.map(metrics);$('#sumw').classList.toggle('hide',view!=='e');if(view!=='e')return;
 const R=[['Guesses made',m=>m.guesses],['Plausible outputs (candidates)',m=>m.candidates],['Plausible wrong outputs',m=>m.plausible_wrong],['Plausible-wrong rate',m=>pc(m.plausible_wrong_rate)],['Distinguishability (1 - FPR)',m=>m.distinguishability==null?'n/a':m.distinguishability.toFixed(3)],['TPR (genuine output accepted)',m=>m.tpr==null?'n/a':m.tpr],['FPR (wrong outputs accepted)',m=>pc(m.fpr)],['Advantage (TPR - FPR)',m=>m.advantage==null?'n/a':m.advantage.toFixed(3)],
 ['Genuine password accepted as candidate',m=>m.found?'yes':m.correct_tried?'no':'not reached'],['Attacker success (first candidate is genuine)',m=>m.top1?'yes':'no'],['Genuine position among candidates',m=>m.cand_pos<0?'n/a':m.cand_pos+1],
 ['Total attack time',m=>ms(m.total_ms),m=>m.total_ms],['Key derivation time',m=>ms(m.kdf_ms),m=>m.kdf_ms],['Decrypt-only time per guess',m=>m.dec_us_per_guess.toFixed(1)+' µs',m=>m.dec_us_per_guess],['Encryption time (incl. KDF)',m=>ms(m.enc_ms),m=>m.enc_ms],['Ciphertext size (with salt)',m=>m.blob_bytes+' B',m=>m.blob_bytes]];
 const two=M.length===2;$('#sum').innerHTML=`<table><tr><th>Metric</th>${S.map(s=>`<th>${NAME[s.pub.id]}</th>`).join('')}${two?'<th>Overhead (right ÷ left)</th>':''}</tr>`+R.map(([l,f,n])=>`<tr><td>${l}</td>${M.map(m=>`<td>${f(m)}</td>`).join('')}${two?`<td>${n&&n(M[0])>0?(n(M[1])/n(M[0])).toFixed(2)+'×':''}</td>`:''}</tr>`).join('')+'</table>'}
async function advance(){for(let i=0;i<S.length;i++){const s=S[i];if(s.done)continue;await step(s);const k=s.rows.length-1;$('#tb'+i).insertAdjacentHTML('beforeend',rowHTML(s,k));stats(s,i)}summary();ctl()}
function ctl(){const d=S.every(s=>s.done);$('#b-step').disabled=!S.length||d||running;$('#b-fin').disabled=!S.length||d;$('#b-run').disabled=!S.length||d;$('#b-run').textContent=running?'Pause':'Run'}
$('#b-seal').onclick=async()=>{msg('');running=false;const dict=getDict(),pt=$('#pt').value,pw=$('#pw').value,it=+$('#it').value,max=Math.max(1,+$('#mx').value||1);
 if(!pt||!pw||!dict.length)return msg('Enter a plaintext, a password and a dictionary.');
 try{const ids=[...new Set([$('#sa').value,$('#sb').value,$('#sc').value].filter(Boolean))];
  S=[];for(const id of ids)S.push(newSession(await seal(id,pt,pw,it),dict,max,$('#po').value,$('#st').value));
  rebuild();ctl()}catch(e){msg(e.message)}};
$('#b-step').onclick=async()=>{$('#b-step').disabled=true;await advance()};
$('#b-run').onclick=async()=>{if(running){running=false;return ctl()}running=true;ctl();while(running&&S.some(s=>!s.done)){await advance();await sleep(+$('#spd').value)}running=false;ctl()};
$('#b-fin').onclick=async()=>{running=false;let n=0;while(S.some(s=>!s.done)){await advance();if(++n%15===0)await sleep(0)}};
// ---------- experiment mode ----------
const KEYS=['run','trial','scheme','workload','dict_size','max_guesses','iterations','attacker_level','plaintext','password','rank','in_dict','guesses','candidates','plausible_wrong','wrong_guesses','distinguishability','tpr','fpr','advantage','found','top1','top10','cand_pos','kdf_ms','dec_ms','total_ms','enc_ms','blob_bytes'];
$('#b-exp').onclick=()=>{$('#xs').value='s';document.querySelector('[data-t=exp]').click();$('#xd').value=$('#ds').value==='custom'?'300':$('#ds').value;$('#xm').value=$('#mx').value;$('#xi').value=$('#it').value;$('#xk').value=$('#st').value};
$('#x-run').onclick=async()=>{const sch=[...document.querySelectorAll('#xsch input:checked')].map(c=>c.value),N=+$('#xn').value,it=+$('#xi').value,max=+$('#xm').value,klass=$('#xw').value,fixed=$('#xs').value==='s',lvl=$('#xk').value;
 if(!sch.length)return;const dict=fixed?getDict():dictionary(+$('#xd').value);runNo++;stopX=false;$('#x-run').disabled=true;$('#x-stop').disabled=false;let err='';
 for(let t=0;t<N&&!stopX&&!err;t++){const pt=fixed?$('#pt').value:randomInstance(klass),pw=fixed?$('#pw').value:dict[zipfPick(dict.length)],salt=rnd(16);   // paired design: same plaintext, password and salt for every scheme in a trial; fresh per trial

  for(const id of sch){try{const s=newSession(await seal(id,pt,pw,it,salt),dict,max,'all',lvl);while(!s.done)await step(s);const m=metrics(s);
   EXP.push({run:runNo,trial:t+1,scheme:id,workload:classify(pt),dict_size:dict.length,max_guesses:s.max,iterations:it,attacker_level:lvl,plaintext:pt,password:pw,...m})}catch(e){err=e.message;break}}
  $('#xp').style.width=(t+1)/N*100+'%';$('#xs2').textContent=err||`Trial ${t+1} of ${N}`;if(t%2===1){expRender();await sleep(0)}}
 $('#x-run').disabled=false;$('#x-stop').disabled=true;expRender()};
$('#x-stop').onclick=()=>stopX=true;
$('#x-clr').onclick=()=>{EXP=[];expRender()};
async function save(name,data){let d=null;try{d=window.claude?await claude.use('downloads'):null}catch(e){}if(!d)return $('#xs2').textContent='Saving files is not available in this view.';try{await d.save({filename:name,data})}catch(e){if(e.code!=='declined')$('#xs2').textContent=e.message}}
$('#x-csv').onclick=()=>save('trials.csv',[KEYS.join(','),...EXP.map(r=>KEYS.map(k=>{const v=r[k];return typeof v==='string'?'"'+v.replace(/"/g,'""')+'"':v??''}).join(','))].join('\n'));
$('#x-json').onclick=()=>save('trials.json',JSON.stringify(EXP,null,1));
const avg=a=>a.length?a.reduce((x,y)=>x+y,0)/a.length:null;
function expRender(){$('#x-csv').disabled=$('#x-json').disabled=!EXP.length;
 const by={};EXP.forEach(r=>(by[r.scheme]??=[]).push(r));const ids=IDS.map(x=>x[0]).filter(k=>by[k]);
 $('#xt').innerHTML=ids.length?'<table><tr><th>Scheme</th><th>Trials</th><th>Avg candidates</th><th>TPR</th><th>FPR</th><th>Advantage</th><th>Distinguishability</th><th>Top-1 success</th><th>Top-10 success</th><th>Genuine accepted</th><th>Avg time / guess</th><th>Avg encrypt</th></tr>'+ids.map(k=>{const a=by[k];return`<tr><td>${NAME[k]}</td><td>${a.length}</td><td>${avg(a.map(r=>r.candidates)).toFixed(1)}</td>${(()=>{const c=a.filter(r=>r.correct_tried),t=c.length?c.filter(r=>r.found).length/c.length:null,w=a.reduce((x,r)=>x+r.wrong_guesses,0),f=w?a.reduce((x,r)=>x+r.plausible_wrong,0)/w:null;return`<td>${pc(t)}</td><td>${pc(f)}</td><td>${t==null||f==null?'n/a':(t-f).toFixed(3)}</td>`})()}<td>${avg(a.filter(r=>r.distinguishability!=null).map(r=>r.distinguishability))?.toFixed(3)??'n/a'}</td><td>${pc(avg(a.map(r=>+r.top1)))}</td><td>${pc(avg(a.map(r=>+r.top10)))}</td><td>${pc(avg(a.map(r=>+r.found)))}</td><td>${(avg(a.map(r=>r.total_ms/Math.max(r.guesses,1)))).toFixed(3)} ms</td><td>${ms(avg(a.map(r=>r.enc_ms)))}</td></tr>`}).join('')+'</table>':'<p class="mute">No trials yet.</p>';
 $('#xl').innerHTML=EXP.length?'<table><tr><th>Run</th><th>Trial</th><th>Scheme</th><th>Rank</th><th>Guesses</th><th>Candidates</th><th>Distinguishability</th><th>Top-1</th><th>Time</th></tr>'+EXP.slice(-25).reverse().map(r=>`<tr><td>${r.run}</td><td>${r.trial}</td><td>${r.scheme}</td><td>${r.rank<0?'not in dict':r.rank+1}</td><td>${r.guesses}</td><td>${r.candidates}</td><td>${r.distinguishability==null?'n/a':r.distinguishability.toFixed(3)}</td><td>${r.top1?'yes':'no'}</td><td>${ms(r.total_ms)}</td></tr>`).join('')+'</table>':'';
 if(typeof Chart==='undefined')return;
 const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim(),mx=Math.max(1,...EXP.map(r=>r.max_guesses)),ks=[1,2,5,10,20,50,100,200,500,1000].filter(k=>k<=mx),COL={'PBE-GCM':'#2f6f8f','PBE-CBC':'#4aa3a0','PBE-CTR':'#8e7cc3','HE-v1':'#c0392b','HE-v2':'#2e7d4f'};
 const ds=ids.map(k=>({label:NAME[k],data:ks.map(x=>avg(by[k].map(r=>+(r.cand_pos>=0&&r.cand_pos<x)))),borderColor:COL[k],backgroundColor:COL[k],tension:.15,borderWidth:2,pointRadius:2}));
 if(EXP.length)ds.push({label:'Dictionary order only',data:ks.map(x=>avg(EXP.map(r=>+(r.rank>=0&&r.rank<Math.min(x,r.max_guesses))))),borderColor:'#8795a6',backgroundColor:'#8795a6',borderDash:[6,4],borderWidth:2,pointRadius:0});
 if(chart)chart.destroy();chart=new Chart($('#xc'),{type:'line',data:{labels:ks,datasets:ds},options:{maintainAspectRatio:false,animation:false,scales:{y:{min:0,max:1,ticks:{callback:v=>v*100+'%',color:css('--mute')},grid:{color:css('--line')}},x:{title:{display:true,text:'candidates verified (k)',color:css('--mute')},ticks:{color:css('--mute')},grid:{color:css('--line')}}},plugins:{legend:{labels:{color:css('--ink'),boxWidth:14}}}}})}
expRender();
</script></body></html>
