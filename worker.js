import { AutoTokenizer, AutoModelForCausalLM, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2';
env.allowLocalModels=false;
env.useBrowserCache=true;
env.backends.onnx.wasm.numThreads=1;
let tokenizer,model;
const activeDevice='webgpu';
const MAX_BYTES=1_000_000_000;
const send=(kind,fields={})=>postMessage({kind,...fields});
const message=e=>e?.message||String(e);
async function load(id,source){
  if(!navigator.gpu)throw Error('WebGPU indisponible dans le worker. Essayez Chrome ou Edge à jour, ou l’API Hugging Face.');
  const adapter=await navigator.gpu.requestAdapter();
  if(!adapter)throw Error('Aucun adaptateur GPU. Utilisez l’API Hugging Face ou un navigateur compatible.');
  const base=source?.split('/').pop();
  const candidates=[...new Set([id,...(base?['onnx-community/'+base+'-ONNX','onnx-community/'+base,'Xenova/'+base]:[])])].filter(Boolean);
  const choices=[['q4','model_q4.onnx'],['q8','model_quantized.onnx'],['int8','model_int8.onnx'],['uint8','model_uint8.onnx'],['fp32','model.onnx']];
  let choice,bytes,revision;const diagnostics=[];
  for(const candidate of candidates){try{
    const response=await fetch('https://huggingface.co/api/models/'+candidate+'?blobs=true',{signal:AbortSignal.timeout(15000)});
    if(!response.ok){diagnostics.push(candidate+' : HTTP '+response.status);continue;}
    const info=await response.json();if(info.gated||info.private)continue;
    const files=info.siblings||[];
    const sizeOf=f=>f.size??f.lfs?.size;
    for(const c of choices){
      const path='onnx/'+c[1];
      if(!files.some(f=>f.rfilename===path))continue;
      const selected=files.filter(f=>f.rfilename===path||f.rfilename.startsWith(path+'_data')||f.rfilename.startsWith(path+'data')||f.rfilename.startsWith(path+'.data')||(!f.rfilename.includes('/')&&/^(.*\.json|.*\.model|vocab\.txt|merges\.txt|.*\.tiktoken)$/i.test(f.rfilename)));
      if(selected.some(f=>!Number.isFinite(sizeOf(f))||sizeOf(f)<0)){diagnostics.push(candidate+' '+c[0]+' : taille inconnue');continue;}
      const total=selected.reduce((sum,f)=>sum+sizeOf(f),0);
      if(total>MAX_BYTES){diagnostics.push(candidate+' '+c[0]+' : '+Math.ceil(total/1e6)+' Mo > 1 Go');continue;}
      choice=c;bytes=total;id=candidate;revision=info.sha||'main';break;
    }
    if(choice)break;
  }catch(e){diagnostics.push(candidate+' : '+message(e));}}
  if(!choice)throw Error('Aucun fichier ONNX compatible de taille vérifiée ≤ 1 Go. '+diagnostics.join(' ; '));
  send('progress',{text:`Chargement WebGPU : ${id} (${choice[0]}) — environ ${Math.ceil(bytes/1e6)} Mo de fichiers modèle.`});
  tokenizer=await AutoTokenizer.from_pretrained(id,{revision});
  try{
    model=await AutoModelForCausalLM.from_pretrained(id,{revision,device:'webgpu',dtype:choice[0],progress_callback:p=>{if(p.status==='progress')send('progress',{text:`Téléchargement : ${p.file} — ${Math.round(p.progress)} %`,percent:p.progress});else if(p.status==='initiate')send('progress',{text:'Chargement : '+p.file})}});
  }catch(e){throw Error(message(e)+' — Ce modèle ou sa quantification peut être incompatible avec votre GPU. Essayez GPT-2 Small ou l’API Hugging Face.');}
  send('ready',{dtype:choice[0],id,device:activeDevice});
}
// Distribution initiale et distribution de tirage (température + top-k).
function nextDistributions(scores,temperature,k=50){
  let max=-Infinity;
  for(const x of scores){if(Number.isNaN(x)||x===Infinity)throw Error('Logits invalides.');if(x>max)max=x;}
  if(!Number.isFinite(max)||!Number.isFinite(temperature)||temperature<=0)throw Error('Distribution non calculable.');
  let total=0;const ranked=[];
  for(let id=0;id<scores.length;id++){const x=scores[id];if(!Number.isFinite(x))continue;total+=Math.exp(x-max);ranked.push({id,x});}
  ranked.sort((a,b)=>b.x-a.x||a.id-b.id);
  const kept=ranked.slice(0,k),den=kept.reduce((s,v)=>s+Math.exp((v.x-max)/temperature),0);
  if(!(total>0&&den>0))throw Error('Distribution non calculable.');
  return {before:ranked.slice(0,10).map(v=>({id:v.id,p:100*Math.exp(v.x-max)/total})),after:kept.slice(0,10).map(v=>({id:v.id,p:100*Math.exp((v.x-max)/temperature)/den})),candidates:kept,kept:kept.length};
}
async function generate(d){
  if(!model)throw Error('Modèle non chargé.');
  const text=d.chat?tokenizer.apply_chat_template([{role:'user',content:d.prompt}],{tokenize:false,add_generation_prompt:true}):d.prompt;
  let inputs,output,result;
  try{
    inputs=await tokenizer(text,{add_special_tokens:!d.chat});
    const n=inputs.input_ids.dims[1];
    if(n>800)throw Error('Texte trop long : '+n+' tokens. Limite pédagogique : 800.');
    try{
      output=await model(inputs);
      const logits=output.logits,vocab=logits.dims.at(-1);
      let data;try{data=logits.data;}catch(e){if(typeof logits.getData!=='function')throw e;data=await logits.getData();}
      const scores=data.slice(-vocab),dist=nextDistributions(scores,d.greedy?1:d.temp);
      const decode=items=>items.map(x=>({...x,token:tokenizer.decode([x.id],{skip_special_tokens:false})}));
      send('predictions',{items:decode(dist.before),candidates:dist.candidates.map((v,i)=>({...v,token:i<10?tokenizer.decode([v.id],{skip_special_tokens:false}):''}))});
    }catch(e){send('prediction-warning',{message:message(e)});}
    finally{if(output)for(const value of Object.values(output))value?.dispose?.();}
    const start=performance.now();
    result=await model.generate({...inputs,max_new_tokens:d.count,do_sample:!d.greedy,repetition_penalty:1,no_repeat_ngram_size:0,...(!d.greedy?{temperature:d.temp,top_k:50,top_p:1}:{})});
    const seconds=(performance.now()-start)/1000;
    const ids=result.tolist()[0].slice(n).map(Number);
    const generated=tokenizer.decode(ids,{skip_special_tokens:true,clean_up_tokenization_spaces:false});
    send('result',{prompt:d.prompt,text:generated,count:ids.length,seconds,speed:ids.length/Math.max(seconds,0.001),device:activeDevice});
  }finally{result?.dispose?.();if(inputs)for(const v of Object.values(inputs))v?.dispose?.();}
}
self.onmessage=async({data:d})=>{try{if(d.kind==='load')await load(d.id,d.source);else if(d.kind==='generate')await generate(d)}catch(e){send('error',{message:message(e)})}};
