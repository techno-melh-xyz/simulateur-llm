import { AutoTokenizer, AutoModelForCausalLM, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2';
env.allowLocalModels=false;
env.useBrowserCache=true;
// One thread avoids requiring cross-origin isolation on GitHub Pages.
env.backends.onnx.wasm.numThreads=1;
let tokenizer,model;
const send=(kind,fields={})=>postMessage({kind,...fields});
async function load(id,source,heavy){
  // Inspect the public repository instead of assuming the quantized filename.
  const base=source?.split('/').pop();
  const candidates=[...new Set([id, ...(base?['onnx-community/'+base+'-ONNX','onnx-community/'+base,'Xenova/'+base]:[])])];
  const choices=[['q4','model_q4.onnx'],['q8','model_quantized.onnx'],['int8','model_int8.onnx'],['uint8','model_uint8.onnx'],...(!heavy?[['fp32','model.onnx']]:[])];
  let choice;
  for(const candidate of candidates){try{const response=await fetch('https://huggingface.co/api/models/'+candidate);if(!response.ok)continue;const info=await response.json();if(info.gated||info.private)continue;const files=new Set((info.siblings||[]).map(x=>x.rfilename));choice=choices.find(([,file])=>files.has('onnx/'+file));if(choice){id=candidate;break}}catch{}}

  if(!choice)throw Error('Aucun fichier ONNX compatible trouvé dans '+id+'.');
  tokenizer=await AutoTokenizer.from_pretrained(id);
  model=await AutoModelForCausalLM.from_pretrained(id,{device:'wasm',dtype:choice[0],progress_callback:p=>{if(p.status==='progress')send('progress',{text:`Téléchargement : ${p.file} — ${Math.round(p.progress)} %`,percent:p.progress});else if(p.status==='initiate')send('progress',{text:'Chargement : '+p.file})}});
  send('ready',{dtype:choice[0],id});
}
async function generate(d){
  if(!model)throw Error('Modèle non chargé.');
  const text=d.chat?tokenizer.apply_chat_template([{role:'user',content:d.prompt}],{tokenize:false,add_generation_prompt:true}):d.prompt;
  const inputs=await tokenizer(text,{add_special_tokens:!d.chat});
  const n=inputs.input_ids.dims[1];
  if(n>800)throw Error('Texte trop long : '+n+' tokens. Limite pédagogique : 800.');
  const output=await model(inputs);
  const logits=output.logits;const vocab=logits.dims.at(-1);const scores=logits.data.slice(-vocab);
  let max=-Infinity;for(const x of scores)if(x>max)max=x;
  const exp=Array.from(scores,x=>Math.exp(x-max));const total=exp.reduce((a,b)=>a+b,0);
  const items=exp.map((x,id)=>({id,p:100*x/total})).sort((a,b)=>b.p-a.p).slice(0,10).map(x=>({token:tokenizer.decode([x.id],{skip_special_tokens:false}),p:x.p}));
  send('predictions',{items});
  for(const value of Object.values(output))if(value&&typeof value.dispose==='function')value.dispose();
  const start=performance.now();
  const result=await model.generate({...inputs,max_new_tokens:d.count,do_sample:!d.greedy,...(!d.greedy?{temperature:d.temp,top_k:50}:{})});
  const seconds=(performance.now()-start)/1000;
  const ids=result.tolist()[0].slice(n).map(Number);
  const generated=tokenizer.decode(ids,{skip_special_tokens:true,clean_up_tokenization_spaces:false});
  send('result',{prompt:d.prompt,text:generated,count:ids.length,seconds,speed:ids.length/Math.max(seconds,0.001)});
  result.dispose?.();for(const v of Object.values(inputs))v.dispose?.();
}
self.onmessage=async({data:d})=>{try{if(d.kind==='load')await load(d.id,d.source,d.heavy);else if(d.kind==='generate')await generate(d)}catch(e){send('error',{message:e.message||String(e)})}};
