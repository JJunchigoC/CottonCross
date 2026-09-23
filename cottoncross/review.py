"""Create a local, offline image review and optional crossing-point annotation page."""
import argparse
import json
from pathlib import Path
import os
from .common import load_json

HTML=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CottonCross 数据复核</title>
<style>
body{margin:0;background:#111b2b;color:#e7eef8;font:15px system-ui,"Microsoft YaHei"}header{padding:22px 30px;background:#18263b}h1{font-size:24px;margin:0 0 8px}.muted{color:#9eafc6}main{display:grid;grid-template-columns:290px 1fr;gap:22px;padding:24px}aside{background:#18263b;padding:18px;border-radius:12px;height:fit-content}button,select{font:inherit;border:0;border-radius:6px;padding:9px;margin:5px 0;max-width:100%}button{background:#63dcc3;color:#111b2b;cursor:pointer}.secondary{background:#32465f;color:#fff}.stage{position:relative;display:inline-block;max-width:100%;line-height:0}img{max-width:100%;max-height:79vh;object-fit:contain}svg{position:absolute;left:0;top:0;width:100%;height:100%;cursor:crosshair}.note{line-height:1.8;font-size:13px}#status{margin:14px 0}.tag{color:#63dcc3}input{margin:9px 0}label{display:block;margin:10px 0}
</style><header><h1>CottonCross · 棉花纤维交叉复核</h1><span class="muted">本地离线 · 原图 / 预测切换 · 投影交叉候选，不代表物理接触或上下层次</span></header>
<main><aside><select id="selector"></select><div><button id="prev" class="secondary">上一张</button> <button id="next" class="secondary">下一张</button></div>
<label><input type="checkbox" id="prediction">显示模型预测叠加</label><label><input type="checkbox" id="annotate">人工标记交叉点</label><div class="note muted">在原图上单击添加黄色标记；点击已有标记可删除。模型预测不会自动作为人工真值。标完所有真实交叉点后，再勾选“本图标注完成”。</div>
<label><input type="checkbox" id="complete">本图标注完成（包括确认为零交叉）</label>
<button id="export">导出人工标注 JSON</button><button id="clear" class="secondary">清空当前人工标记</button><input id="import" type="file" accept=".json">
<div id="status"></div><div class="note muted">红圈：非歧义关节点（不等于真实交叉）；橙圈：配对不确定。不同颜色线条：重建轨迹。黄色点：你手工标记的真值。<br>没有人工标注时，真实准确率保持“未评估”。<br>标注保存在当前浏览器；请导出文件留存。可导入之前的 JSON 继续。</div></aside>
<section><div class="stage"><img id="image"><svg id="marks"></svg></div><p id="caption" class="muted"></p></section></main>
<script>
const records=__RECORDS__;
let idx=0,annotations={};try{annotations=JSON.parse(localStorage.getItem('cottoncross-annotations-v1')||'{}')}catch(e){}
const el=id=>document.getElementById(id);const sel=el('selector');
for(const [i,r] of records.entries()){const o=document.createElement('option');o.value=i;o.textContent=`第 ${r.page} 页 · ${r.split}`;sel.appendChild(o)}
function current(){const r=records[idx];return annotations[r.id]||(annotations[r.id]={id:r.id,page:r.page,points_xy_original:[],complete:false})}
function persist(){try{localStorage.setItem('cottoncross-annotations-v1',JSON.stringify(annotations))}catch(e){}renderMarks()}
function renderMarks(){const r=records[idx],a=current();const svg=el('marks');svg.replaceChildren();svg.setAttribute('viewBox',`0 0 ${r.w} ${r.h}`);
if(!el('prediction').checked)for(const [i,p] of a.points_xy_original.entries()){let circle=document.createElementNS('http://www.w3.org/2000/svg','circle');circle.setAttribute('cx',p[0]-r.x0);circle.setAttribute('cy',p[1]-r.y0);circle.setAttribute('r',9);circle.setAttribute('stroke','#ffd54f');circle.setAttribute('stroke-width',4);circle.setAttribute('fill','none');svg.appendChild(circle)}
el('complete').checked=a.complete;el('status').textContent=`人工点数：${a.points_xy_original.length} · ${a.complete?'已完成':'未完成'}`;}
function load(){sel.value=idx;const r=records[idx];el('image').src=el('prediction').checked?r.overlay:r.original;el('caption').textContent=`${r.id} | ROI 原始坐标 [${r.x0}, ${r.y0}] | ${r.w} × ${r.h} px | ${r.split}`;renderMarks()}
sel.onchange=()=>{idx=+sel.value;load()};el('prev').onclick=()=>{idx=(idx+records.length-1)%records.length;load()};el('next').onclick=()=>{idx=(idx+1)%records.length;load()};el('prediction').onchange=load;
el('marks').onclick=e=>{if(!el('annotate').checked||el('prediction').checked)return;const r=records[idx],a=current(),box=el('marks').getBoundingClientRect();const p=[(e.clientX-box.left)/box.width*r.w+r.x0,(e.clientY-box.top)/box.height*r.h+r.y0];const near=a.points_xy_original.findIndex(q=>Math.hypot(q[0]-p[0],q[1]-p[1])<15);if(near>=0)a.points_xy_original.splice(near,1);else a.points_xy_original.push(p.map(v=>+v.toFixed(2)));a.complete=false;persist()};
el('complete').onchange=()=>{current().complete=el('complete').checked;persist()};el('clear').onclick=()=>{if(confirm('清空当前图的人工标记？')){current().points_xy_original=[];current().complete=false;persist()}};
el('export').onclick=()=>{const data={schema:'cottoncross.crossing-ground-truth.v1',coordinate_system:'original extracted image pixels',annotations:Object.values(annotations)};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download='crossing_annotations.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
el('import').onchange=async e=>{try{const d=JSON.parse(await e.target.files[0].text());if(d.schema!=='cottoncross.crossing-ground-truth.v1')throw Error('格式不匹配');for(const a of d.annotations)annotations[a.id]=a;persist();load()}catch(err){alert('导入失败：'+err.message)}};load();
</script></html>'''


def create(real,prediction,out):
    real=Path(real).resolve();prediction=Path(prediction).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for r in load_json(real/'manifest.json'):
        x0,y0,x1,y1=r['roi_xyxy'];overlay=prediction/f"{r['id']}_overlay.jpg"
        original=real/r['roi']
        rows.append(dict(id=r['id'],page=r['page'],split=r['split'],x0=x0,y0=y0,w=x1-x0,h=y1-y0,
                         original=os.path.relpath(original,out).replace('\\','/'),
                         overlay=os.path.relpath(overlay if overlay.exists() else original,out).replace('\\','/')))
    (out/'index.html').write_text(HTML.replace('__RECORDS__',json.dumps(rows)),encoding='utf8')
    print(out/'index.html')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--real',default='data/real');p.add_argument('--predictions',default='results/real/main/cfxnet')
    p.add_argument('--out',default='results/real/review');a=p.parse_args();create(a.real,a.predictions,a.out)
