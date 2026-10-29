import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';

const $ = id => document.getElementById(id);
const panel=document.querySelector('aside');
$('panel').setAttribute('aria-expanded',String(innerWidth>700));
$('panel').onclick=()=>{
  const open=getComputedStyle(panel).display==='none';
  panel.classList.toggle('open',open);panel.classList.toggle('closed',!open);
  $('panel').setAttribute('aria-expanded',String(open));
};
const scene = new THREE.Scene();
scene.background = new THREE.Color('#151b22');
const renderer = new THREE.WebGLRenderer({ antialias:false });
renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));
renderer.setSize(innerWidth,innerHeight);
document.body.appendChild(renderer.domElement);
const camera = new THREE.PerspectiveCamera(55,innerWidth/innerHeight,.2,4000);
camera.up.set(0,0,1);
const controls = new OrbitControls(camera,renderer.domElement);
controls.enableDamping = true;
const spark = new SparkRenderer({renderer});
scene.add(spark);
const state = { gs:'Загрузка 3DGS…',bim:'Загрузка BIM…',las:false };
let bimGroup, lasPoints, manifest, selectionBox, splats, dateData, dates, generation=0, lasLoading;
let qualityManifest, qualityManifests, qualityPoints, qualityLoading, qualityBox;
const loadedBim = new Map();
const reportError = error => { $('error').textContent = String(error.message || error); console.error(error); };
window.addEventListener('unhandledrejection',event => reportError(event.reason));
window.addEventListener('error',event => reportError(event.error || event.message));
function status(){ $('status').textContent = `${state.gs}\n${state.bim}\n${state.las ? `LAS ${dateData.date}: каждый 10-й пункт` : ''}`; }
function view(position,target){ camera.up.set(0,0,1);camera.position.fromArray(position);controls.target.fromArray(target);controls.update(); }
function overview(){
  const target=new THREE.Vector3(-25,90,5);
  const position=new THREE.Vector3(-305,-350,305).normalize().multiplyScalar(camera.aspect<1?1000:700).add(target);
  view(position.toArray(),target.toArray());
}
overview();
$('overview').onclick = overview;
$('top').onclick = () => view([-20,75,720],[-20,76,5]);
$('photo').onclick = () => { if(manifest?.photo_view) view(manifest.photo_view.position,manifest.photo_view.target); };
async function loadDate(date){
  const token=++generation;
  $('date').disabled=true;$('error').textContent='';
  $('loading').textContent='…';
  dateData=dates.find(item=>item.date===date);
  $('heading').textContent=`${date} · The Loop`;
  $('metrics').textContent=`3DGS → свой LAS: медиана ${dateData.median_m.toFixed(2)} м · P90 ${dateData.p90_m.toFixed(2)} м\nБлиже 1 м: ${(dateData.within_1m*100).toFixed(1)}% отложенных проб\nДиагностическая проверка, не геодезическая точность`;
  selectionText();
  updateQualitySummary();qualitySelection(false);
  state.gs=`Загрузка 3DGS ${date}…`;state.las=false;status();
  if(splats){scene.remove(splats);splats.dispose();splats=null;}
  if(lasPoints){scene.remove(lasPoints);lasPoints.geometry.dispose();lasPoints.material.dispose();lasPoints=null;}
  lasLoading=null;
  if(qualityPoints){scene.remove(qualityPoints);qualityPoints.geometry.dispose();qualityPoints.material.dispose();qualityPoints=null;}
  qualityLoading=null;updateQualityVisibility();
  try{
    const mesh=new SplatMesh({url:dateData.gs_url,onProgress:event=>{
      if(token===generation && event.total){const percent=Math.round(event.loaded/event.total*100);$('loading').textContent=percent<100?`${percent}%`:'…';state.gs=`Загрузка 3DGS ${date}: ${percent}%`;status();}
    }});
    splats=mesh;mesh.visible=$('gs').checked;scene.add(mesh);
    await mesh.initialized;
    $('loading').textContent='';
    state.gs=`3DGS ${date}: ${dateData.gaussians.toLocaleString('ru-RU')} гауссиан`;status();
    if($('las').checked)await loadLas().catch(reportError);
    if($('quality').checked)await loadQuality().catch(reportError);
  }catch(error){$('loading').textContent='Ошибка';state.gs=`Не удалось загрузить 3DGS ${date}`;status();reportError(error);}
  finally{$('date').disabled=false;}
}
$('date').onchange=()=>loadDate($('date').value);
$('gs').onchange=()=>{if(splats)splats.visible=$('gs').checked;};
function updateBim(){
  if(!bimGroup)return;
  bimGroup.visible=$('bim').checked;
  bimGroup.children[0].material.opacity=Number($('opacity').value);
  bimGroup.children[1].visible=$('xray').checked;
  if(selectionBox)selectionBox.visible=$('bim').checked;
}
for(const id of ['bim','xray','opacity']) $(id).oninput=updateBim;
async function loadBim(version){
  state.bim=`Загрузка BIM ${version}…`;status();
  if(!loadedBim.has(version)){
    const geometry=await new PLYLoader().loadAsync(`/results/0929_bim/${version}_local.ply`);
    geometry.computeVertexNormals();
    const group=new THREE.Group();
    group.add(new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({color:'#ffad44',transparent:true,opacity:.35,side:THREE.DoubleSide,depthWrite:false})));
    const edges=new THREE.LineSegments(new THREE.EdgesGeometry(geometry,25),new THREE.LineBasicMaterial({color:'#ffe0a3',transparent:true,opacity:.4,depthTest:false}));
    edges.renderOrder=10;group.add(edges);group.visible=false;scene.add(group);loadedBim.set(version,group);
  }
  if($('version').value!==version)return;
  if(bimGroup)bimGroup.visible=false;
  bimGroup=loadedBim.get(version);updateBim();
  const count=manifest?.versions?.[version]?.converted;
  state.bim=`BIM ${version}: ${count ?? ''} элементов · общий сдвиг координат`;
  status();
}
$('version').onchange=()=>loadBim($('version').value).catch(reportError);
async function loadLas(){
  if(!lasPoints && $('las').checked){
    const token=generation;
    if(!lasLoading)lasLoading=new PLYLoader().loadAsync(dateData.las_url);
    const geometry=await lasLoading;
    if(token!==generation){geometry.dispose();return;}
    if(lasPoints){lasPoints.visible=$('las').checked;return;}
    lasPoints=new THREE.Points(geometry,new THREE.PointsMaterial({size:.4,vertexColors:true}));scene.add(lasPoints);state.las=true;status();
  }
  if(lasPoints)lasPoints.visible=$('las').checked;
}
$('las').onchange=()=>loadLas().catch(reportError);
function clearQualityBox(){
  if(qualityBox){scene.remove(qualityBox);qualityBox.geometry.dispose();qualityBox.material.dispose();qualityBox=null;}
}
function updateQualityVisibility(){
  const visible=$('quality').checked;
  if(qualityPoints)qualityPoints.visible=visible;
  if(qualityBox)qualityBox.visible=visible;
  $('quality-legend').hidden=!(visible && qualityPoints);
}
function updateQualitySummary(){
  const data=qualityManifest?.dates?.[dateData?.date];if(!data)return;
  $('quality-summary').textContent=qualityManifest.kind==='static'
    ? `Участков: ${qualityManifest.region_count} · центров: ${data.samples.toLocaleString('ru-RU')}\nСреднее медиан участков: ${data.mean_of_region_medians_m.toFixed(2)} м. Каждый участок имеет одинаковый вес.`
    : `Общих ячеек: ${qualityManifest.common_cell_count} · центров: ${data.samples.toLocaleString('ru-RU')}\nМедиана медиан ячеек: ${data.median_of_cell_medians_m.toFixed(2)} м. Каждая ячейка имеет одинаковый вес.`;
}
function chooseQualityDataset(){
  qualityManifest=qualityManifests[$('quality-source').value];
  if(qualityPoints){scene.remove(qualityPoints);qualityPoints.geometry.dispose();qualityPoints.material.dispose();qualityPoints=null;}
  qualityLoading=null;clearQualityBox();
  $('quality-region').replaceChildren(new Option('Все участки выборки',''));
  for(const item of qualityManifest.regions)$('quality-region').appendChild(new Option(item.label,item.id));
  $('quality-region-info').textContent='';
  const fixed=qualityManifest.kind==='static';
  $('quality-description').textContent=fixed
    ? 'Две сохраняющиеся поверхности. Краны и техника вне контрольных объёмов. Локальная диагностика, не независимый контроль точности.'
    : 'Одинаковые ячейки 20 × 20 м во всех датах. Только точки, исключённые из подгонки.';
  $('quality-report').href=fixed?'/results/static_structures/REPORT.md':'/results/common_areas/REPORT.md';
  updateQualitySummary();updateQualityVisibility();loadQuality().catch(reportError);
}
$('quality-source').onchange=chooseQualityDataset;
async function loadQuality(){
  if(!qualityManifest || !$('quality').checked){updateQualityVisibility();return;}
  if(!qualityPoints){
    const token=generation;
    if(!qualityLoading)qualityLoading=new PLYLoader().loadAsync(qualityManifest.dates[dateData.date].points_url);
    const pending=qualityLoading;let geometry;
    try{geometry=await pending;}catch(error){if(pending===qualityLoading)qualityLoading=null;throw error;}
    if(token!==generation || pending!==qualityLoading){geometry.dispose();return;}
    if(!qualityPoints){
      qualityPoints=new THREE.Points(geometry,new THREE.PointsMaterial({size:.65,vertexColors:true,transparent:true,opacity:.95,depthTest:false,depthWrite:false}));
      qualityPoints.renderOrder=30;scene.add(qualityPoints);
    }
  }
  updateQualityVisibility();
}
$('quality').onchange=()=>loadQuality().catch(reportError);
function qualitySelection(moveCamera){
  clearQualityBox();
  const region=qualityManifest?.regions?.find(item=>item.id===$('quality-region').value);
  if(!region){$('quality-region-info').textContent='';return;}
  const row=region.dates[dateData.date],b=region.bounds_xy;
  const center=[(b[0]+b[2])/2,(b[1]+b[3])/2,row.z_median];
  if(moveCamera)view([center[0]-35,center[1]-45,center[2]+45],center);
  if(region.polygon){
    const vertices=[],polygon=region.polygon,[lo,hi]=region.z_range;
    for(let i=0;i<polygon.length;i++){
      const a=polygon[i],next=polygon[(i+1)%polygon.length];
      vertices.push(...a,lo,...next,lo,...a,hi,...next,hi,...a,lo,...a,hi);
    }
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));
    qualityBox=new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({color:0x39e5e2}));
  }else qualityBox=new THREE.Box3Helper(new THREE.Box3(new THREE.Vector3(b[0],b[1],row.z05-.5),new THREE.Vector3(b[2],b[3],row.z95+.5)),0x39e5e2);
  qualityBox.material.depthTest=false;qualityBox.material.transparent=true;qualityBox.renderOrder=20;qualityBox.visible=$('quality').checked;scene.add(qualityBox);
  $('quality-region-info').textContent=`${dateData.date}: ${row.count} контрольных центров\nМедиана ${row.median_m.toFixed(2)} м · 90% ближе ${row.p90_m.toFixed(2)} м\nБольшое расстояние требует осмотра; оно не определяет причину расхождения.`;
}
$('quality-region').onchange=()=>{
  $('focus').value='';$('selection').textContent='';
  if(selectionBox){scene.remove(selectionBox);selectionBox.geometry.dispose();selectionBox.material.dispose();selectionBox=null;}
  $('quality').checked=true;qualitySelection(true);loadQuality().catch(reportError);
};
function selectionText(){
  const item=manifest?.landmarks?.find(v=>v.id===$('focus').value);if(!item)return;
  const distance=dateData?.landmark_distances?.[item.id];
  $('selection').textContent=`${item.label} · ориентир BIM v3 #${item.id}${distance==null?'':` · медиана поверхности до LAS ${dateData.date}: ${distance.toFixed(2)} м`}. Пробы включают скрытые поверхности.`;
}
$('focus').onchange=()=>{
  const item=manifest?.landmarks?.find(v=>v.id===$('focus').value);if(!item)return;
  $('quality-region').value='';$('quality-region-info').textContent='';clearQualityBox();
  const c=item.center;view([c[0]-25,c[1]-35,c[2]+35],c);
  if(selectionBox){scene.remove(selectionBox);selectionBox.geometry.dispose();selectionBox.material.dispose();}
  selectionBox=new THREE.Box3Helper(new THREE.Box3(new THREE.Vector3().fromArray(item.bounds_min),new THREE.Vector3().fromArray(item.bounds_max)),0x39e5e2);
  selectionBox.material.depthTest=false;selectionBox.material.transparent=true;selectionBox.renderOrder=20;selectionBox.visible=$('bim').checked;scene.add(selectionBox);
  selectionText();
};
async function init(){
  manifest=await fetch('/results/0929_bim/manifest.json').then(r=>{if(!r.ok)throw Error('BIM ещё подготавливается — обновите страницу после завершения');return r.json();});
  for(const item of manifest.landmarks || []){const option=document.createElement('option');option.value=item.id;option.textContent=item.label;$('focus').appendChild(option);}
  const dateManifest=await fetch('/results/multidate/manifest.json').then(r=>{if(!r.ok)throw Error('Съёмки ещё подготавливаются');return r.json();});
  dates=dateManifest.dates;$('date').replaceChildren();
  for(const item of dates){const option=document.createElement('option');option.value=item.date;option.textContent=`${item.label} · ${item.date}`;$('date').appendChild(option);}
  $('date').value='0929';
  await Promise.all([loadBim($('version').value),loadDate('0929')]);
  try{
    const [common,fixed]=await Promise.all(['/results/common_areas/manifest.json','/results/static_structures/manifest.json'].map(url=>fetch(url).then(r=>{if(!r.ok)throw Error('Данные проверки пока недоступны');return r.json();})));
    qualityManifests={common,static:fixed};chooseQualityDataset();
    $('quality').disabled=false;$('quality-region').disabled=false;$('quality-source').disabled=false;
  }catch(error){$('quality-summary').textContent=error.message;}
}
init().catch(reportError);
window.addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera);});
