import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';

const $ = id => document.getElementById(id);
// Display labels are separate from the frozen scientific manifests.
const structureLabels = {
  '9525':'Deck near FP4/FP5', '49665':'Canopy, north section', '11021':'FP1 support',
  '10912':'Pile cap: check visibility', '5828':'Kiosk, west section',
  roof:'Fixed-block roof', bank_mid:'Riverside structure, middle section',
};
const structureLabel = item => structureLabels[item.id] ?? item.label;
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
const state = { gs:'Loading 3DGS…',bim:'Loading BIM…',las:false };
let bimGroup, lasPoints, manifest, selectionBox, splats, dateData, dates, generation=0, lasLoading;
let qualityManifest, qualityManifests, qualityPoints, qualityLoading, qualityBox;
const loadedBim = new Map();
const reportError = error => { $('error').textContent = String(error.message || error); console.error(error); };
window.addEventListener('unhandledrejection',event => reportError(event.reason));
window.addEventListener('error',event => reportError(event.error || event.message));
function status(){ $('status').textContent = `${state.gs}\n${state.bim}\n${state.las ? `LAS ${dateData.date}: every 10th point` : ''}`; }
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
  $('metrics').textContent=`3DGS → same-date LAS: median ${dateData.median_m.toFixed(2)} m · P90 ${dateData.p90_m.toFixed(2)} m\nWithin 1 m: ${(dateData.within_1m*100).toFixed(1)}% of held-out samples\nDiagnostic agreement, not survey accuracy`;
  selectionText();
  updateQualitySummary();qualitySelection(false);
  state.gs=`Loading 3DGS ${date}…`;state.las=false;status();
  if(splats){scene.remove(splats);splats.dispose();splats=null;}
  if(lasPoints){scene.remove(lasPoints);lasPoints.geometry.dispose();lasPoints.material.dispose();lasPoints=null;}
  lasLoading=null;
  if(qualityPoints){scene.remove(qualityPoints);qualityPoints.geometry.dispose();qualityPoints.material.dispose();qualityPoints=null;}
  qualityLoading=null;updateQualityVisibility();
  try{
    const mesh=new SplatMesh({url:dateData.gs_url,onProgress:event=>{
      if(token===generation && event.total){const percent=Math.round(event.loaded/event.total*100);$('loading').textContent=percent<100?`${percent}%`:'…';state.gs=`Loading 3DGS ${date}: ${percent}%`;status();}
    }});
    splats=mesh;mesh.visible=$('gs').checked;scene.add(mesh);
    await mesh.initialized;
    $('loading').textContent='';
    state.gs=`3DGS ${date}: ${dateData.gaussians.toLocaleString('en-US')} Gaussians`;status();
    if($('las').checked)await loadLas().catch(reportError);
    if($('quality').checked)await loadQuality().catch(reportError);
  }catch(error){$('loading').textContent='Error';state.gs=`Could not load 3DGS ${date}`;status();reportError(error);}
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
  state.bim=`Loading BIM ${version}…`;status();
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
  state.bim=`BIM ${version}: ${count ?? ''} elements · shared coordinate origin`;
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
    ? `Regions: ${qualityManifest.region_count} · centers: ${data.samples.toLocaleString('en-US')}\nMean of region medians: ${data.mean_of_region_medians_m.toFixed(2)} m. Each region has equal weight.`
    : `Shared cells: ${qualityManifest.common_cell_count} · centers: ${data.samples.toLocaleString('en-US')}\nMedian of cell medians: ${data.median_of_cell_medians_m.toFixed(2)} m. Each cell has equal weight.`;
}
function chooseQualityDataset(){
  qualityManifest=qualityManifests[$('quality-source').value];
  if(qualityPoints){scene.remove(qualityPoints);qualityPoints.geometry.dispose();qualityPoints.material.dispose();qualityPoints=null;}
  qualityLoading=null;clearQualityBox();
  $('quality-region').replaceChildren(new Option('All sampled regions',''));
  for(const [index,item] of qualityManifest.regions.entries()){
    const label=qualityManifest.kind==='static'?structureLabel(item):`Zone ${index+1} · ${item.bounds_xy[0]}, ${item.bounds_xy[1]} m`;
    $('quality-region').appendChild(new Option(label,item.id));
  }
  $('quality-region-info').textContent='';
  const fixed=qualityManifest.kind==='static';
  $('quality-description').textContent=fixed
    ? 'Two persistent surfaces. Cranes and equipment lie outside the control volumes. Local diagnostics, not independent accuracy control.'
    : 'The same 20 × 20 m cells across all dates. Only points excluded from fitting.';
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
  $('quality-region-info').textContent=`${dateData.date}: ${row.count} control centers\nMedian ${row.median_m.toFixed(2)} m · 90% within ${row.p90_m.toFixed(2)} m\nLarge distances need inspection; they do not identify the cause of a discrepancy.`;
}
$('quality-region').onchange=()=>{
  $('focus').value='';$('selection').textContent='';
  if(selectionBox){scene.remove(selectionBox);selectionBox.geometry.dispose();selectionBox.material.dispose();selectionBox=null;}
  $('quality').checked=true;qualitySelection(true);loadQuality().catch(reportError);
};
function selectionText(){
  const item=manifest?.landmarks?.find(v=>v.id===$('focus').value);if(!item)return;
  const distance=dateData?.landmark_distances?.[item.id];
  $('selection').textContent=`${structureLabel(item)} · BIM v3 landmark #${item.id}${distance==null?'':` · median surface distance to LAS ${dateData.date}: ${distance.toFixed(2)} m`}. Samples include hidden surfaces.`;
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
  manifest=await fetch('/results/0929_bim/manifest.json').then(r=>{if(!r.ok)throw Error('BIM outputs are missing. Run prepare_bim.py and report_bim.py, then reload.');return r.json();});
  for(const item of manifest.landmarks || []){const option=document.createElement('option');option.value=item.id;option.textContent=structureLabel(item);$('focus').appendChild(option);}
  const dateManifest=await fetch('/results/multidate/manifest.json').then(r=>{if(!r.ok)throw Error('Capture outputs are missing. Complete the baseline export steps in docs/DATA.md, then reload.');return r.json();});
  dates=dateManifest.dates;$('date').replaceChildren();
  for(const item of dates){const option=document.createElement('option');option.value=item.date;option.textContent=`${['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][Number(item.date.slice(0,2))-1]} ${item.date.slice(2)} · ${item.date}`;$('date').appendChild(option);}
  $('date').value='0929';
  await Promise.all([loadBim($('version').value),loadDate('0929')]);
  try{
    const [common,fixed]=await Promise.all(['/results/common_areas/manifest.json','/results/static_structures/manifest.json'].map(url=>fetch(url).then(r=>{if(!r.ok)throw Error('Quality data is missing. Run the common-area and fixed-structure checks, then reload.');return r.json();})));
    qualityManifests={common,static:fixed};chooseQualityDataset();
    $('quality').disabled=false;$('quality-region').disabled=false;$('quality-source').disabled=false;
  }catch(error){$('quality-summary').textContent=error.message;}
}
init().catch(reportError);
window.addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera);});
