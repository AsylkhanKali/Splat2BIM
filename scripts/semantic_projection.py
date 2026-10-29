"""Frozen drone SegFormer predictions on independently rendered training clouds."""
import json

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

GROUPS=['unknown','built','paved','bare_ground','vegetation','water','movable']
GROUP_IDS=[[0,21,22,23],[9,10,11,12,13,14],[1],[2,4,6],[3,8,19,20],[5,7],[15,16,17,18]]
COLORS=np.array([[130,130,130],[234,158,42],[166,109,209],[173,116,76],[56,169,98],[65,155,222],[235,80,105]],np.uint8)
PARAMETERS={'raster_size':1024,'tile_size':512,'quarter_turns':[0,1,2,3],
    'crop_quantiles':[.002,.998],'hole_fill_pixels':4,'confidence':.6,
    'static_groups':[1,2,3,4],'min_class_points':128,'min_shared_groups':2,'max_source_points_per_class':512,
    'semantic_distance_cap_m':20.,'semantic_distance_quantile':.6,'semantic_weight':.5}


def rasterize(points,colors,up,size=1024):
    """PCA heading and supplied training-plane normal; no destination transform."""
    up=np.asarray(up,float);up=up/np.linalg.norm(up);center=np.median(points,axis=0)
    centered=points-center;tangent=centered-np.outer(centered@up,up)
    _,vectors=np.linalg.eigh(np.cov(tangent.T));x=vectors[:,-1]
    x-=up*(x@up);x/=np.linalg.norm(x)
    if x[np.argmax(abs(x))]<0:x=-x
    frame=np.c_[x,np.cross(up,x),up];local=centered@frame
    lo,hi=np.quantile(local[:,:2],PARAMETERS['crop_quantiles'],axis=0)
    step=max(hi-lo)/(size-8);mid=(hi+lo)/2;origin=mid-step*(size-1)/2
    uv=np.rint((local[:,:2]-origin)/step).astype(int);uv[:,1]=size-1-uv[:,1]
    inside=np.all((uv>=0)&(uv<size),axis=1);ids=np.flatnonzero(inside);pixel=uv[ids,1]*size+uv[ids,0]
    order=np.lexsort((ids,-local[ids,2],pixel));_,first=np.unique(pixel[order],return_index=True)
    winners=ids[order[first]];indices=np.full((size,size),-1,np.int64)
    indices[uv[winners,1],uv[winners,0]]=winners;observed=indices>=0
    image=np.full((size,size,3),128,np.uint8);image[observed]=np.uint8(np.clip(colors[indices[observed]],0,1)*255)
    distance,nearest=distance_transform_edt(~observed,return_indices=True)
    fill=(~observed)&(distance<=PARAMETERS['hole_fill_pixels'])
    image[fill]=image[nearest[0][fill],nearest[1][fill]]
    metadata={'frame':frame.tolist(),'center':center.tolist(),'origin_xy':origin.tolist(),
        'pixel_size_source_units':float(step),'visible_points':int(observed.sum()),'input_points':len(points)}
    return image,indices,metadata


class Segmenter:
    def __init__(self,folder):
        import torch
        from transformers import SegformerForSemanticSegmentation,SegformerImageProcessor
        torch.set_num_threads(2);torch.manual_seed(29);torch.use_deterministic_algorithms(True)
        self.torch=torch;self.processor=SegformerImageProcessor.from_pretrained(folder,local_files_only=True)
        self.model=SegformerForSemanticSegmentation.from_pretrained(folder,local_files_only=True,use_safetensors=True).eval()
        config=json.loads((folder/'config.json').read_text())
        assert [config['id2label'][str(i)] for i in [1,9,17,19]]==['paved-area','roof','car','tree']

    def predict(self,image):
        torch=self.torch;h,w=image.shape[:2];tile=PARAMETERS['tile_size']
        result=np.zeros((h,w,len(GROUPS)),np.float32)
        with torch.inference_mode():
            for y in range(0,h,tile):
                for x in range(0,w,tile):
                    patch=image[y:y+tile,x:x+tile];total=np.zeros((*patch.shape[:2],len(GROUPS)),np.float32)
                    for turn in PARAMETERS['quarter_turns']:
                        rotated=np.ascontiguousarray(np.rot90(patch,turn))
                        inputs=self.processor(images=rotated,return_tensors='pt')
                        probabilities=self.model(**inputs).logits.softmax(dim=1)
                        grouped=torch.stack([probabilities[:,ids].sum(1) for ids in GROUP_IDS],dim=1)
                        grouped=torch.nn.functional.interpolate(grouped,size=rotated.shape[:2],mode='bilinear',align_corners=False)
                        total+=np.rot90(grouped[0].permute(1,2,0).numpy(),-turn)
                    result[y:y+tile,x:x+tile]=total/len(PARAMETERS['quarter_turns'])
        return result


def visible_labels(points,indices,probabilities):
    observed=indices>=0;ids=indices[observed];prob=probabilities[observed]
    labels=prob.argmax(1).astype(np.uint8);confidence=prob.max(1)
    labels[confidence<PARAMETERS['confidence']]=0
    return {'points':points[ids],'input_indices':ids,'probabilities':prob.astype(np.float16),
        'labels':labels,'confidence':confidence.astype(np.float32)}


def save_images(folder,side,image,indices,probabilities):
    labels=probabilities.argmax(2);labels[probabilities.max(2)<PARAMETERS['confidence']]=0
    colors=COLORS[labels].copy();colors[indices<0]=[245,245,245]
    cv2.imwrite(str(folder/f'{side}_rgb.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(folder/f'{side}_semantic.png'),cv2.cvtColor(colors,cv2.COLOR_RGB2BGR))
