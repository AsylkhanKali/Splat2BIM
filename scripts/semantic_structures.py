"""LoveDA labels and correspondence banks fixed before ICP (semantic revision 2)."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from object_consensus import match
from semantic_projection import PARAMETERS,GROUPS,rasterize

STRUCTURE_PARAMETERS={'context_fill_pixels':24,'context_median_kernel':3,
    'neighbor_start_rms_m':2.,'quarter_turns':[0,1,2,3]}
LOVEDA_GROUP_IDS=[[0,1],[2],[3],[5],[6,7],[4],[]]


def context_projection(points,colors,up):
    image,indices,metadata=rasterize(points,colors,up,PARAMETERS['raster_size'])
    observed=indices>=0;distance,nearest=distance_transform_edt(~observed,return_indices=True)
    fill=(~observed)&(distance<=STRUCTURE_PARAMETERS['context_fill_pixels'])
    image[fill]=image[nearest[0][fill],nearest[1][fill]]
    image=cv2.medianBlur(image,STRUCTURE_PARAMETERS['context_median_kernel'])
    return image,indices,{**metadata,'context_only_filled_pixels':int(fill.sum())}


class StructureSegmenter:
    def __init__(self,folder):
        import torch
        from transformers import SegformerConfig,SegformerForSemanticSegmentation,SegformerImageProcessor
        torch.set_num_threads(2);torch.manual_seed(29);torch.use_deterministic_algorithms(True)
        self.torch=torch;self.processor=SegformerImageProcessor.from_pretrained(folder,local_files_only=True)
        config=SegformerConfig.from_pretrained(folder,local_files_only=True)
        assert [config.id2label[i] for i in range(8)]==['Ignore','Background','Building','Road','Water','Barren','Forest','Agricultural']
        self.model=SegformerForSemanticSegmentation(config)
        self.model.load_state_dict(torch.load(folder/'pytorch_model.bin',map_location='cpu',weights_only=True),strict=True)
        self.model.eval()

    def predict(self,image):
        torch=self.torch;h,w=image.shape[:2];tile=PARAMETERS['tile_size'];result=np.zeros((h,w,len(GROUPS)),np.float32)
        with torch.inference_mode():
            for y in range(0,h,tile):
                for x in range(0,w,tile):
                    patch=image[y:y+tile,x:x+tile];total=np.zeros((*patch.shape[:2],len(GROUPS)),np.float32)
                    for turn in STRUCTURE_PARAMETERS['quarter_turns']:
                        rotated=np.ascontiguousarray(np.rot90(patch,turn))
                        probabilities=self.model(**self.processor(images=rotated,return_tensors='pt')).logits.softmax(dim=1)
                        grouped=torch.stack([probabilities[:,ids].sum(1) for ids in LOVEDA_GROUP_IDS],dim=1)
                        grouped=torch.nn.functional.interpolate(grouped,size=rotated.shape[:2],mode='bilinear',align_corners=False)
                        total+=np.rot90(grouped[0].permute(1,2,0).numpy(),-turn)
                    result[y:y+tile,x:x+tile]=total/len(STRUCTURE_PARAMETERS['quarter_turns'])
        return result


def correspondence_bank(matrices,index,anchors,source,target):
    """Union seed identities from the same 2 m neighborhood, before any fitting.

    Later surviving_pairs retains only a one-to-one assignment inside unchanged
    geometric gates. This bank cannot create identities from a refined pose.
    """
    moved=np.einsum('nj,kij->kni',anchors,matrices[:,:3,:3])+matrices[:,None,:3,3]
    rms=np.sqrt(np.mean(np.sum((moved-moved[index])**2,axis=2),axis=1))
    neighbors=np.flatnonzero(rms<STRUCTURE_PARAMETERS['neighbor_start_rms_m']);bank={}
    for neighbor in neighbors:
        for pair in match(source,target,matrices[neighbor]):
            key=(pair['source_id'],pair['target_id'])
            if key not in bank:bank[key]={**pair,'proposed_by_pool_indices':[]}
            bank[key]['proposed_by_pool_indices'].append(int(neighbor))
    return [bank[key] for key in sorted(bank)],neighbors.tolist()
