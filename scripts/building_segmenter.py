import numpy as np
from semantic_projection import GROUPS,PARAMETERS
STRUCTURE_PARAMETERS={'quarter_turns':[0,1,2,3]}
LOVEDA_GROUP_IDS=[[0,1],[2],[3],[5],[6,7],[4],[]]

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
