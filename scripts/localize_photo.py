"""Localize one calibrated image against the prepared 0929 reference map."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from camera_localization import PARAMETERS,scaled_camera,features,build_matcher,correspondences,estimate_pose
from check_las_vertical import file_hash
from run_camera_localization import OUT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True,help='JSON with width, height, K and 8 FULL_OPENCV distortion parameters')
    parser.add_argument('--output',type=Path,required=True,help='New JSON file; existing results are never overwritten')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    cv2.setNumThreads(2);cv2.setRNGSeed(29)
    protocol=json.loads((OUT/'protocol.json').read_text());calibration=json.loads(args.calibration.read_text())
    assert protocol['parameters']==PARAMETERS
    inputs=json.loads((OUT/'inputs.json').read_text());cache=json.loads((OUT/'reference_features/protocol.json').read_text())
    if cache['inputs']!=inputs or cache['parameters']!=PARAMETERS or cache['source_sha256']!=file_hash(Path(__file__).with_name('camera_localization.py')):
        raise ValueError('Reference cache does not match the recorded experiment.')
    reference_hashes={f['sha256'] for f in inputs if Path(f['path']).name in protocol['reference_names']}
    digest=file_hash(args.image)
    if digest in reference_hashes:raise ValueError('This image is already in the reference index. Use a separate query photo.')
    camera={**calibration,'K':np.array(calibration['K']),'distortion':np.array(calibration['distortion'])}
    if camera['K'].shape!=(3,3) or camera['distortion'].shape!=(8,) or min(camera['width'],camera['height'])<=0:
        raise ValueError('Expected positive image dimensions, a 3x3 K and 8 distortion parameters.')
    with Image.open(args.image) as image:
        if abs(image.width/image.height-camera['width']/camera['height'])>1e-3:
            raise ValueError('Photo and calibration aspect ratios differ; supply the calibration for this photo.')
    K,d,size=scaled_camera(camera);refs=[]
    for name in protocol['reference_names']:
        with np.load(OUT/'reference_features'/(name+'.npz')) as saved:
            refs.append({'name':name,'descriptors':saved['descriptors'],'points':saved['points']})
    xy,descriptor=features(args.image,size)
    pixels,points,retrieved=correspondences(xy,descriptor,refs,build_matcher(refs))
    result,_=estimate_pose(pixels,points,K,d,size)
    result.update(image=str(args.image.resolve()),image_sha256=digest,
        calibration_sha256=file_hash(args.calibration),reference_protocol_sha256=file_hash(OUT/'protocol.json'),
        intrinsics=K.tolist(),distortion=d.tolist(),image_size=list(size),retrieved=retrieved,
        coordinate_origin=[825800,842100,0],meaning='Internal quality gate only; calibrated reference map and overlapping imagery required.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(result['status'],args.output)


if __name__=='__main__':main()
