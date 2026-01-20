import os
import torch
import numpy as np
import cv2


def central_crop(frame, crop_size=224):
    _, width, height, _ = frame.size()
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = (width + crop_size) // 2
    bottom = (height + crop_size) // 2
    return frame[:,left:right,top:bottom,:]


def apply_10_crops_to_clip(frames, crop_size=224):
    T, H, W, C = frames.shape  # Extraemos las dimensiones del tensor
    cropped_clips = []
    for i in range(10):
        cropped_frames = []
        for frame in frames:
            if i == 0:  # Superior izquierda
                crop = frame[:crop_size, :crop_size, :]
            elif i == 1:  # Superior derecha
                crop = frame[:crop_size, W - crop_size:, :]
            elif i == 2:  # Inferior izquierda
                crop = frame[H - crop_size:, :crop_size, :]
            elif i == 3:  # Inferior derecha
                crop = frame[H - crop_size:, W - crop_size:, :]
            elif i == 4:  # Centro
                frame = frame.unsqueeze(0)
                crop = central_crop(frame, crop_size)
                crop = crop.squeeze(0)
            elif i == 5:  # Superior izquierda reflejada
                crop = torch.flip(frame[:crop_size, :crop_size, :], dims=[1])
            elif i == 6:  # Superior derecha reflejada
                crop = torch.flip(frame[:crop_size, W - crop_size:, :], dims=[1])
            elif i == 7:  # Inferior izquierda reflejada
                crop = torch.flip(frame[H - crop_size:, :crop_size, :], dims=[1])
            elif i == 8:  # Inferior derecha reflejada
                crop = torch.flip(frame[H - crop_size:, W - crop_size:, :], dims=[1])
            elif i == 9:  # Centro reflejado
                frame = frame.unsqueeze(0)
                frame = central_crop(frame, crop_size)
                crop = torch.flip(frame, dims=[1])
                crop = crop.squeeze(0)
            cropped_frames.append(crop)
        cropped_frames = torch.stack(cropped_frames)
        cropped_clips.append(cropped_frames)
    cropped_clips = torch.stack(cropped_clips)
    return cropped_clips

def featuresExtractor_OneVideo(net, pathVideoFrames, crops = 1, clipSize = 16, crop_size=224, Mode = "UniFomer-S"):
        listFrames = os.listdir(pathVideoFrames)  
        listFrames = sorted(listFrames) 
        clipListPathFullFrame = []
        for nameframe in listFrames:
            pathFullFrames = os.path.join(pathVideoFrames, nameframe)
            clipListPathFullFrame.append(pathFullFrames)
        clip = np.array([cv2.imread(frame) for frame in clipListPathFullFrame]) 
        clip = torch.from_numpy(clip)
        clip = clip.to(torch.float32) / 255.0
        if crops == 1:
            X = central_crop(clip, crop_size)
        if crops == 10:
            X = apply_10_crops_to_clip(clip, crop_size)
        
        if X.size(0) == clipSize:
            X = X.unsqueeze(0)
        features = []
        for cropX in X:
            cropX_1= cropX
            cropX_1 = cropX_1.cuda()
            cropX_1 = cropX_1.permute(3, 0, 1, 2) 
            if Mode == "I3D-1024":
                cropX_1 = cropX_1.unsqueeze(0)
                with torch.no_grad():
                    model_features = net.extract_features(cropX_1)
                    model_features = model_features[0].unsqueeze(0)
            elif Mode == "TSM":
                cropX_1 = cropX_1.permute(1, 0, 2, 3)
                with torch.no_grad():
                    stn_features = net(cropX_1)
                    model_features = stn_features[0].unsqueeze(0)
            features.append(model_features)
        features = torch.stack(features)
        features = features.permute(1,0,2)
        features = features.squeeze(0)
        return features



def featureStractor_I3D_inceptionV1_1024(crops, clipSize, crop_size, pathMainDataframes):
    from resourcesFE.I3D_Inception.pytorch_i3d import InceptionI3d
    load_model = "./resourcesFE/I3D_Inception/rgb_imagenet.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = InceptionI3d(400, in_channels=3)
    net.load_state_dict(torch.load(load_model))
    net.to(device)
    net.eval()
    features  = featuresExtractor_OneVideo(net, pathMainDataframes, crops, clipSize, crop_size=crop_size)
    return features

def featureStractor_TSM_2048(crops, clipSize, crop_size, pathMainDataframes):
    from resourcesFE.TSM.ops.models import TSN

    num_class = 400
    modality = "RGB"
    this_arch = "resnet50"
    net = TSN(num_class, 
                clipSize, 
                modality,
                base_model=this_arch,
                consensus_type='avg',
                img_feature_dim= 256,
                pretrain='imagenet',
                is_shift= True,
                shift_div=8,
                shift_place= "blockres",
                non_local=False
                )
    net = net.cuda()

    weights_list = ["./resourcesFE/TSM/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment16_e50.pth"]
    if 'tpool' in weights_list[0]:
        from resourcesFE.TSM.ops.temporal_shift import make_temporal_pool
        make_temporal_pool(net.base_model, 16)  # since DataParallel

    checkpoint = torch.load(weights_list[0], weights_only=True)
    try:
        checkpoint = checkpoint['state_dict']
    except Exception as e:
        print(f"[ERROR]: {e}")
        print(f"Loading checkpoint without key 'state_dict'")

    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
    replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                    'base_model.classifier.bias': 'new_fc.bias',
                    }
    for k, v in replace_dict.items():
        if k in base_dict:
            base_dict[v] = base_dict.pop(k)

    net.load_state_dict(base_dict)

    features  = featuresExtractor_OneVideo(net, pathMainDataframes, crops, clipSize, crop_size=crop_size, Mode = "TSM")

    return features

