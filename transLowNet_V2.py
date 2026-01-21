import os
import cv2
import time
import math
import torch
import numpy as np
from datetime import datetime
from scipy.ndimage import label
from models_Inference import Detector_VAD, violenceOneCrop, Model_V3_Connection
from collections import deque

import signal
import sys

running = True

def handle_sigint(signum, frame):
    global running
    print("SIGINT recibido, cerrando limpiamente...")
    running = False

signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)




def background_subtraction(frames):
    fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=100, detectShadows=True)
    fgbg.setNMixtures(3)
    fgbg.setVarThresholdGen(4.0)
    masks = []
    countMask = 0
    for frame in frames:
        fg_mask = fgbg.apply(frame)  # Aplicar sustracción de fondo
        # plt.imshow(fg_mask, cmap='gray')  # Usa el mapa de colores en escala de grises
        # plt.axis('off')  # Ocultar los ejes
        # plt.savefig(f'daveFileTest/Mask/{countMask}.png', bbox_inches='tight', pad_inches=0) 
        countMask += 1
        masks.append(fg_mask)
    return masks



def filter_small_regions(binary_matrix, min_area=500):
    labeled_matrix, num_features = label(binary_matrix)

    if num_features == 1:
        return binary_matrix

    region_sizes = np.bincount(labeled_matrix.ravel())
    region_sizes[0] = 0 

    max_region_label = np.argmax(region_sizes)
    
    filtered_matrix = np.zeros_like(binary_matrix)
    for label_idx, size in enumerate(region_sizes):
        if size >= min_area or label_idx == max_region_label:  # Mantener si es grande o la más grande
            filtered_matrix[labeled_matrix == label_idx] = 1

    return filtered_matrix


def bbx_overlap(bbx1, bbx2):
    """Verifica si dos bounding boxes se superponen."""
    x1, y1, w1, h1 = bbx1
    x2, y2, w2, h2 = bbx2

    if x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2:
        return True
    return False

def merge_bbxs(bbxs):
    """Fusiona una lista de bounding boxes en uno solo."""
    if not bbxs:
        return None

    x_min = min(bbx[0] for bbx in bbxs)
    y_min = min(bbx[1] for bbx in bbxs)
    x_max = max(bbx[0] + bbx[2] for bbx in bbxs)
    y_max = max(bbx[1] + bbx[3] for bbx in bbxs)

    return x_min, y_min, x_max - x_min, y_max - y_min


def mergeOverlap_Bbox(bBox_list):
    fullBbox = []
    bBox_Used = [False] * len(bBox_list)

    for i in range(len(bBox_list)):
        if bBox_Used[i]:
            continue

        bbxsMerge = [bBox_list[i]]
        bBox_Used[i] = True

        for j in range(i + 1, len(bBox_list)):
            if bBox_Used[j]:
                continue

            if bbx_overlap(bBox_list[i], bBox_list[j]):
                bbxsMerge.append(bBox_list[j])
                bBox_Used[j] = True

        if len(bbxsMerge) > 1:  # Solo fusiona si hay más de un bounding box superpuesto
            fusionado = merge_bbxs(bbxsMerge)
            fullBbox.append(fusionado)
        else:
            fullBbox.append(bBox_list[i])  # Si no hay superposición, agrega el bounding box original

    return fullBbox
def min_edge_distance(bbx1, bbx2):
    """Calcula la distancia mínima entre los bordes de dos bounding boxes."""
    x1, y1, w1, h1 = bbx1
    x2, y2, w2, h2 = bbx2

    # Calcular los bordes de los bounding boxes
    left1, right1, top1, bottom1 = x1, x1 + w1, y1, y1 + h1
    left2, right2, top2, bottom2 = x2, x2 + w2, y2, y2 + h2

    # Calcular las distancias entre los bordes
    dist_left = abs(right1 - left2)
    dist_right = abs(right2 - left1)
    dist_top = abs(bottom1 - top2)
    dist_bottom = abs(bottom2 - top1)

    # Calcular la distancia mínima
    if right1 < left2:
        x_dist = dist_left
    elif right2 < left1:
        x_dist = dist_right
    else:
        x_dist = 0

    if bottom1 < top2:
        y_dist = dist_top
    elif bottom2 < top1:
        y_dist = dist_bottom
    else:
        y_dist = 0

    return math.sqrt(x_dist**2 + y_dist**2)


def merge_closest_bbxs_iterative(bBox_list, max_distance):
    """
    Fusiona los bounding boxes más cercanos entre sí de forma iterativa.

    Args:
        bBox_list (list): Lista de bounding boxes (x, y, w, h).
        max_distance (int): Distancia máxima para considerar dos bounding boxes cercanos.

    Returns:
        list: Lista de bounding boxes fusionados y no superpuestos.
    """

    fullBbox = bBox_list[:]  # Copia la lista original
    cambios = True

    while cambios:
        cambios = False
        nuevos_fullBbox = []
        bBox_Used = [False] * len(fullBbox)

        for i in range(len(fullBbox)):
            if bBox_Used[i]:
                continue

            bbxsMerge = [fullBbox[i]]
            bBox_Used[i] = True

            for j in range(i + 1, len(fullBbox)):
                if bBox_Used[j]:
                    continue

                if min_edge_distance(fullBbox[i], fullBbox[j]) <= max_distance:
                    bbxsMerge.append(fullBbox[j])
                    bBox_Used[j] = True
                    cambios = True  # Se realizó una fusión

            if len(bbxsMerge) > 1:
                fusionado = merge_bbxs(bbxsMerge)
                nuevos_fullBbox.append(fusionado)
            else:
                nuevos_fullBbox.append(fullBbox[i])

        fullBbox = nuevos_fullBbox[:]  # Actualiza la lista de bounding boxes

    return fullBbox

def find_largest_bbx(bBox_list):

    if not bBox_list:
        return None

    largest_bbx = bBox_list[0]
    largest_area = largest_bbx[2] * largest_bbx[3]  # Área inicial

    for bbx in bBox_list[1:]:
        area = bbx[2] * bbx[3]
        if area > largest_area:
            largest_bbx = bbx
            largest_area = area

    return largest_bbx


def magMotionV2 (saveFramesFull_path):
    listwholeFrames = saveFramesFull_path # Se usa todo para mejor version 
    
    
    framesOri = []
    for _, frameNameAbnormal in enumerate(listwholeFrames):
        fullNameFrameAbnormal = frameNameAbnormal
        frameAbnormal = cv2.imread(fullNameFrameAbnormal)
        framesOri.append(frameAbnormal)
        
    
    WOBck = background_subtraction(framesOri)
    WOBck = np.mean(WOBck[1:], axis=0)  
    
    
    min_val = np.min(WOBck)
    max_val = np.max(WOBck)
    WOBck = (WOBck - min_val) / (max_val - min_val)
    
    WOBck = np.where(WOBck == 0, 0, 1)
    WOBck = filter_small_regions(WOBck, min_area=5000) # cambiar para mejorar mejor 35, 2000 para video 15, 5000
    
    WOBck = WOBck.astype(np.float32)  

    _, binary_mask = cv2.threshold(WOBck, 0, 1, cv2.THRESH_BINARY)
    binary_mask = (binary_mask * 255).astype(np.uint8)



    # plt.imshow(binary_mask, cmap='gray')  # Usa el mapa de colores en escala de grises
    # plt.axis('off')  # Ocultar los ejes
    # plt.savefig('saveMask/matrix_image.png', bbox_inches='tight', pad_inches=0)  # Guardar la imagen


    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bbx_list_GCFD = [cv2.boundingRect(cnt) for cnt in contours]
    bbx_list_GCFD = mergeOverlap_Bbox(bbx_list_GCFD)
    bbx_list_GCFD = merge_closest_bbxs_iterative(bbx_list_GCFD, 30)
    
    x1, y1, w1, h1= find_largest_bbx(bbx_list_GCFD)
    x2, y2 = x1 + w1, y1 + h1

    return x1, y1, x2, y2


def indexClipSampling(lengthWholeVideo, sampling_rate, clipSize):
    listIndexVideo = []
    flagList = 0
    total_frames_needed = sampling_rate * clipSize
    max_clips = lengthWholeVideo // total_frames_needed

    for _ in range(max_clips):
        listIndexClip = []
        for _ in range(clipSize):
            indexFrame = flagList % lengthWholeVideo  # rebobinar si se pasa
            flagList += sampling_rate
            listIndexClip.append(indexFrame)
        listIndexVideo.append(listIndexClip)

    return listIndexVideo

def centralCrop(clip, target_size=320):
    _, h, w, _ = clip.shape
    top = (h - target_size) // 2
    left = (w - target_size) // 2
    return clip[:, top:top + target_size, left:left + target_size, :]

def featuresExtractor_OneVideo(net, pathVideoFrames, sampling_rate, clipSize = 16, crop_size=224):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    listFrames = os.listdir(pathVideoFrames)
    listFrames = sorted(listFrames)
    addIndexFrame = 0
    while len(listFrames) % clipSize != 0:
        listFrames.append(listFrames[addIndexFrame])
        addIndexFrame += 1

    indexWholeVideo = indexClipSampling(len(listFrames), sampling_rate, clipSize)

    for clips in range(len(indexWholeVideo)):
        clipFrames = []

        for indexClip in indexWholeVideo[clips]:
            clipFrames.append(listFrames[indexClip])

        clipListPathFullFrame = []
        for nameframe in clipFrames:
            pathFullFrames = os.path.join(pathVideoFrames, nameframe)
            clipListPathFullFrame.append(pathFullFrames)
        clip = []
        for frame_path in clipListPathFullFrame:
            img = cv2.imread(frame_path)
            if img is None:
                raise RuntimeError(f"Failed to read frame: {frame_path}")
            h, w = img.shape[:2]
            scale = crop_size / min(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            clip.append(rgb)

        clip = np.stack(clip, axis=0)
        clip = torch.from_numpy(clip).float() / 255.0
        clip = centralCrop(clip, target_size=crop_size)
        clip = clip.permute(3, 0, 1, 2)
        mean_tensor = torch.tensor(mean, device=clip.device).view(1, 3, 1, 1, 1)
        std_tensor = torch.tensor(std, device=clip.device).view(1, 3, 1, 1, 1)
        clip = (clip - mean_tensor) / std_tensor # torch.Size([1, 3, 16, 320, 320])
        clip = clip.float().to(next(net.parameters()).device)

        with torch.no_grad():
            modelFeatures = net(clip)  # salida con shape: [1, C, T, H, W]
            modelFeatures = modelFeatures.squeeze(0)  # shape: [C, T, H, W]
            modelFeatures = modelFeatures.mean(dim=(1, 2, 3))  # promedio sobre T, H, W → shape: [C]
            modelFeatures = modelFeatures.unsqueeze(0)  # shape: [1, C]

    return modelFeatures

model_transform_params  = {
    "x3d_s": {
        "side_size": 182,
        "crop_size": 182,
        "num_frames": 13,
        "sampling_rate": 6,
    },
    "x3d_m": {
        "side_size": 256,
        "crop_size": 256,
        "num_frames": 16,
        "sampling_rate": 5,
    },
    "x3d_l": {
        "side_size": 320,
        "crop_size": 320,
        "num_frames": 16,
        "sampling_rate": 5,
    }
}

# ======================== Main configurations ======================== #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
selectStudyMode = 1
nameModel =  ["x3d_l","x3d_m","x3d_s"]
model_name = nameModel[selectStudyMode]
modeFullClass = True  # True 13 Classes, False 09 Classes
modeCapureVideo = False # True --> Stream, False --> Offline
modeFe = True  # True Camera + FullModels, False Camera
crop_size = model_transform_params[model_name]["crop_size"]
num_frames = model_transform_params[model_name]["num_frames"]
sampling_rate = model_transform_params[model_name]["sampling_rate"]
frameC = 3
save_local = True
videoName_Path = "./John.mp4"
mainSaveFrames_Path = "./FrameStream"
mainSaveAbnormalFrames_Path = "./framesAbnromal"
os.makedirs(mainSaveFrames_Path, exist_ok=True)
os.makedirs(mainSaveAbnormalFrames_Path, exist_ok=True)
FeModel = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)

height, width, layers = 240, 320, 3
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
# ======================== Main configurations ======================== #





#////////////////// FE Model ////////////////// # 
FeModel = FeModel.eval()
FeModel = FeModel.to(device)
del FeModel.blocks[-1]
#////////////////// FE Model ////////////////// # 


#////////////////// Detector Model ////////////////// # 
weightsDetectorName = [
    "13_0.7962_x3d_l_maxScore_MIL_BERT_MB.pkl",
    "13_0.8352_x3d_m_maxScore_MIL_BERT_MB.pkl",
    "13_0.7959_x3d_s_maxScore_MIL_BERT_MB.pkl"
]
weightsDetectorName = weightsDetectorName[selectStudyMode]
modelWigth_path_Detection = f"./weightsEl_Salvador/{model_name}/{weightsDetectorName}"
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]

if model_name == "x3d_l":
    DetectorModel = Detector_VAD(192)
else:
    DetectorModel = Model_V3_Connection(192)
state_dict = torch.load(
    modelWigth_path_Detection,
    map_location=torch.device("cpu")
)
state_dict = {
    k.replace("module.", ""): v
    for k, v in state_dict.items()
}
DetectorModel.load_state_dict(state_dict)
DetectorModel = DetectorModel.to(device).eval()
#////////////////// Detector Model ////////////////// # 





#////////////////// Classification Model ////////////////// # 
if modeFullClass:
    weightsClassName = [
        "13_0.3061_x3d_l_0.1_winSliding_MIL_MODI_M.pkl",
        "13_0.3159_x3d_m_0.1_winSliding_MIL_MODI_M.pkl",
        "13_0.2953_x3d_l_0.1_winSliding_MIL_MODI_M.pkl"
    ]
    nClasses = 13
else:
    weightsClassName = [
        "09_0.3372_x3d_l_0.1_winSliding_MIL_MODI_M.pkl",
        "09_0.3102_x3d_m_0.1_winSliding_MIL_MODI_M.pkl",
        "09_0.3034_x3d_s_0.1_winSliding_MIL_MODI_M.pkl"
    ]
    nClasses = 9

weightsClassName = weightsClassName[selectStudyMode]
modelWigth_path_Classification = f"./weightsEl_Salvador/{model_name}/{weightsClassName}"

classModel = violenceOneCrop(192, nClasses)
state_dict = torch.load(
    modelWigth_path_Classification,
    map_location=torch.device("cpu")
)
state_dict = {
    k.replace("module.", ""): v
    for k, v in state_dict.items()
}

classModel.load_state_dict(state_dict)
classModel = classModel.to(device).eval()


abnormalClasses = ['Abuse',         
                  'Arrest', 
                  'Arson', 
                  'Assault', 
                  'Burglary', 
                  'Explosion', 
                  'Fighting', 
                  'RoadAccidents', 
                  'Robbery', 
                  'Shooting', 
                  'Shoplifting', 
                  'Stealing', 
                  'Vandalism',
                  'Normal']
#////////////////// Classification Model ////////////////// # 



temporalFrames = num_frames * sampling_rate
os.makedirs(mainSaveFrames_Path, exist_ok=True)





# ======== Main Code ========= #

countAbnormalClass = 0
abnormalFlag = deque(maxlen=5) # 3 para tiendas # 5 para asalto, 
def flagAdd(witness):
    abnormalFlag.append(witness)
    # print(sum(abnormalFlag))
    return  sum(abnormalFlag)

dateNow = str(datetime.now())
videosCheck_dir = os.path.join(f"./test2_{dateNow[:10]}_{dateNow[11:13]}_{dateNow[14:16]}_{dateNow[17:19]}.mp4")
video = cv2.VideoWriter(videosCheck_dir, fourcc, fps, (width, height))

countClips = 0
frame_count = 0
countCLipsOnline = 0
scoresThr = 0.004146758
timeFull = []
frames = []

cap = cv2.VideoCapture(0)
start_time = time.time()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count == temporalFrames:
        path_frames = sorted(os.listdir(mainSaveFrames_Path))
        pathFull_frames = [os.path.join(mainSaveFrames_Path, frame) for frame in path_frames]

        if modeFe:
            featuresX3D  = featuresExtractor_OneVideo(FeModel, mainSaveFrames_Path, sampling_rate, num_frames, crop_size)
            with torch.no_grad():
                scores = DetectorModel(featuresX3D)
                if scores >= scoresThr:
                    classPred = classModel(featuresX3D)
                    pred = list(classPred.cpu().detach().numpy())
                    classAbnormal = np.argmax(pred, axis=-1).item()
                    warningAlert = flagAdd(1)
                    if countAbnormalClass > 1:
                        x1_magMotion, y1_magMotion, x2_magMotion, y2_magMotion = magMotionV2(pathFull_frames[:-16])
                        dateNow = str(datetime.now())
                        dateNow = f"{dateNow[:-16]}_{dateNow[-15:-7]}"
                        # print(f"{featuresClips.shape}, {predScore}, Clips range: {startFrame} = {finishFrame}, {abnormalClasses[index.item()]}, {abnormalClass}, {bits}, {dateNow}" )
                        # print(f"{warningAlert}, {abnormalClasses[index.item()]}, {predScore}, {dateNow},  {bits}")
                        paths = pathFull_frames[:-16]
                        for indexFrameAbnormal, path in enumerate(paths):
                            # Leer la imagen
                            image = cv2.imread(path)
                            if image is not None:
                                # Dibujar el bounding box
                                cv2.rectangle(image, (x1_magMotion, y1_magMotion), (x2_magMotion, y2_magMotion), (0, 0, 255), 2)
                                frameFinal = cv2.putText(image, abnormalClasses[classAbnormal], (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                cv2.imwrite(f"{mainSaveAbnormalFrames_Path}/bus_1_{dateNow}_{abnormalClasses[classAbnormal]}_{indexFrameAbnormal}.jpg", image)
                                video.write(image)

                    countAbnormalClass += 1
                else:
                    warningAlert = flagAdd(0)
                    if warningAlert == 0:
                        countAbnormalClass = 0
                    else:
                        countAbnormalClass += 1
                    classAbnormal = 13



            finish_time = time.time()
            timeFull.append(finish_time - start_time)
            print(f"Clip: {countClips}, label: {abnormalClasses[classAbnormal]}, Score: {scores}, Frames: {temporalFrames}, sizeFeatures: {featuresX3D.shape}, InferenceTime: {finish_time - start_time}")
        else:
            dateNow = str(datetime.now())
            dateNow = f"{dateNow[:-16]}_{dateNow[-15:-7]}"
            paths = pathFull_frames[:-16]
            for indexFrameAbnormal, path in enumerate(paths):
                # Leer la imagen
                image = cv2.imread(path)
                if image is not None:
                    # Dibujar el bounding box
                    cv2.rectangle(image, (x1_magMotion, y1_magMotion), (x2_magMotion, y2_magMotion), (0, 0, 255), 2)
                    frameFinal = cv2.putText(image, "Normal", (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.imwrite(f"{mainSaveAbnormalFrames_Path}/bus_1_{dateNow}_{abnormalClasses[-1]}_{indexFrameAbnormal}.jpg", image)
                    video.write(image)
            finish_time = time.time()
            timeFull.append(finish_time - start_time)
            print(f"Clip: {countClips}, Frames: {temporalFrames}, InferenceTime: {finish_time - start_time}")
        countClips += 1
        frame_count = 0
        start_time = time.time()


    frame = cv2.resize(frame, (width, height))
    if save_local:
        frame_path = os.path.join(mainSaveFrames_Path, f"frame_{frame_count:02d}.png")
        cv2.imwrite(frame_path, frame)
    frame_count += 1
    # if countClips == 2:
    #     break

meanTimeInference = sum(timeFull)/len(timeFull)
print(f"Frames per Clips: {sampling_rate * num_frames}, FPS: {(sampling_rate * num_frames) / meanTimeInference}, Inference time average: {meanTimeInference}")
print(meanTimeInference)
cap.release()
