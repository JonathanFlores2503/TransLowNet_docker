import numpy as np
import torch
from resourcesFE.UniFormer.uniformer import uniformer_small
import os
import pickle
import cv2
from PIL import Image
import torchvision.transforms as T
from transforms import GroupNormalize, GroupScale, Stack, ToTorchFormatTensor
import time
from anomalyDetector.model import Model_V2, violenceOneCrop
import torch.nn.functional as F
import math
import re
import Jetson.GPIO as GPIO
from scipy.ndimage import label
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque


pins = [40, 38, 36]
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(True)

for pin in pins:
    GPIO.setup(pin, GPIO.OUT)


def UniFormerFE(lengthXsnippet = 16, weightsVersion = "uniformer_small_k600_16x8.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = f"/home/jetson/Documents/Jonathan/TransLowNet_V2/resourcesFE/UniFormer/{weightsVersion}"
    model = uniformer_small()
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model

def modelDetection(feature_size, modelWigth_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model_V2(feature_size)
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(modelWigth_path, weights_only = True ).items()})
    model = model.to(device).eval()
    return model

def modelClassification(feature_size, modelWigth_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = violenceOneCrop(feature_size, 13)
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(modelWigth_path, weights_only = True ).items()})
    model = model.to(device).eval()
    return model


def central_crop(frame, crop_size=224):
    width, height = frame.size
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = (width + crop_size) / 2
    bottom = (height + crop_size) / 2
    return frame.crop((left, top, right, bottom))

def apply_10_crops_to_clip(frames):
    cropped_clips = []
    for i in range(10):
        cropped_frames = []
        for frame in frames:
            if i < 5:
                if i == 0:
                    crop = frame.crop((0, 0, 224, 224))  # Superior izquierda
                elif i == 1:
                    crop = frame.crop((frame.width - 224, 0, frame.width, 224))  # Superior derecha
                elif i == 2:
                    crop = frame.crop((0, frame.height - 224, 224, frame.height))  # Inferior izquierda
                elif i == 3:
                    crop = frame.crop((frame.width - 224, frame.height - 224, frame.width, frame.height))  # Inferior derecha
                elif i == 4:
                    crop = central_crop(frame, 224)  # Recorte central
            else:
                if i == 5:
                    crop = frame.crop((0, 0, 224, 224)).transpose(Image.FLIP_LEFT_RIGHT)  # Superior izquierda reflejada
                elif i == 6:
                    crop = frame.crop((frame.width - 224, 0, frame.width, 224)).transpose(Image.FLIP_LEFT_RIGHT)  # Superior derecha reflejada
                elif i == 7:
                    crop = frame.crop((0, frame.height - 224, 224, frame.height)).transpose(Image.FLIP_LEFT_RIGHT)  # Inferior izquierda reflejada
                elif i == 8:
                    crop = frame.crop((frame.width - 224, frame.height - 224, frame.width, frame.height)).transpose(Image.FLIP_LEFT_RIGHT)  # Inferior derecha reflejada
                elif i == 9:
                    crop = central_crop(frame, 224).transpose(Image.FLIP_LEFT_RIGHT)  # Centro reflejada
            cropped_frames.append(crop)
        cropped_clips.append(cropped_frames)
    return cropped_clips



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




# videoName_Path = "/home/jetson/Documents/Jonathan/TransLowNet_V2/Shoplifting028_x264.mp4"
# videoName_Path = "/home/jetson/Documents/Jonathan/TransLowNet_V2/Video_26.mp4"
# videoName_Path = "/home/jetson/Documents/Jonathan/TransLowNet_V2/Video_20.mp4"
# videoName_Path = "/home/jetson/Documents/Jonathan/TransLowNet_V2/Video_15.mp4"
# videoName_Path = "/home/jetson/Documents/Jonathan/TransLowNet_V2/Video_33.mp4"
# videoName_Path = "/home/jetson/Documents/Jonathan/TransLowNet_V2/gibran1.mp4"

videoName_Path = "John.mp4"


mainSaveFrames_Path = "/home/jetson/Documents/Jonathan/TransLowNet_V2/FrameStream"
mainSaveAbnormalFrames_Path = "/home/jetson/Documents/Jonathan/TransLowNet_V2/framesAbnromal"

os.makedirs(mainSaveFrames_Path, exist_ok=True)
os.makedirs(mainSaveAbnormalFrames_Path, exist_ok=True)

temporalFrames = 16
crop_size = 224
scale_size = 256
input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]
original_fps = 30
porposed_fps = 30
final_fps = original_fps // porposed_fps
save_local = True
feature_size = 512

transform = T.Compose([
    GroupScale(int(scale_size)),
    Stack(),
    ToTorchFormatTensor(),
    GroupNormalize(input_mean, input_std)
])

abnormalClasses = [ "Abuso",          #00   O Abuse
                    "Robo",        #01   X Arrest
                    "Arson",         #02   X
                    "Assault",       #03   O
                    "Robo",        #04   O Burglary # pelea para 26
                    "Asalto",     #05   X
                    "Pelea",      #06   O
                    "RoadAccidents", #07   X
                    "Asalto",       #08   O
                    "Shooting",      #09   O
                    "Robo",   #10    Shoplifting
                    "Robo",        #11 Stealing
                    "Vandalism"]     #12
                    #Normal          #13 000


# abnormalClasses = ['Abuse',         
#                   'Arrest', 
#                   'Arson', 
#                   'Assault', 
#                   'Burglary', 
#                   'Explosion', 
#                   'Fighting', 
#                   'RoadAccidents', 
#                   'Robbery', 
#                   'Shooting', 
#                   'Shoplifting', 
#                   'Stealing', 
#                   'Vandalism']


secuencias = [
    [0, 0, 0], # Normal
    [0, 0, 1], # Abuse
    [0, 1, 0], # Assault
    [0, 1, 1], # Burglary
    [1, 0, 0], # Fighting
    [1, 0, 1], # Shooting
    [1, 1, 0], # Vandalism
    [1, 1, 1], # Los demas
]



frame_count = 0
frames = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

featureExtractorModel = UniFormerFE(lengthXsnippet=temporalFrames).to(device)

modelWigth_path = "/home/jetson/Documents/Jonathan/TransLowNet_V2/anomalyDetector/Uniformer-S/UniFormer_S_0.8227_10C.pkl"
detectionModel = modelDetection(feature_size, modelWigth_path)


modelClass_path ="/home/jetson/Documents/Jonathan/TransLowNet_V2/anomalyClassifier/Uniformer-S/0.3642_10_13Classes_ClassV3.pkl"
classModel = modelClassification(feature_size, modelClass_path)
finishFrame = 0


height, width, layers = 240, 320, 3
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

abnormalFlag = deque(maxlen=5) # 3 para tiendas # 5 para asalto, 

def flagAdd(witness):
    abnormalFlag.append(witness)
    # print(sum(abnormalFlag))
    return  sum(abnormalFlag)



# cap = cv2.VideoCapture(videoName_Path)
cap = cv2.VideoCapture(0)




countAbnormalClass = 0
bits = secuencias[0]
for pin, bit in zip(pins, bits):
    GPIO.output(pin, bit)

time.sleep(60)


dateNow = str(datetime.now())

videosCheck_dir = os.path.join(f"/home/jetson/Documents/Jonathan/TransLowNet_V2/test2_{dateNow}.mp4")
video = cv2.VideoWriter(videosCheck_dir, fourcc, fps, (width, height))

countCLipsOnline = 0
start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break


    if frame_count == temporalFrames:

        path_frames = sorted(os.listdir(mainSaveFrames_Path))
        pathFull_frames = [os.path.join(mainSaveFrames_Path, frame) for frame in path_frames]

        vr = []
        for videoFrame in range(len(pathFull_frames)):
            pathFull_frames.append(pathFull_frames[videoFrame])
            with Image.open(pathFull_frames[videoFrame]) as img:
                vr.append(img.copy())

        cropped_clips = apply_10_crops_to_clip(vr)


        featuresClips = []
        with torch.no_grad():
            for cropIndex in range(1):
                transformed_segment = transform(cropped_clips[cropIndex])
                clipTensor = transformed_segment.view(1, transformed_segment.size(0)//3, 3, transformed_segment.size(1), transformed_segment.size(2)).permute(0, 2, 1, 3, 4)
                featureVector = featureExtractorModel(clipTensor.to(device))
                torch.cuda.empty_cache()

                featureVector = featureVector
                featuresClips.append(featureVector)
            featuresClips = torch.stack(featuresClips, dim=1)
        # 1,10,512
        with torch.no_grad():
            predScore = detectionModel(featuresClips) # 0-1
            torch.cuda.empty_cache()

        startFrame = finishFrame
        finishFrame = startFrame + 15


        if predScore > 0.50: # 0.25 tienda, 15 a 0.05
            with torch.no_grad():
                classPred = classModel(featuresClips) # 1,13
                torch.cuda.empty_cache()
            _, index = torch.max(classPred, dim=1)
            abnormalClass = index.item()

            warningAlert = flagAdd(1)
           
            if countAbnormalClass > 1:
                
                x1_magMotion, y1_magMotion, x2_magMotion, y2_magMotion = magMotionV2(pathFull_frames[:-16])

                if abnormalClass == 0:
                    bits = secuencias[1]
                elif abnormalClass == 3:
                    bits = secuencias[2]
                elif abnormalClass == 4 or abnormalClass == 8 or abnormalClass == 10 or abnormalClass == 11:
                    bits = secuencias[3]
                elif abnormalClass == 6:
                    bits = secuencias[4]
                elif abnormalClass == 9:
                    bits = secuencias[5]
                elif abnormalClass == 12:
                    bits = secuencias[6]
                else:
                    bits = secuencias[7]
            
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
                        frameFinal = cv2.putText(image, abnormalClasses[index.item()], (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.imwrite(f"{mainSaveAbnormalFrames_Path}/bus_1_{dateNow}_{abnormalClasses[index.item()]}_{indexFrameAbnormal}.jpg", image)
                        # video.write(image)
            # else:
            #     dateNow = str(datetime.now())
            #     dateNow = f"{dateNow[:-16]}_{dateNow[-15:-7]}"
            #     paths = pathFull_frames[:-16]
            #     for indexFrameAbnormal, path in enumerate(paths):
            #         # Leer la imagen
            #         image = cv2.imread(path)
            #         if image is not None:
                        # Dibujar el bounding box
                        # cv2.rectangle(image, (x1_magMotion, y1_magMotion), (x2_magMotion, y2_magMotion), (0, 0, 255), 2)
                        # frameFinal = cv2.putText(image, "Normal", (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        # cv2.imwrite(f"{mainSaveAbnormalFrames_Path}/bus_1_{dateNow}_{abnormalClasses[index.item()]}_{indexFrameAbnormal}.jpg", image)
                        # video.write(image)

            
            countAbnormalClass += 1
         
                

        else:
            bits = secuencias[0]
            
            warningAlert = flagAdd(0)
            if warningAlert == 0:
                countAbnormalClass = 0
            else:
                countAbnormalClass += 1
            # print(f"{warningAlert}, Normal, {predScore},  {bits}")
            
            

            paths = pathFull_frames[:-16]
            # for indexFrame, path in enumerate(paths):
            #     # Leer la imagen
            #     image = cv2.imread(path)
            #     if image is not None:
            #         framesSave = str(indexFrame).zfill(2)
            #         clipsIndex = str(countCLipsOnline).zfill(4)
            #         cv2.imwrite(f"/home/jetson/Documents/Jonathan/TransLowNet_V2/test/bus_1_{countCLipsOnline}_{framesSave}.jpg", image)
                    # image = cv2.putText(image, "Normal", (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                #     video.write(image)
                # else:
                #     print(f"Error al cargar la imagen: {path}")

        # finish_time = time.time()
        # tineFinish = finish_time - start_time
        # print(f"Clip: {countCLipsOnline}, TiempoInference: {tineFinish}")
        # start_time = time.time()  

        for pin, bit in zip(pins, bits):
            GPIO.output(pin, bit)
        
        frame_count = 0
        countCLipsOnline += 1
        # print(finish_time - start_time)
        # start_time = time.time()


    frame = cv2.resize(frame, (width, height))
    if save_local:
        frame_path = os.path.join(mainSaveFrames_Path, f"frame_{frame_count:02d}.png")
        cv2.imwrite(frame_path, frame)
    frame_count += 1

    # if countCLipsOnline == 200:
    #     break
cap.release()

# while(1):
#     bits = secuencias[-1]
#     for pin, bit in zip(pins, bits):
#         GPIO.output(pin, bit)