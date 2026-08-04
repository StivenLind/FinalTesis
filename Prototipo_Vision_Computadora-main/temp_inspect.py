import cv2
import os
import mediapipe as mp
from mediapipe.tasks.python import vision
print('mp version', mp.__version__)
print('vision members', [x for x in dir(vision) if 'Landmark' in x or 'pose' in x.lower()])
print('Image class', mp.Image)
print('ImageFormat', mp.ImageFormat)
print('RunningMode', mp.tasks.vision.RunningMode)
options = vision.PoseLandmarkerOptions(base_options=mp.tasks.python.BaseOptions(model_asset_path='pose_landmarker_lite.task'), running_mode=mp.tasks.vision.RunningMode.IMAGE)
pose = vision.PoseLandmarker.create_from_options(options)
print('pose type', type(pose))
path = os.path.join('Entrenamiento Data Set','img1.jpg')
print('image path', path)
img = cv2.imread(path)
print('img loaded', img is not None)
if img is None:
    raise SystemExit('image missing')
mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
results = pose.detect(mpimg)
print('result type', type(results))
print('result attrs', [a for a in dir(results) if not a.startswith('_')])
print('pose_landmarks attr?', hasattr(results, 'pose_landmarks'))
pl = getattr(results, 'pose_landmarks', None)
print('pose_landmarks repr', pl)
print('pose_landmarks type', type(pl))
print('pose_landmarks len', len(pl) if pl else 0)
if pl:
    first = pl[0]
    print('first type', type(first))
    print('first dir sample', [a for a in dir(first) if not a.startswith('_')][:60])
    if hasattr(first, 'landmark'):
        print('first.landmark len', len(first.landmark))
        print('landmark 0 attrs', [a for a in dir(first.landmark[0]) if not a.startswith('_')][:60])
        print('first landmark x,y', first.landmark[0].x, first.landmark[0].y)
    else:
        print('first has no .landmark')
