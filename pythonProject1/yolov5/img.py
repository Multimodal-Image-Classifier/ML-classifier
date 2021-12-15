import torch

model = torch.hub.load('ultralytics/yolov5', 'custom',   )


results = model(r'D:\pythonProject1\custom_data\test\images\IMG_3504.JPG')


results.show()