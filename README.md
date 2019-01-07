created by Uday Soni on 05/01/2019

This pythonn script uses some libararis.
1. OpenCv
2. numpy

install them before running the script.
other import libaries used are by default in python. So , dont worry about them.

download the yolov3.weights here.

https://pjreddie.com/media/files/yolov3.weights`

You can run the script by this command.

> python yolo_python.py  --config yolov3.cfg  --weights yolov3.weights  --classes yolov3.txt

This code is wriiten to take realtime camera feed and detect objects.

You can edit this file to run on image(saved on computer, not webcam feed).

I have commented some part of the code and added some comments to guide you for chaging code if you want.

In this case your command will be-

> python yolo_python.py --img image_path  --config yolov3.cfg  --weights yolov3.weights  --classes yolov3.txt
   
