#YOLO-server
Make Post Request
url: http://10.10.10.200:5000/predict

image: {select image}
thresh: {threshold value for the detection, if not provided then 0.25 will be used by default}

Return a json object

count: Total Prediction
head: { #detected boxes
    confidence: confidence value for the detected box
    {   #coordinates of the detected box
        xmin
        ymin
        xmax
        ymax
    }
}