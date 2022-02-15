from os import listdir, remove, getpid, environ
from gc import collect
from psutil import Process
from flask import Flask, render_template, request
from PIL import Image, ImageDraw, ImageFont
from numpy import array
from cv2 import cvtColor, imwrite, resize, COLOR_BGR2GRAY, hconcat, convertScaleAbs

pid= getpid()
ps=Process(pid)
memory_use=ps.memory_info()
print(memory_use.rss*0.000001, memory_use.vms*0.000001)

from instancenormalization import InstanceNormalization
from tensorflow.keras.models import load_model

environ['CUDA_VISIBLE_DEVICES'] = '-1'

memory_use=ps.memory_info()
print(memory_use.rss*0.000001, memory_use.vms*0.000001)

g_model = load_model('model.h5',custom_objects={'InstanceNormalization':InstanceNormalization},compile=False)

app = Flask(__name__)

count=0
@app.route('/')
def hello_world():
    files = listdir('static')
    for i in files:
        if(i != 'style.css'):
            remove(f'static/{i}')
    print(files)
    return render_template("index.html") 

def text2str(text):
    img = Image.new('RGB', (900, 800),color = (255, 255, 255))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("font/arial.ttf", 50)
    d.text((0,0), text, font=font,fill=(0,0,0))
    text_width, text_height = d.textsize(text,font=font)
    open_cv_image = array(img)
    image = open_cv_image[:, :, ::-1].copy()[0:text_height+3,0:text_width]
    del d,font,text_height,text_width,open_cv_image
    collect()
    return image

def ValuePredictor(text):
    print('vp','-'*100)
    memory_use=ps.memory_info()
    print(memory_use.rss*0.000001, memory_use.vms*0.000001)
    global count
    global g_model
    farray=[]
    length=len(text)
    for i in range(0,length,3):
        if(i+3>length):
            val=text[i:length]
        else:
            val=text[i:i+3]
        temp = text2str(val)
        temp = cvtColor(temp, COLOR_BGR2GRAY)

        temp = resize(temp, (256, 256))

        temp = (temp - 127.5) / 127.5
        img = temp.reshape((1,256,256,1))
        
        memory_use=ps.memory_info()
        print(memory_use.rss*0.000001, memory_use.vms*0.000001)
        
        Ximg = g_model.predict(img)
        
        memory_use=ps.memory_info()
        print(memory_use.rss*0.000001, memory_use.vms*0.000001)
        
        img = (Ximg+1) / 2.0

        img = img.reshape((256,256))
        farray.append(img)
        del Ximg, img, temp, memory_use
        collect()

    imx=hconcat(farray)
    fname=f'image_{count}.png'
    imwrite(f'static/{fname}',convertScaleAbs(imx, alpha=(255.0)))
    count+=1
    return fname

@app.route('/result', methods = ['POST']) 
def result(): 
    if request.method == 'POST': 
        dict_list = request.form.to_dict()
        ftext = dict_list['str']

        print('res1','-'*100)
        prediction=ValuePredictor(ftext)
        print('res2','-'*100)

        memory_use=ps.memory_info()
        print(memory_use.rss*0.000001, memory_use.vms*0.000001)

        return render_template("result.html", prediction = prediction)

if __name__ == '__main__':
    print('main','-'*100)
    app.run(debug=False)