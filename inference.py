import json
import torch
from PIL import Image
from torchvision import transforms as T
from net import get_model
from time import *
#from convert import pytorch_to_caffe


######################################################################
# Settings
# ---------
dataset_dict = {
    'market'  :  'Market-1501',
    'duke'  :  'DukeMTMC-reID',
    'attri_bot_cox1' : 'attri_bot_cox1'
}
num_cls_dict = { 'market':30, 'duke':23,'attri_bot_cox1':26 }
num_ids_dict = { 'market':751, 'duke':702,'attri_bot_cox1':10000 }

transforms = T.Compose([
    T.Resize(size=(288, 144)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


######################################################################
# Argument
# ---------
# model _ id inference

model_name = '{}_nfc'.format('resnet50')
num_label, num_id = 7, 1000
classes = ['baby','child','Teenager','Youth','Middle-age','old','gender']

######################################################################
# Model and Data
# ---------
def load_network(network):
    save_path = 'net_last.pth'
    network.load_state_dict(torch.load(save_path),False)
    print('Resume model from {}'.format(save_path))
    return network

def load_image(path):
    src = Image.open(path)
    src = transforms(src)
    src = src.cuda()
    src = src.unsqueeze(dim=0)
    return src


model = get_model(model_name, num_label, use_id = False, num_id=num_id)
model = load_network(model)
model = model.cuda()
model.eval()
print(model)
src = load_image('test.jpg')
######################################################################
# Inference
# ---------
class predict_decoder(object):

    def __init__(self, dataset):
        self.label_list = classes
        with open('./doc/attribute.json', 'r') as f:
            self.attribute_dict = json.load(f)[dataset]
        self.dataset = dataset
        self.num_label = len(self.label_list)

    def decode(self, pred):
        pred = pred.squeeze(dim=0)
        for idx in range(self.num_label):
            name, chooce = self.attribute_dict[self.label_list[idx]]
            #print('chooce:idx--',chooce,idx)
            if chooce[pred[idx]]:
                print('{}: {}'.format(name, chooce[pred[idx]]))


#Dec = predict_decoder(args.dataset)
#Dec.decode(pred)
name = 'age_gender_6'
out = model.forward(src)
print('**** out ',out)
print('###',type(src))
print('src____size',src.size())

#pytorch_to_caffe.trans_net(model, src, name)

#pytorch_to_caffe.save_prototxt('net_last.prototxt')
#pytorch_to_caffe.save_caffemodel('net_last.caffemodel')
print("Done!")