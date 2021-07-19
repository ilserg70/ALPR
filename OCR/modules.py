import torch.optim as optim
import numpy as np
import cv2
import os

def optimizer_build(model,params):
    if params.adam:
        optimizer = optim.Adam(model.parameters(), lr=params.lr, betas=(params.beta1, 0.999),
                               weight_decay=params.weight_decay)
    elif params.adadelta:
        optimizer = optim.Adadelta(model.parameters(), weight_decay=params.weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    return optimizer

def error_save(errors,target,pred,image,hash_id,path):
    errors.append((target, pred))
    img = image.squeeze(1).permute(1,2,0).cpu().numpy()
    img = np.array(img*255+128)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    name = os.path.join(path,'incorrect_'+target+'_'+pred +'_'+hash_id+'.jpg')

    cv2.imwrite(name,img)