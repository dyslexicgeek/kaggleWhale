%reload_ext autoreload
%autoreload 2
%matplotlib inline

import torch
from fastai.imports import *

from fastai.conv_learner import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

sz=224
arch=resnet34
bs=24

PATH = 'data/whale/'
label_csv = f'{PATH}train.csv'

print (label_csv)

ls {PATH}

n = len(list(open(label_csv)))-1
val_idxs = get_cv_idxs(n)
print(val_idxs)

label_df = pd.read_csv(label_csv)

mytfm = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1);
data = ImageClassifierData.from_csv(PATH,'train',f'{PATH}/train.csv', bs, mytfm, test_name='test', val_idxs=val_idxs)

filename = PATH + data.trn_ds.fnames[1];
img = PIL.Image.open(filename); img

size_dict = {k: PIL.Image.open(PATH+k).size for k in data.trn_ds.fnames}

row_sz, col_sz = list(zip(*size_dict.values()))

row_sz = np.array(row_sz)
col_sz = np.array(col_sz)

plt.hist(row_sz)

plt.hist(col_sz)

data.classes.count
shutil.rmtree(f'{PATH}tmp', ignore_errors=True)

len(data.val_ds.fnames), len(data.val_ds.y)

lrn = ConvLearner.pretrained(arch, data, precompute=True, ps=0.5)

lrn.fit(1e-2,10)

log_preds, y = lrn.TTA()

probs = np.exp(log_preds)
metrics.log_loss(y, probs)
accuracy(log_preds,y), 
metrics.log_loss(y, probs)
