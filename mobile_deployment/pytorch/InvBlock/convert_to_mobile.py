import torch
import models.imagenet as customized_models
import torchvision.models as models
from torchvision.models import mobilenet_v2

default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

model = models.__dict__['mobilenetv2'](width_mult=1)
# model = mobilenet_v2(pretrained=True)
print(torch.__version__)
model.eval()
input_tensor = torch.rand(1,3,224,224)

script_model = torch.jit.trace(model,input_tensor)
model_name = 'mbv2_no_id.pt'
print('saving model: ', model_name)
script_model.save(model_name)
