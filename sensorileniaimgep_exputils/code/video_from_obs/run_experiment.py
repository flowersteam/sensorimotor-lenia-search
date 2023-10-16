from experiment_config import *
import torch
import numpy as np
import os
os.environ['FFMPEG_BINARY'] = 'ffmpeg'
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from IPython.display import HTML, display, clear_output

class VideoWriter:
  def __init__(self, filename, fps=30.0, **kw):
    self.writer = None
    self.params = dict(filename=filename, fps=fps, **kw)

  def add(self, img):
    img = np.asarray(img)
    if self.writer is None:
      h, w = img.shape[:2]
      self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
    if img.dtype in [np.float32, np.float64]:
      img = np.uint8(img.clip(0, 1)*255)
    if len(img.shape) == 2:
      img = np.repeat(img[..., None], 3, -1)
    self.writer.write_frame(img)

  def close(self):
    if self.writer:
      self.writer.close()

  def __enter__(self):
    return self

  def __exit__(self, *kw):
    self.close()

  def show(self, **kw):
      self.close()
      fn = self.params['filename']
      display(mvp.ipython_display(fn, **kw))
        
        
torch.manual_seed(seed)
np.random.seed(seed)



pf=os.environ["ALL_CCFRSCRATCH"]+"/sensorimotor_lenia/resources/"+type_expe+"_exploration/"


path=pf+"parameters"

for file in os.listdir(path):
    if os.path.isfile(os.path.join(path,file)) and file.startswith('seed'+str(seed)):
        

        observations=torch.load(pf+"observations/observations_"+file,map_location='cpu')
        
        with VideoWriter(pf+"videos/"+file[:15]+".mp4", 120.0) as vid:
                                  for timestep in range(observations.shape[0]):
                                     rgb_im=np.concatenate([observations[timestep,:,:,0].detach().cpu().unsqueeze(-1).numpy().repeat(2,2),observations[timestep,:,:,1].detach().cpu().unsqueeze(-1).numpy()],axis=2)

                                     vid.add(rgb_im)


        

print("finished")

