import torch
import torch.nn.functional as F

def spectral_loss(output, output2, target, tendency, wavenum_init, lamda_reg, time_step):

   loss1 = torch.mean((output-target)**2)
   
   # out_fft = torch.fft.rfft(output,dim=1)
   # target_fft = torch.fft.rfft(target,dim=1)
   
   # loss2 = torch.mean(torch.abs(out_fft[:,wavenum_init:] - target_fft[:,wavenum_init:]))


   out_du_fft =torch.fft.rfft((output-output2)/time_step,dim=1)
   target_du_fft =torch.fft.rfft(tendency/time_step,dim=1)

   loss3 = torch.mean(torch.abs(out_du_fft[:,0:]-target_du_fft[:,0:]))

   loss = loss1 + lamda_reg*loss3
   # loss = loss1*(1+lamda_reg*loss3)
  
   return loss