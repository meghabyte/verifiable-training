import torch
import numpy as np
import os



class Tensor012Logger:
  def __init__(self, log_filename, mode='wb', threshold=0, device='cuda'):
    assert mode=='rb' or mode=='wb', "mode must be 'rb' or 'wb'"
    if mode=='rb':
      assert os.path.exists(log_filename), f"File '{log_filename}' not found."
    try:
      self.f = open(log_filename, mode)
    except:
      print(f"Error: open('{log_filename}', '{mode}').")
    self.mode = mode
    self.threshold = threshold
    self.device = device
    self.tensor_list = []
    self.tensor_len_list = []
    self.numel = 0
    self.isopen = True
    self.metadata = []
    self.metadata_filename = log_filename+'_metadata'
    if mode=='rb':
      try:
        with open(self.metadata_filename) as fmd:
          self.metadata = [tuple(int(x) for x in line.split(',')) for line in fmd]
      except:
        print(f"Error: open('{self.metadata_filename}', 'r').")
        self.metadata = None
    self.tensors_read = 0
    self.fn = log_filename

  # destroy the object
  def __del__(self):
    if self.mode == 'wb':
      self.flush_tensors()
      try:
        fmd = open(self.metadata_filename, 'w')
        for e in self.metadata:
          fmd.write(f"{e[0]}, {e[1]}\n")
        fmd.close()
      except:
        print(f"Error: open('{self.metadata_filename}', 'w').")
    self.f.close()

  # add a new tensor
  def put_tensor(self, t):
    assert self.mode == 'wb', "mode must be 'wb'."
    assert isinstance(t, torch.Tensor), "t must be a torch.Tensor."
    self.tensor_len_list.append(torch.numel(t))
    if (n:=torch.numel(t)%5) != 0:
      t = torch.cat([t,t.new_zeros(5-n)])
    weights = t.new([1, 3, 9, 27, 81])
    groups = t.view(-1, 5)
    t_compressed = (groups * weights).sum(dim=1)
    self.tensor_list.append(t_compressed)
    self.metadata.append((self.tensor_len_list[-1],torch.numel(t_compressed)))
    self.numel += torch.numel(t_compressed)
    if self.threshold >= 0 and self.numel > self.threshold:
      for t in self.tensor_list:
        self.f.write(bytearray(t.tolist()))
      self.numel = 0
      self.tensor_list = []

  # get a new tensor
  def get_tensor(self, tsize):
    assert self.mode == 'rb', "mode must be 'rb'."
    if self.metadata != None:
      if tsize != self.metadata[self.tensors_read][0]:
        print(f"attempting to read tensor of size {tsize} when {self.metadata[self.tensors_read][0]} was logged.")
      self.tensors_read += 1
    if (n:=tsize%5) != 0:
      tsize_incr = 5 - n
      tsize_padded = tsize + tsize_incr
    else:
      tsize_padded = tsize
    num_bytes = tsize_padded // 5
    byte_array = self.f.read(num_bytes)

    tensor_elements = []
    for byte in byte_array:
      base3_digits = []
      for i in range(5):
        remainder = byte % 3
        base3_digits.append(remainder)
        byte //= 3
      tensor_elements.extend(base3_digits[:])
    t = torch.tensor(tensor_elements, device=self.device, dtype=torch.uint8)
    return t[:tsize]

  def verify_tensor_final(self, t):
    assert isinstance(t, torch.Tensor), "t must be a torch.Tensor."
    assert self.mode == 'rb', "mode must be 'rb'."
    tsize = torch.numel(t)
    if (n:=tsize%5) != 0:
      n_pad = 5 - n
      tsize_padded = tsize + n_pad
    else:
      n_pad = 0
      tsize_padded = tsize
    num_bytes_when_compressed = tsize_padded // 5
    byte_array_logged = self.f.read(num_bytes_when_compressed)
    t_compressed_logged = t.new_tensor(list(byte_array_logged))
    t_logged = torch.zeros((t_compressed_logged.size(0),5),device=t.device,dtype=torch.uint8)
    for i in range(5):
      # Extract the i-th base-3 digit (starting from the least significant digit)
      t_logged[:, i] = (t_compressed_logged // (3 ** i)) % 3
    t_logged = t_logged.flatten()
    if n_pad!=0:
      t_logged = t_logged[:-n_pad]
    return torch.logical_and(t_logged!=t,t_logged==0),torch.logical_and(t_logged!=t,t_logged==2)

  def flush_tensors(self):
    assert self.mode == 'wb', "mode must be 'wb'."
    for t in self.tensor_list:
        self.f.write(bytearray(t.tolist()))
    self.numel = 0
    self.tensor_list = []

  def logfile_b2a(log_filename,out_filename=None,chunk_size=4096,line_size=60):
    assert os.path.exists(log_filename), f"File '{log_filename}' not found."
    if out_filename != None:
      fout = open(out_filename, "w")
    with open(log_filename, "rb") as fin:
      base3_digits = ''
      while True:
        byte_array = fin.read(chunk_size)
        if not byte_array:
          if len(base3_digits) > 0:
            if out_filename:
              fout.write(base3_digits)
              fout.close()
            else:
              print(base3_digits)
          break
        for byte in byte_array:
          for i in range(5):
            remainder = byte % 3
            base3_digits += str(remainder)
            byte //= 3
          if len(base3_digits) >= line_size:
            if out_filename:
              fout.write(base3_digits)
            else:
              print(base3_digits)
            char_count = 0
            base3_digits = ''