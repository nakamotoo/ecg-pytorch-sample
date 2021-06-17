import torch
import numpy as np
import os
import codecs
import csv

class ECG_2D(torch.utils.data.Dataset):
    def __init__(self, root = None, split = "train", mean = 0., std = 1., filelist_name = "FileList.csv", data_folder="./data"):
        if root is None:
            root = os.getcwd()
            
        self.filelist_name = filelist_name
        self.split = split
        self.mean = mean
        self.std = std
        
        self.folder = os.path.join(root, data_folder)
        self.fnames = []
        self.targets = []
        self.traces = []
        self.trace_mode = False # トレースによるオーグメンテーションを行うか
        
        with open(os.path.join(root, self.folder, filelist_name)) as f:
            self.header = f.readline().strip().split(",")
            filenameIndex = self.header.index("filename")
            splitIndex = self.header.index("split")
            targetIndex = self.header.index("target")
            
            if "start_point" in self.header and "end_point" in self.header:
                startIndex = self.header.index("start_point")
                endIndex = self.header.index("end_point")
            
            for line in f:
                lineSplit = line.strip().split(',')
                fileName = lineSplit[filenameIndex]
                fileMode = lineSplit[splitIndex].lower()
                target = int(lineSplit[targetIndex])
                if "start_point" in self.header and "end_point" in self.header:
                    start_point = int(lineSplit[startIndex])
                    end_point = int(lineSplit[endIndex])
                
                if split in ["all", fileMode]:
                    if (fileMode == "test") and start_point != 0:
                        continue
                    self.fnames.append(fileName)
                    self.targets.append(target)
                    if "start_point" in self.header and "end_point" in self.header:
                        self.traces.append((start_point, end_point))
                    

    def __getitem__(self, index):
        filename = self.fnames[index]
        target = self.targets[index]
        
        if "start_point" in self.header and "end_point" in self.header:
            start_point, end_point = self.traces[index]
        
        filepath_ecg = os.path.join(self.folder,"ecg" ,filename)
        with codecs.open(filepath_ecg, 'r', 'utf-8', 'ignore') as f:
            reader = csv.reader(f)
            data_list =  list(reader)
            if "start_point" in self.header and "end_point" in self.header:
                data = np.array(data_list[1:])[start_point:end_point, 0:12].transpose(1, 0).reshape(1, 12, -1).astype(np.float32)
            else:
                data = np.array(data_list[1:])[:, 0:12].transpose(1, 0).reshape(1, 12, -1).astype(np.float32)
#             data= data[:, [1,2,5,0,4,11,3,6,7,8,9,10], :]

                
        
        # 標準化
        if isinstance(self.mean, (float, int)):
            data -= self.mean
        else:
            data -= self.mean.reshape(1, 12, 1)
        
        
        if isinstance(self.std, (float, int)):
            data /= self.std
        else:
            data /= self.std.reshape(1, 12, 1)
            
        return data, target
    
    def __len__(self):
        return len(self.fnames)