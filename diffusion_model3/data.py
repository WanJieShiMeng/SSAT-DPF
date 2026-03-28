import numpy as np
import scipy.io as sio
import h5py
import torch
import torch.utils.data

""" Training dataset"""

class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):

        self.len = Xtrain.shape[0]
        self.x_data = Xtrain.astype(np.float32)
        self.y_data = torch.LongTensor(ytrain)
        self.factor = 8

    def __getitem__(self, index):
        file_index = index // self.factor
        aug_num = int(index % self.factor)
        x_aug = self.x_data[file_index].transpose(1,2,0)
        x_aug = data_augmentation(x_aug, mode=aug_num)
        x_aug = torch.from_numpy(x_aug.copy()).permute(2, 0, 1)
        # 根据索引返回数据和对应的标签
        return x_aug, self.y_data[file_index]
    def __len__(self):

        # 返回文件数据的数目
        return self.len * self.factor

class TrainDS_orig(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):

        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        x = self.x_data[index]
        # x = x[torch.randint(0, x.shape[0], (3,)), :, :]
        return x, self.y_data[index]
    def __len__(self):

        # 返回文件数据的数目
        return self.len
""" Testing dataset"""

class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):

        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):

        # 返回文件数据的数目
        return self.len


def data_augmentation(label, mode=0):
    if mode == 0:
        # original
        return label
    elif mode == 1:
        # flip up and down
        return np.flipud(label)   # flipud  上下翻转
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(label)    # rot90  逆时针旋转90度*k, k = 负数时顺时针旋转
    elif mode == 3:
        # rotate 90 degree and flip up and down
        return np.flipud(np.rot90(label))
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(label, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        return np.flipud(np.rot90(label, k=2))
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(label, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        return np.flipud(np.rot90(label, k=3))

# PCA降维
from sklearn.decomposition import PCA
def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX

class HSIDataLoader(object):
    def __init__(self, param={}) -> None:
        self.data_param = param.get('data', {})
        self.data = None #原始读入X数据 shape=(h,w,c)
        self.labels = None #原始读入Y数据 shape=(h,w,1)

        # 参数设置
        self.data_sign = self.data_param.get('data_sign', 'Indian')
        self.patch_size = self.data_param.get('patch_size', 32) # n * n
        self.padding = self.data_param.get('padding', True) # n * n
        self.remove_zeros = self.data_param.get('remove_zeros', False)
        self.batch_size = self.data_param.get('batch_size', 50)
        self.select_spectral = self.data_param.get('select_spectral', []) # [] all spectral selected

        self.squzze = True

        self.split_row = 0
        self.split_col = 0

        self.light_split_ori_shape = None
        self.light_split_map = []



    def load_data(self):
        data, labels = None, None
        if self.data_sign == "Indian_pines":
            data = sio.loadmat('/home/wangxiangyu/Project/diffusion_super_resolution/diffusion_data/Indian_pines_corrected.mat')['indian_pines_corrected']
            labels = sio.loadmat('/home/wangxiangyu/Project/diffusion_super_resolution/diffusion_data/Indian_pines_gt.mat')['indian_pines_gt']
        elif self.data_sign == "PaviaC":
            data = sio.loadmat("/home/wxy/Pavia/Pavia.mat")['pavia']
            labels = sio.loadmat("/home/wxy/Pavia/Pavia_gt.mat")['pavia_gt']

            # data = sio.loadmat('D:\Project\Data\Pavia Centre\Pavia.mat')['pavia']
            # print(data.shape)
            # labels = sio.loadmat("D:\Project\Data\Pavia Centre\Pavia_gt.mat")['pavia_gt']
        elif self.data_sign == "houstonU":
            data = h5py.File('/home/wangxiangyu/houstonU/HoustonU.mat')['houstonU']
            data = np.transpose(data, (2, 1, 0))[:,:,:-2]
            print(data.shape)
            labels = h5py.File('/home/wangxiangyu/houstonU/HoustonU_gt.mat')['houstonU_gt']
            labels = np.transpose(labels,(1,0))
        elif self.data_sign == "Chikusei":
            data = h5py.File('/home/wangxiangyu/Chikusei/HyperspecVNIR_Chikusei_20140729.mat')['chikusei']
            data = np.transpose(data,(2,1,0))
            # data = data[106:2410,143:2191,:]
            print(data.shape)
            # print(data[1,1,:5])
            labels = sio.loadmat('/home/wangxiangyu/Chikusei/HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat')['GT'][0][0][0]
        elif self.data_sign == "houstonU2013":
            data = h5py.File('/home/wxy/houstonU2013/Houston_U.mat')['houston']
            data = np.transpose(data, (2, 1, 0))
            print(data.shape)
            labels = sio.loadmat('/home/wxy/houstonU2013/Houston_gt_U.mat')['imggt']
            print(labels.shape)
        else:
            pass
        print("ori data load shape is", data.shape, labels.shape)
        if len(self.select_spectral) > 0:  #user choose spectral himself
            data = data[:,:,self.select_spectral]
        return data, labels

    def get_ori_data(self):
        return np.transpose(self.data, (2,0,1)), self.labels

    def _padding(self, X, margin=2):
        # pading with zeros
        w,h,c = X.shape
        new_x, new_h, new_c = w+margin*2, h+margin*2, c
        returnX = np.zeros((new_x, new_h, new_c))
        start_x, start_y = margin, margin
        returnX[start_x:start_x+w, start_y:start_y+h,:] = X
        return returnX

    def get_patches_by_light_split(self, X, Y, patch_size=1):
        h, w, c = X.shape
        row = h // patch_size
        if h % patch_size != 0:
            row += 1
        col = w // patch_size
        if w % patch_size != 0:
            col += 1
        res = np.zeros((row*col, patch_size, patch_size, c))
        self.light_split_ori_shape = X.shape
        resY = np.zeros((row*col))
        index = 0
        for i in range(row):
            for j in range(col):
                start_row = i*patch_size
                if start_row + patch_size > h:
                    start_row = h - patch_size
                start_col = j*patch_size
                if start_col + patch_size > w:
                    start_col = w - patch_size

                res[index, :,:,:] = X[start_row:start_row+patch_size, start_col:start_col+patch_size, :]
                self.light_split_map.append([index, start_row, start_row+patch_size, start_col, start_col+patch_size])
                index += 1
        print(index)
        return res, resY

    def reconstruct_image_by_light_split(self, inputX, pathch_size=1):
        '''
        input shape is (batch, h, w, c)
        '''
        assert self.light_split_ori_shape is not None
        ori_h, ori_w, ori_c = self.light_split_ori_shape
        batch, h, w, c = inputX.shape
        assert batch == len(self.light_split_map) # light_split_map必须与batch值相同
        X = np.zeros((ori_h, ori_w, c))
        for tup in self.light_split_map:
            index, start_row, end_row, start_col, end_col = tup
            X[start_row:end_row, start_col:end_col, :] = inputX[index, :, :, :]
        return X


    def get_patches_by_split(self, X, Y, patch_size=1):
        h, w, c = X.shape
        row = h // patch_size
        col = w // patch_size
        newX = X
        res = np.zeros((row*col, patch_size, patch_size, c))
        resY = np.zeros((row*col))
        index = 0
        for i in range(row):
            for j in range(col):
                res[index,:,:,:] = newX[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size,:]
                index += 1
        self.split_row = row
        self.split_col = col
        print(index)
        return res, resY
    def split_to_big_image(self, splitX):
        '''
        input splitX shape (batch, 1, spe, h, w)
        return newX shape (spe, bigh, bigw)
        '''
        patch_size = self.patch_size
        batch, channel, spe, h, w = splitX.shape
        assert self.split_row * self.split_col == batch
        newX = np.zeros((spe, self.split_row * patch_size, self.split_col * patch_size))
        index = 0
        for i in range(self.split_row):
            for j in range(self.split_col):
                index = i * self.split_col + j
                newX[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = splitX[index, 0, :, :, :]
        return newX



    def re_build_split(self, X_patches, patch_size):
        '''
        X_pathes shape is (batch, channel=1, spectral, height, with)
        return shape is (height, width, spectral)
        '''
        h,w,c = self.data.shape
        row = h // patch_size
        if h % patch_size > 0:
            row += 1
        col = w // patch_size
        if  w % patch_size > 0:
            col += 1
        newX = np.zeros((c, row*patch_size, col*patch_size))
        for i in range(row):
            for j in range(col):
                newX[:,i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = X_patches[i*col+j,0,:,:,:]
        return np.transpose(newX, (1,2,0))

    def get_patches(self, X, Y, patch_size=1, remove_zero=False):
        w,h,c = X.shape
        #1. padding
        margin = (patch_size - 1) // 2
        if self.padding:
            X_padding = self._padding(X, margin=margin)
        else:
            X_padding = X

        #2. zero patchs
        temp_w, temp_h, temp_c = X_padding.shape
        row = temp_w - patch_size + 1
        col = temp_h - patch_size + 1
        X_patchs = np.zeros((row * col, patch_size, patch_size, c)) #one pixel one patch with padding
        Y_patchs = np.zeros((row * col))
        patch_index = 0
        for r in range(0, row):
            for c in range(0, col):
                temp_patch = X_padding[r:r+patch_size, c:c+patch_size, :]
                X_patchs[patch_index, :, :, :] = temp_patch
                patch_index += 1

        if remove_zero:
            X_patchs = X_patchs[Y_patchs>0,:,:,:]
            Y_patchs = Y_patchs[Y_patchs>0]
            Y_patchs -= 1
        print(patch_index)
        return X_patchs, Y_patchs #(batch, w, h, c), (batch)


    def custom_process(self, data):
        '''
        没用到
        pavia数据集 增加一个光谱维度 从103->104 其中第104维为103的镜像维度
        data shape is [h, w, spe]
        '''

        if self.data_sign == "Pavia":
            h, w, spe = data.shape
            new_data = np.zeros((h,w,spe+1))
            new_data[:,:,:spe] = data
            new_data[:,:,spe] = data[:,:,spe-1]
            return new_data
        return data


    def get_patches_by_stride_split(self,X,patch_size,stride):
        h, w, c = X.shape
        row = (h - patch_size) // stride + 1
        col = (w - patch_size) // stride + 1
        newX = X
        res = np.zeros((row * col, patch_size, patch_size, c))
        resY = np.zeros((row*col))
        index = 0
        for i in range(row):
            for j in range(col):
                res[index, :, :, :] = newX[i * stride:i * stride + patch_size, j * stride:j * stride + patch_size, :]
                index += 1
        self.split_row = row
        self.split_col = col
        print(index)
        return res,resY


    def generate_torch_dataset(self, split=False, light_split=False, stride = True):
        #1. 根据data_sign load data
        self.data, self.labels = self.load_data()

        #1.1 norm化
        # norm_data = np.zeros(self.data.shape)
        # # 原始的正则化方法
        # for i in range(self.data.shape[2]):
        #     input_max = np.max(self.data[:,:,i])
        #     input_min = np.min(self.data[:,:,i])
        #     norm_data[:,:,i] = (self.data[:,:,i]-input_min)/(input_max-input_min) * 2 - 1  # [-1,1]

        # 和后续SR正则化一样
        norm_data = self.data / np.max(self.data)

        print('[data] load data shape data=%s, label=%s' % (str(norm_data.shape), str(self.labels.shape)))
        self.data = norm_data

        #1.2 专门针对特殊的数据集补充或删减一些光谱维度
        norm_data = self.custom_process(norm_data)

        #2. 获取patchs
        if stride :
            if self.data_sign == "Chikusei":
                stride = self.patch_size // 2
            elif self.data_sign == "PaviaC":
                stride = self.patch_size // 4
            elif self.data_sign == "houstonU":
                stride = self.patch_size // 4

            X_patchs, Y_patchs = self.get_patches_by_stride_split(norm_data, patch_size=self.patch_size, stride = stride)
            print('[data stride split] data patches shape data=%s' % (str(X_patchs.shape)))
        else:
            if not split and not light_split:
                X_patchs, Y_patchs = self.get_patches(norm_data, self.labels, patch_size=self.patch_size, remove_zero=False)
                print('[data not split] data patches shape data=%s, label=%s' % (str(X_patchs.shape), str(Y_patchs.shape)))
            elif split:
                X_patchs, Y_patchs = self.get_patches_by_split(norm_data, self.labels, patch_size=self.patch_size)
                print('[data split] data patches shape data=%s, label=%s' % (str(X_patchs.shape), str(Y_patchs.shape)))
            elif light_split:
                X_patchs, Y_patchs = self.get_patches_by_light_split(norm_data, self.labels, patch_size=self.patch_size)
                print('[data light split] data patches shape data=%s, label=%s' % (str(X_patchs.shape), str(Y_patchs.shape)))


        #4. 调整shape来满足torch使用
        X_all = X_patchs.transpose((0, 3, 1, 2))
        # X_all = np.expand_dims(X_all, axis=1)
        print('------[data] after transpose train, test------')
        print("X.shape=", X_all.shape)
        print("Y.shape=", Y_patchs.shape)
        trainset = TrainDS_orig(X_all, Y_patchs)
        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                batch_size=self.batch_size,
                                                shuffle=True
                                                )
        return train_loader, X_all, Y_patchs




if __name__ == "__main__":
    # dataloader = HSIDataLoader({"data":{"padding":False, "select_spectral":[1,99,199]}})
    # train_loader = dataloader.generate_torch_dataset()
    # train_loader,X,Y = dataloader.generate_torch_dataset(split=True)
    # newX = dataloader.re_build_split(X, dataloader.patch_size)
    # print(newX.shape)

    #dataloader = HSIDataLoader(
    #    {"data":{"data_sign":"Pavia", "padding":False, "batch_size":256, "patch_size":16, "select_spectral":[]}})
    #train_loader,X,Y = dataloader.generate_torch_dataset(light_split=True)
    #print(X.shape)

    dataloader = HSIDataLoader(
        {"data":{"data_sign":"houstonU2013", "padding":False, "batch_size":10, "patch_size":32, "select_spectral":[]}})
    train_loader,X,Y = dataloader.generate_torch_dataset() # split = True: 无重复  light_split: row or col 除以 patch剩余的也充分利用
    print(X.shape)
    print(len(train_loader.dataset))
    # x = np.random.random((16,16,2))
    # x = data_augmentation(x,2)
    # print(x.shape)


