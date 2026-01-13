#添加关系矩阵：共存关系图+直接跟随关系图
#PConv替换第一层卷积
#MSF替换全连接层
#推理
from random import randint
import pandas as pd
import numpy as np
import pprint
import copy
from sklearn import model_selection
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import dataloader, dataset, TensorDataset
import torch
from time import *
from torch.nn import init
pd.set_option('display.max_columns', None)
dataframe = pd.read_csv("log/0.15bpi2012.csv", low_memory=False, index_col=0)
npdata = dataframe.values

def nptolist(npdata):
    l1 = []
    n = 0
    for i in npdata:
        l2 = []
        for j in i:
            if isinstance(j, str):
                l2.append(j)
        l1.append(l2)
        n += 1
    print("***********")
    return l1


def geteventlist(l):
    eventlist = []
    for i in l:
        for j in i:
            if j not in eventlist:
                eventlist.append(j)
    return eventlist, len(eventlist)


def gettraceevent(trace):
    l = []
    for i in trace:
        if i not in l:
            l.append(i)
    return l


def initbkanktrace(eventistlen):
    l = []
    for i in range(eventistlen):
        l.append(0)
    return l


def inittaglist(eventistlen):  # 初始化tag
    l = []
    for i in range(eventistlen):
        j = []
        for i in range(eventistlen):
            j.append(0)
        l.append(j)
    return l


def delevent(data):  # 删除活动并返回被删除的活动以及它直接前继与后继
    l1 = data.copy()
    l2 = []
    for i in l1:
        length = len(i)
        r = randint(0, length - 1)
        l2.append(i[r])
        del i[r]
    return l1, l2


def getallindex(l1, m):  # 获得列表中所有元素的下标,l1是列表，m是要查的元素
    length = len(l1)
    n = l1.count(m)
    indexn = []
    index = 0
    for i in list(range(n)):
        index = l1.index(m, index, length)
        indexn.append(index)
        index += 1
    return indexn


def getorder(log, eventlist, eventlistlen):
    l = eventlist
    ll = inittaglist(eventlistlen)
    for i in range(eventlistlen):
        t1 = []
        for trace in log:
            if l[i] in trace:
                t2 = []
                t1 = getallindex(trace, l[i])
                l1 = gettraceevent(trace)  # 临时列表，储存特定迹的事件
                for j in l1:
                    if l[i] != j:
                        t2 = getallindex(trace, j)
                        index = l.index(j)
                        if max(t1) < min(t2):
                            if ll[i][index] != 2 and ll[index][i] != 2:
                                ll[i][index] = 1
                                ll[index][i] = 1
                        elif max(t1) > min(t2) and min(t1) < max(t2):
                            ll[i][index] = 2
                            ll[index][i] = 2
                        elif max(t1) > max(t2) and ll[i][index] == 1:  # 前面俩个是迹内判断，这个是迹间判断，
                            # 虽然放在同一级循环内，但是逻辑是俩层判断才形成的迹间判断
                            ll[i][index] = 2
                            ll[index][i] = 2
                    else:
                        number = trace.count(j)
                        if number == 1 and ll[i][i] != 2:
                            ll[i][i] = 3
                        else:
                            ll[i][i] = 2
        t1.clear()
    for i in range(len(ll)):
        for j in range(len(ll)):
            if ll[i][j] == 0:
                ll[i][j] = 3
                print("改了")
    return ll

#共存关系图
def get_coexistence_graph(log, eventlist):
    size = len(eventlist)
    coexist_graph = [[0]*size for _ in range(size)]
    for trace in log:
        unique_events = set(trace)
        for e1 in unique_events:
            for e2 in unique_events:
                i, j = eventlist.index(e1), eventlist.index(e2)
                coexist_graph[i][j] = 1
    return coexist_graph

#直接跟随关系图
def get_direct_follow_graph(log, eventlist):
    size = len(eventlist)
    direct_follow_graph = [[0]*size for _ in range(size)]
    for trace in log:
        for i in range(len(trace)-1):
            a = trace[i]
            b = trace[i+1]
            if a in eventlist and b in eventlist:
                idx_a = eventlist.index(a)
                idx_b = eventlist.index(b)
                direct_follow_graph[idx_a][idx_b] = 1
    return direct_follow_graph


def getconcurrencegraph(ll):
    lc = copy.deepcopy(ll)
    l = len(ll)
    for i in range(l):
        for j in range(l):
            if ll[i][j] != 2:
                lc[i][j] = 0
            else:
                lc[i][j] = 1
    return lc


def getqequencegraph(ll):
    ls = copy.deepcopy(ll)
    l = len(ll)
    for i in range(l):
        for j in range(l):
            if ll[i][j] != 1:
                ls[i][j] = 0
    return ls


def getexclusivegraph(ll):
    le = copy.deepcopy(ll)
    l = len(ll)
    for i in range(l):
        for j in range(l):
            if ll[i][j] != 3:
                le[i][j] = 0
            else:
                le[i][j] = 1
    return le


def expandlist(l, eventlistl, blanklist):
    tblanklist = blanklist.copy()
    lenth = len(eventlistl)
    for i in range(176-lenth):
        l.append(tblanklist)
    return l


def getbehaviorgraph(li, l11, lc, le, ls, coexist, directf, blanklist):
    ms = []
    mc = []
    me = []
    mcoexist = []
    mdirectf = []
    molcoexist = coexist.copy()
    moldirectf = directf.copy()
    molc = lc.copy()
    mole = le.copy()
    mols = ls.copy()
    tblanklist = blanklist.copy()
    for i in l11:
        eventl = gettraceevent(i)
        tracelc = molc.copy()
        tracele = mole.copy()
        tracels = mols.copy()
        tracecoexist = molcoexist.copy()
        tracedirect = moldirectf.copy()
        for j in li:
            if j not in eventl:
                n = li.index(j)
                tracecoexist[n] = tblanklist
                tracedirect[n] = tblanklist

        for j in li:
            if j not in eventl:
                n = li.index(j)
                tracelc[n] = tblanklist
                tracele[n] = tblanklist
                tracels[n] = tblanklist
                # for j in range(eventlistlen):
                #     tracelc[n][j]=0
                #     tracele[n][j]=0
                #     tracels[n][j]=0
        tracels=expandlist(tracels,li,blanklist)
        tracele=expandlist(tracele,li,blanklist)
        tracelc=expandlist(tracelc,li,blanklist)
        tracecoexist = expandlist(tracecoexist, li, blanklist)
        tracedirect = expandlist(tracedirect, li, blanklist)
        ms.append(tracels)
        me.append(tracele)
        mc.append(tracelc)
        mcoexist.append(tracecoexist)
        mdirectf.append(tracedirect)
    return ms, me, mc, mcoexist, mdirectf


def getsln(trace, eventlist):
    sln = []
    length = len(trace)
    for i in range(length):
        line = []
        for j in range(len(eventlist)):
            line.append(0)
        l = trace[:i + 1]
        for e in eventlist:
            line[eventlist.index(e)] = l.count(e)
        sln.append(line)
    l = []
    for j in range(len(eventlist)):
        l.append(0)
    for i in range(176-len(trace) ):
        sln.append(l)
    return sln


def getslsc(trace, eventlist, blanklist):
    length = len(eventlist)  # 这是获取关系，所以不关心迹的长度
    sls = inittaglist(length)
    slc = inittaglist(length)
    el = gettraceevent(trace)
    for e in el:
        l1 = getallindex(trace, e)
        for j in el:
            if j != e:
                l2 = getallindex(trace, j)
                if max(l1) < min(l2):
                    sls[eventlist.index(e)][eventlist.index(j)] = 1
                    sls[eventlist.index(j)][eventlist.index(e)] = 1
                else:
                    if sls[eventlist.index(e)][eventlist.index(j)] != 1:
                        slc[eventlist.index(e)][eventlist.index(j)] = 1
                        slc[eventlist.index(j)][eventlist.index(e)] = 1
            else:
                if trace.count(e) != 1:
                    slc[eventlist.index(e)][eventlist.index(j)] = 1
                    slc[eventlist.index(j)][eventlist.index(e)] = 1
                # else:
                #     sls[eventlist.index(e)][eventlist.index(j)]=1
                #     sls[eventlist.index(j)][eventlist.index(e)]=1
    sls=expandlist(sls,eventlist,blanklist)
    slc=expandlist(slc,eventlist,blanklist)
    return sls, slc


def getmultigraph(das, eventlist, lc, le, ls, blanklist, coexist, directf):
    actgraph = []
    lln = []
    lls = []
    llc = []
    multifeaturelist1 = []
    multifeaturelist2 = []
    multifeaturelist3 = []
    for i in range(len(das)):
        sls, slc = getslsc(das[i], eventlist, blanklist)
        lls.append(sls)
        llc.append(slc)
        sln = getsln(das[i], eventlist)
        lln.append(sln)
    ms, me, mc, mcoexist, mdirectf = getbehaviorgraph(eventlist, das, lc, le, ls, coexist, directf, blanklist)
    print(len(lln[0]), len(lls[0]), len(llc[0]), len(ms[0]), len(me[0]), len(mc[0]))
    print(len(lln[0][0]), len(lls[0][0]), len(llc[0][0]), len(ms[0][0]), len(me[0][0]), len(mc[0][0]))
    print("*************************************************************************")
    for i in range(len(das)):
        multifeature1 = np.array([lln[i], lls[i], llc[i], mcoexist[i], mdirectf[i], ms[i], me[i], mc[i]])
        multifeature2 = np.array([lln[i]])
        multifeature3 = np.array([lln[i], lls[i], llc[i]])
        multifeaturelist1.append(multifeature1)
        multifeaturelist2.append(multifeature2)
        multifeaturelist3.append(multifeature3)
    multifeaturelist1 = np.array(multifeaturelist1)
    multifeaturelist2 = np.array(multifeaturelist2)
    multifeaturelist3 = np.array(multifeaturelist3)
    print("multifeature_length:", len(multifeaturelist1))
    return multifeaturelist1, multifeaturelist2, multifeaturelist3


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Pinwheel_shaped_Convolution(nn.Module):
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''

    def __init__(self, c1, c2, k, s):
        super().__init__()

        # self.k = k
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttentionFusion, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q_feature, K_feature):
        B, C, N = Q_feature.shape

        Q_feature = Q_feature.permute(0, 2, 1)
        K_feature = K_feature.permute(0, 2, 1)

        Q = self.query(Q_feature)  # shape: [B, N, C]
        K = self.key(K_feature)  # shape: [B, N, C]
        V = self.value(K_feature)  # shape: [B, N, C]

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(C, dtype=torch.float32))
        attention_weights = self.softmax(attention_scores)  # shape: [B, N, N]

        attended_features = torch.matmul(attention_weights, V)  # shape: [B, N, C]
        attended_features = attended_features.permute(0, 2, 1)

        return attended_features


class MSF(nn.Module):  # Attention-based Feature Fusion Module
    def __init__(self, feature_dim):
        super(MSF, self).__init__()
        self.CA1 = CrossAttentionFusion(feature_dim)
        self.CA2 = CrossAttentionFusion(feature_dim)
        self.relu = nn.ReLU()

    def forward(self, low, mid, high):
        low_new = self.CA1(mid, low)
        high_new = self.CA2(mid, high)
        fused_features = self.relu(low_new + mid + high_new)
        return fused_features


class CNN_Net1(nn.Module):
    def __init__(self):
        super(CNN_Net1, self).__init__()
        self.conv1 = Pinwheel_shaped_Convolution(c1=8, c2=8, k=3, s=1)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=4)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 新加一层，为高频特征

        self.msf = MSF(feature_dim=64)  # 输入通道数（注意需一致）

        self.out_conv = nn.Conv1d(64, 36, kernel_size=1)  # 相当于全连接分类

    def forward(self, x):
        low = F.relu(F.max_pool2d(self.conv1(x), 2))  # shape: [B, 8, H/2, W/2]
        mid = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(low)), 2))  # [B, 32, H/4, W/4]
        high = F.relu(F.max_pool2d(self.conv3(mid), 2))  # [B, 64, H/8, W/8]

        # 对应维度变换：将 [B, C, H, W] reshape 为 [B, C, H*W]
        # low_flat = F.interpolate(low, size=high.shape[2:], mode='bilinear')  # [B, 8, H/8, W/8]
        # mid_flat = F.interpolate(mid, size=high.shape[2:], mode='bilinear')  # [B, 32, H/8, W/8]
        low_flat = F.interpolate(low, size=high.shape[2:], mode='bilinear', align_corners=False)
        mid_flat = F.interpolate(mid, size=high.shape[2:], mode='bilinear', align_corners=False)
        low_flat = F.pad(low_flat, (0,0,0,0,0,64-8))  # pad channel to 64
        mid_flat = F.pad(mid_flat, (0,0,0,0,0,64-32))  # pad to 64
        low_flat = low_flat.view(low_flat.size(0), 64, -1)
        mid_flat = mid_flat.view(mid_flat.size(0), 64, -1)
        high_flat = high.view(high.size(0), 64, -1)  # already correct

        fused = self.msf(low_flat, mid_flat, high_flat)  # [B, 64, L]

        out = self.out_conv(fused)  # [B, 36, L]
        out = F.adaptive_avg_pool1d(out, 1).squeeze(-1)  # [B, 36]
        return F.log_softmax(out, dim=1)

    def getstr(self):
        return 'CNN_Net1'


class CNN_Net2(nn.Module):
    def __init__(self):
        super(CNN_Net2, self).__init__()
        self.conv1 = Pinwheel_shaped_Convolution(c1=1, c2=8, k=3, s=1)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=4)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.msf = MSF(feature_dim=64)
        self.out_conv = nn.Conv1d(64, 36, kernel_size=1)

    def forward(self, x):
        low = F.relu(F.max_pool2d(self.conv1(x), 2))
        mid = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(low)), 2))
        high = F.relu(F.max_pool2d(self.conv3(mid), 2))

        low_flat = F.interpolate(low, size=high.shape[2:], mode='bilinear', align_corners=False)
        mid_flat = F.interpolate(mid, size=high.shape[2:], mode='bilinear', align_corners=False)
        low_flat = F.pad(low_flat, (0, 0, 0, 0, 0, 64 - 8))
        mid_flat = F.pad(mid_flat, (0, 0, 0, 0, 0, 64 - 32))

        low_flat = low_flat.view(low_flat.size(0), 64, -1)
        mid_flat = mid_flat.view(mid_flat.size(0), 64, -1)
        high_flat = high.view(high.size(0), 64, -1)

        fused = self.msf(low_flat, mid_flat, high_flat)
        out = self.out_conv(fused)
        out = F.adaptive_avg_pool1d(out, 1).squeeze(-1)
        return F.log_softmax(out, dim=1)

    def getstr(self):
        return 'CNN_Net2'


class CNN_Net3(nn.Module):
    def __init__(self):
        super(CNN_Net3, self).__init__()
        self.conv1 = Pinwheel_shaped_Convolution(c1=3, c2=8, k=3, s=1)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=4)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.msf = MSF(feature_dim=64)
        self.out_conv = nn.Conv1d(64, 36, kernel_size=1)

    def forward(self, x):
        low = F.relu(F.max_pool2d(self.conv1(x), 2))
        mid = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(low)), 2))
        high = F.relu(F.max_pool2d(self.conv3(mid), 2))

        low_flat = F.interpolate(low, size=high.shape[2:], mode='bilinear', align_corners=False)
        mid_flat = F.interpolate(mid, size=high.shape[2:], mode='bilinear', align_corners=False)
        low_flat = F.pad(low_flat, (0, 0, 0, 0, 0, 64 - 8))
        mid_flat = F.pad(mid_flat, (0, 0, 0, 0, 0, 64 - 32))

        low_flat = low_flat.view(low_flat.size(0), 64, -1)
        mid_flat = mid_flat.view(mid_flat.size(0), 64, -1)
        high_flat = high.view(high.size(0), 64, -1)

        fused = self.msf(low_flat, mid_flat, high_flat)
        out = self.out_conv(fused)
        out = F.adaptive_avg_pool1d(out, 1).squeeze(-1)
        return F.log_softmax(out, dim=1)

    def getstr(self):
        return 'CNN_Net3'


def labeltotorch_tensor(l, eventlist):
    labellist = []
    for i in l:
        labellist.append(eventlist.index(i))
    return labellist


def trainandtest(train_loader,test_loader,cnn,loss_func,lrc,train_dataset,epochs,device):
    model=cnn().to(device)
    opt= torch.optim.SGD(model.parameters(), lr=lrc, momentum=0.9)
    trainloss_count = []
    for epoch in range(epochs):
        running_loss = 0
        running_acc = 0
        for i, (x, y) in enumerate(train_loader):
            # batch_x = Variable(x)
            # batch_y = Variable(y)
            batch_x = x.to(device)
            batch_y = y.to(device)
            # 获取最后输出
            out = model(batch_x)  # torch.Size([128,10])
            # 获取损失
            loss = loss_func(out, batch_y)
            # 使用优化器优化损失
            opt.zero_grad()  # 清空上一步残余更新参数值
            loss.backward()  # 误差反向传播，计算参数更新值
            opt.step()  # 将参数更新值施加到net的parmeters上
            running_loss += loss.item()
            _, predict = torch.max(out, 1)
            correct_num = (predict == batch_y).sum()
            running_acc += correct_num.item()
            if i % 40 == 0:
                loss1 = loss.cpu()
                trainloss_count.append(loss1.detach().numpy())
                print('{}:\t'.format(i), loss.item())
                # torch.save(model, r'log_CNN1'+str(lrc)+str(epochs))
                # torch.save(model,'log_CNN_new'+str(lrc)+model.getstr())
        running_loss /= len(train_dataset)
        running_acc /= len(train_dataset)
        print("[%d/%d] Loss: %.5f, Acc: %.5f" % (epoch + 1, epochs, running_loss, 100 * running_acc))
    torch.save(model, 'train_bpi2012' + str(lrc) + model.getstr())
    model.eval()
    testloss = 0
    testacc = 0
    outlist = []
    prelist = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            output1 = output.cpu()
            testloss += F.nll_loss(output, target.to(device), reduction='sum').item()
            pred = output1.max(1)[1]
            pre = pred.clone()
            pre = pre.detach().numpy()
            for i in pre:
                prelist.append(i)
            p = output1.max(1)[0]
            p = p.detach().numpy()
            for i in p:
                outlist.append(i)
            testacc += pred.eq(target).sum().item()
    testloss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(testloss, testacc,
            len(test_loader.dataset),100. * testacc / len(test_loader.dataset)))
    return 100. * testacc / len(test_loader.dataset)


def loadmodelandpre(test_loader,smodel,device):
    # model=torch.load(smodel)
    model = torch.load(smodel, map_location=torch.device('cpu'))
    model.eval()
    testloss = 0
    testacc = 0
    outlist = []
    prelist = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            output1 = output.cpu()
            testloss += F.nll_loss(output, target.to(device), reduction='sum').item()
            pred = output1.max(1)[1]
            pre = pred.clone()
            pre = pre.detach().numpy()
            for i in pre:
                prelist.append(i)
            p = output1.max(1)[0]
            p = p.detach().numpy()
            for i in p:
                outlist.append(i)
            testacc += pred.eq(target).sum().item()
    testloss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(testloss, testacc,
            len(test_loader.dataset),100. * testacc / len(test_loader.dataset)))
    return 100. * testacc / len(test_loader.dataset),prelist


if __name__ == '__main__':
    # start_time=time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = nptolist(npdata)
    datatrain, datatest = model_selection.train_test_split(npdata, train_size=0.77)
    datatrain = nptolist(datatrain)
    datatest = nptolist(datatest)
    eventlist, eventlistlen = geteventlist(data)
    print('eventlist:', len(eventlist))
    blanklist = initbkanktrace(eventlistlen)
    ll = getorder(data, eventlist, eventlistlen)
    lc = getconcurrencegraph(ll)
    ls = getqequencegraph(ll)
    le = getexclusivegraph(ll)
    coexist = get_coexistence_graph(data, eventlist)
    directf = get_direct_follow_graph(data, eventlist)
    datatraintrace, datatrainlabel = delevent(datatrain)
    datatesttrace, datatestlabel = delevent(datatest)
    trainlabel1, trainlabel2 = [], []
    for i in range(len(datatrainlabel)):
        l = []
        l.append(datatrainlabel[i])
        trainlabel1.append(l)
    for i in range(len(datatestlabel)):
        l = []
        l.append(datatestlabel[i])
        trainlabel2.append(l)
    datatestlabel = labeltotorch_tensor(datatestlabel, eventlist)
    datatrainlabel = labeltotorch_tensor(datatrainlabel, eventlist)
    train_datal = np.array(datatrainlabel)
    test_datal = np.array(datatestlabel)
    multitrain1, multitrain2, multitrain3 = getmultigraph(datatraintrace, eventlist, lc, le, ls, blanklist, coexist,
                                                          directf)
    multitest1, multitest2, multitest3 = getmultigraph(datatesttrace, eventlist, lc, le, ls, blanklist, coexist,
                                                       directf)
    # trainlabel1,trainlabel2=[],[]
    # for i in range(len(datatrainlabel)):
    #     l=[]
    #     l.append(datatrainlabel[i])
    #     trainlabel1.append(l)
    # for i in range(len(datatestlabel)):
    #     l=[]
    #     l.append(datatestlabel[i])
    #     trainlabel2.append(l)
    datatraintrace1 = pd.DataFrame(datatraintrace)
    datatesttrace2 = pd.DataFrame(datatesttrace)
    label1 = pd.DataFrame(trainlabel1)
    label2 = pd.DataFrame(trainlabel2)
    # tr2=pd.DataFrame(multitrain2)
    # tr3=pd.DataFrame(multitrain3)
    # datatraintrace1.to_csv('log/datatraintrace1.csv')
    # datatesttrace2.to_csv('log/datatesttrace2.csv')
    # label1.to_csv('log/label1.csv')
    # label2.to_csv('log/label2.csv')
    # tr2.to_csv('log/tr2.csv')
    # tr3.to_csv('log/tr3.csv')

    traindatas1 = torch.from_numpy(multitrain1)
    traindatas1 = traindatas1.float()
    traindatas2 = torch.from_numpy(multitrain2)
    traindatas2 = traindatas2.float()
    traindatas3 = torch.from_numpy(multitrain3)
    traindatas3 = traindatas3.float()

    testdatas1 = torch.from_numpy(multitest1)
    testdatas1 = testdatas1.float()
    testdatas2 = torch.from_numpy(multitest2)
    testdatas2 = testdatas2.float()
    testdatas3 = torch.from_numpy(multitest3)
    testdatas3 = testdatas3.float()

    trainlabeldatas = torch.from_numpy(train_datal)
    trainlabeldatas = trainlabeldatas.long()
    testlabeldatas = torch.from_numpy(test_datal)
    testlabeldatas = testlabeldatas.long()

    train_dataset1 = TensorDataset(traindatas1, trainlabeldatas)
    train_dataset2 = TensorDataset(traindatas2, trainlabeldatas)
    train_dataset3 = TensorDataset(traindatas3, trainlabeldatas)

    test_dataset1 = TensorDataset(testdatas1, testlabeldatas)
    test_dataset2 = TensorDataset(testdatas2, testlabeldatas)
    test_dataset3 = TensorDataset(testdatas3, testlabeldatas)

    train_loader1 = torch.utils.data.DataLoader(dataset=train_dataset1, batch_size=50)
    train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset2, batch_size=50)
    train_loader3 = torch.utils.data.DataLoader(dataset=train_dataset3, batch_size=50)

    test_loader1 = torch.utils.data.DataLoader(dataset=test_dataset1, batch_size=100)
    test_loader2 = torch.utils.data.DataLoader(dataset=test_dataset2, batch_size=100)
    test_loader3 = torch.utils.data.DataLoader(dataset=test_dataset3, batch_size=100)

    loss_func = torch.nn.CrossEntropyLoss()
    acclist31, acclist32, acclist33 = [], [], []
    acclist21, acclist22, acclist23 = [], [], []
    acclist11, acclist12, acclist13 = [], [], []
    start_time = time()
    # acclist31.append(trainandtest(train_loader1, test_loader1, CNN_Net1, loss_func, 0.003, train_dataset1, 100,device))
    # acclist31.append(trainandtest(train_loader2, test_loader2, CNN_Net2, loss_func, 0.003, train_dataset1, 100,device))
    # acclist31.append(trainandtest(train_loader3, test_loader3, CNN_Net3, loss_func, 0.003, train_dataset1, 100,device))
    # print(acclist31)
    pd1, prelist = loadmodelandpre(test_loader1, 'train_bpi2012_t10.003CNN_Net1', device)
    pd2, _ = loadmodelandpre(test_loader2, 'train_bpi2012_t10.003CNN_Net2', device)
    pd3, _ = loadmodelandpre(test_loader3, 'train_bpi2012_t10.003CNN_Net3', device)
    print(pd1)
    print(pd2)
    print(pd3)
    print(prelist[:10])
    end_time=time()
    print(end_time-start_time,":s"
                        )
