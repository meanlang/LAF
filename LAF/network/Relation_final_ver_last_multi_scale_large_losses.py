import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from network.resnet import resnet50

# 定义了一个整体的model，其实也是基于resnet50做的，↓
# 我简单阐述一下整体流程
'''
他这个事情是这样子的
首先一张图像先进入RESNET50（这玩意儿直接是定义好的，就在那个resnet50.py里面，其实pytorch也有现成的）
走完RESNET50以后等于是一个基本的特征提取，会得到一个基本的特征图
得到特征图以后需要把那六个特征截出来（其实也简单，就是指定六个段，就是说那个高度，指定起始位置和结束位置就截出来了）
需要注意的是，论文里面写的是6个特征，但是其实源代码里面他做了3种不同的组合，一个是6个，一个是4个，一个是2个，最后每一份结果都要
还有需要注意的是，源代码里最先做的是GCP，然后再做的是ONE VS REST
但是最后两个得到的特征向量是都需要的，因此两个是都需要做的
还有：ONE VS NET比GCP复杂得多，几乎复杂了十倍左右，因为前者这个等于每个特征向量都要和剩下五个计算一次，等于做了6次运算
具体建议可以去论文看看效果，这里展开来说就太多了（其实我也批注了，在源码里面可以看到）
'''
class RelationModel(nn.Module):
    def __init__(
        self,
        last_conv_stride=1,
        last_conv_dilation=1,
        num_stripes=6,
        local_conv_out_channels=256,
        num_classes=0):
        super(RelationModel, self).__init__()

        self.base = resnet50(
            pretrained=True,
            last_conv_stride=last_conv_stride,
            last_conv_dilation=last_conv_dilation)
        self.num_stripes = num_stripes
        self.num_classes = num_classes

        self.local_6_conv_list = nn.ModuleList()
        self.local_4_conv_list = nn.ModuleList()
        self.local_2_conv_list = nn.ModuleList()
        self.rest_6_conv_list = nn.ModuleList()
        self.rest_4_conv_list = nn.ModuleList()
        self.rest_2_conv_list = nn.ModuleList()
        self.relation_6_conv_list = nn.ModuleList()
        self.relation_4_conv_list = nn.ModuleList()
        self.relation_2_conv_list = nn.ModuleList()
        self.global_6_max_conv_list = nn.ModuleList()
        self.global_4_max_conv_list = nn.ModuleList()
        self.global_2_max_conv_list = nn.ModuleList()
        self.global_6_rest_conv_list = nn.ModuleList()
        self.global_4_rest_conv_list = nn.ModuleList()
        self.global_2_rest_conv_list = nn.ModuleList()
        self.global_6_pooling_conv_list = nn.ModuleList()
        self.global_4_pooling_conv_list = nn.ModuleList()
        self.global_2_pooling_conv_list = nn.ModuleList()
        
        for i in range(num_stripes):
            self.local_6_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

            
        for i in range(4):
            self.local_4_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))


        for i in range(2):
            self.local_2_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))
            
        for i in range(num_stripes):
            self.rest_6_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))


        for i in range(4):
            self.rest_4_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

            
        for i in range(2):
            self.rest_2_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

        self.global_6_max_conv_list.append(nn.Sequential(
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        
        self.global_4_max_conv_list.append(nn.Sequential(
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        
        self.global_2_max_conv_list.append(nn.Sequential(
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))
        
        self.global_6_rest_conv_list.append(nn.Sequential(
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        
        self.global_4_rest_conv_list.append(nn.Sequential(
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        
        self.global_2_rest_conv_list.append(nn.Sequential(
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))
            
        for i in range(num_stripes):
            self.relation_6_conv_list.append(nn.Sequential(
                nn.Conv2d(local_conv_out_channels*2, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

            
        for i in range(4):
            self.relation_4_conv_list.append(nn.Sequential(
                nn.Conv2d(local_conv_out_channels*2, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

            
        for i in range(2):
            self.relation_2_conv_list.append(nn.Sequential(
                nn.Conv2d(local_conv_out_channels*2, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))
            
        self.global_6_pooling_conv_list.append(nn.Sequential(
            nn.Conv2d(local_conv_out_channels*2, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        
        self.global_4_pooling_conv_list.append(nn.Sequential(
            nn.Conv2d(local_conv_out_channels*2, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        
        self.global_2_pooling_conv_list.append(nn.Sequential(
            nn.Conv2d(local_conv_out_channels*2, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))
        
            
        if num_classes > 0:
            self.fc_local_6_list = nn.ModuleList()
            for _ in range(num_stripes):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_local_6_list.append(fc)


            self.fc_local_4_list = nn.ModuleList()
            for _ in range(4):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_local_4_list.append(fc)

                
            self.fc_local_2_list = nn.ModuleList()
            for _ in range(2):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_local_2_list.append(fc)
                
            self.fc_rest_6_list = nn.ModuleList()
            for _ in range(num_stripes):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_rest_6_list.append(fc)


            self.fc_rest_4_list = nn.ModuleList()
            for _ in range(4):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_rest_4_list.append(fc)


            self.fc_rest_2_list = nn.ModuleList()
            for _ in range(2):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_rest_2_list.append(fc)
                
            self.fc_local_rest_6_list = nn.ModuleList()
            for _ in range(num_stripes):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_local_rest_6_list.append(fc)


            self.fc_local_rest_4_list = nn.ModuleList()
            for _ in range(4):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_local_rest_4_list.append(fc)

                
            self.fc_local_rest_2_list = nn.ModuleList()
            for _ in range(2):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_local_rest_2_list.append(fc)

            self.fc_global_6_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_6_list.append(fc)

            
            self.fc_global_4_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_4_list.append(fc)

            
            self.fc_global_2_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_2_list.append(fc)
            
            self.fc_global_max_6_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_max_6_list.append(fc)

            
            self.fc_global_max_4_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_max_4_list.append(fc)

            
            self.fc_global_max_2_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_max_2_list.append(fc)
            
            self.fc_global_rest_6_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_rest_6_list.append(fc)

            self.fc_global_rest_4_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_rest_4_list.append(fc)

            
            self.fc_global_rest_2_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_rest_2_list.append(fc)
        # 全部都是定义，有点多
        # 先不管（深度学习最高奥义：不管）

    def forward(self, x):
        # 首先输入数据，就是这个x，这个x就是，之前dataloader当中咱们做好的一个数据源
        # 有疑惑的看我那个dataloader里面的批注，看完就行
        criterion = nn.CrossEntropyLoss()
        # print(x.shape) # 8 3 384 128 ←也就是打印一下当前的输入的shape值
        # 这个8是batch size影响的，3是因为是RGB图像，384和128的意思是h和w值
        feat = self.base(x) # 8 3 2048 24 8 ←这个也是shape值，8和上面一样，然后3也是一样的，2048是channel，h是24，w是8，这个是特征图
        '''
         第一步，这个base就是那个resnet
         咋说呢，就是把这个x放到resnet里面，然后会得到一个特征图
         简而言之，就是x经过resnet得到的结果
         '''
        #print(feat.shape) 
        assert (feat.size(2) % self.num_stripes == 0)
        stripe_h_6 = int(feat.size(2) / self.num_stripes) # 4
        stripe_h_4 = int(feat.size(2) / 4) # 6
        stripe_h_2 = int(feat.size(2) / 2) # 12
        '''
        然后你看，这里算了三个值，分别对应上面提过的做的三组实验
        就是一个特征图，然后分别分2 4 6份，这里就是做一个分割的分别计算
        分出来的高度批注在后面了
        然后就会得到一堆local
        '''
        local_6_feat_list = []
        local_4_feat_list = []
        local_2_feat_list = []
        '''
        这里说的是，在ONE VS NET里面，需要先单独给分出的部分，算一下自己的local特征
        但是分2 4 6份，那么对应的local特征肯定是不一样的
        这三行的意思就是分别计算对应的local特征值
        '''
        final_feat_list = []
        # 这个是最后组合在一起的final，当然现在还看不出来，不急
        logits_list = []
        # 这个是预测的结果，
        rest_6_feat_list = []
        rest_4_feat_list = []
        rest_2_feat_list = []
        '''
        rest就是：
        提出来的这个local，比如P1的在文章里面叫做P1（P上面一横）
        然后剩下的P2-P6，他们整合在一起，提出的特征就叫做rest
        '''
        logits_local_rest_list = []
        logits_local_list = []
        logits_rest_list = []
        logits_global_list = []
        '''
        logits 这一块，主要是emm，一个一个说吧 
        local_rest就是One VS REST里面最后那个拼接完的结果，就是最后那个结果
        其他的就是emmm自己对应自己的特征
        值得注意的是，这个global意思其实是GCP那里面提出来的那个特征图的（GCP的最终特征图）
        '''

        '''
        这一块构筑了一堆list
        '''
        
        
        for i in range(self.num_stripes):
        # 得到6块中每一个的特征 用的遍历这个做法
            local_6_feat = F.max_pool2d(
                feat[:, :, i * stripe_h_6: (i + 1) * stripe_h_6, :],
                # 这个feature其实就是咱们第一个做的那个经过resnet50过来得到的一个特征
                # 具体操作是：‘：’表示取全部意思就是所有batch我都要，第二个‘：’指的是所有channel我也都要，最后一个‘：’指的是所有的w也全要
                # 中间这个i * stripe_h_6: (i + 1) * stripe_h_6，意思是咱们经过计算的h（刚才算过了已经），从0到4，一共五个，则实现了取的高度的遍历
                # 每一块是4*w
                (stripe_h_6, feat.size(-1)))
                #maxpool成1*1的（展平操作）
            #print(local_6_feat.shape) #8 2048 1 1 等于是得到了一个2048的一个特征向量
            
            local_6_feat_list.append(local_6_feat)
            # 按顺序得到每一块的特征，一块一块走
            # local_6_feat_list这个里面存的是每一块的单独的local特征
            # 思想是酱紫的，先拿到这六个，但是并不会直接计算ONE VS REST这个模块，而是先计算相对简单的GCP
        
        '''
        下面是GCP操作
        '''
            
        global_6_max_feat = F.max_pool2d(feat, (feat.size(2), feat.size(3))) # 8 2048 1 1，全局特征
        # 先进行一个maxpooling操作，需要注意的是，这个F.max_pool2d(feat 的feature并不是local的feature，而是原始的feature
        # 然后下面两个feat.size则是h和w的数值
        # 这个得到的是哪个Pmax
        # print(global_6_max_feat.shape)
        global_6_rest_feat = (local_6_feat_list[0] + local_6_feat_list[1] + local_6_feat_list[2]#局部与全局的差异 
                              + local_6_feat_list[3] + local_6_feat_list[4] + local_6_feat_list[5] - global_6_max_feat)/5
        # 怎么说呢，之前把6个局部提出来，只需要把这六个局部特征加在一起，其实就直接是平均特征了，等于一个平均池化操作
        # 然后呢，再减去我们之前提出来的这个Pmax的，就得到了局部和全局的差异，再除以5，得到的是Pcont（看论文就懂了）
        # print(global_6_rest_feat.shape) # 8 2048 1 1
        global_6_max_feat = self.global_6_max_conv_list[0](global_6_max_feat) # 8 256 1 1
        # 卷积操作，把2048维，降到256就行
        # 具体操作是可以看源
        # 还是说一下吧，大概就是先conv2d然后batchnorm2d一下最后relu就好
        # 让他从Pmax变成Pmax（P上面多一横）
        # print(global_6_max_feat.shape)
        global_6_rest_feat = self.global_6_rest_conv_list[0](global_6_rest_feat) # 8 256 1 1
        # 同样的操作
        # 这个是那个差异的那个Pcont
        # print(global_6_rest_feat.shape)
        # 也是一样的，降到256维
        global_6_max_rest_feat = self.global_6_pooling_conv_list[0](torch.cat((global_6_max_feat, global_6_rest_feat), 1))
        # 这个是完成的那个拼接的操作，把两个256的拼接在一起，然后呢拼接成一个512的，再经过一个卷积，然后得到一个256的特征向量
        # print(global_6_max_rest_feat.shape) # 8 256 1 1
        global_6_feat = (global_6_max_feat + global_6_max_rest_feat).squeeze(3).squeeze(2)
        '''
        ↑这一步很精妙，等于就算下面的分支很垃圾，也可以无所谓
        因为还有上面那个Pmax的分支可以用，非常巧妙
        感觉是参考了resnet50的思想完成的
        妙啊
        '''
        #print(global_6_feat.shape)
        #论文中Global contrastive feature Figure2(b)

        '''
        下面是ONE VS REST的具体操作
        '''

        for i in range(self.num_stripes):
        #对于每块特征，除去自己之后其他的特征组合在一起
            
            rest_6_feat_list.append((local_6_feat_list[(i+1)%self.num_stripes]#论文公式1处的ri 
                                   + local_6_feat_list[(i+2)%self.num_stripes]
                                   + local_6_feat_list[(i+3)%self.num_stripes] 
                                   + local_6_feat_list[(i+4)%self.num_stripes]
                                   + local_6_feat_list[(i+5)%self.num_stripes])/5)
            '''
            这个是第一步操作，大概就是先取遍历操作
            详解一下这个：
            一开始的时候，local是0，然后rest就是1 2 3 4 5，一个一个加上去
            比较值得关注的是，如果local是1，rest反而是2 3 4 5 0，因为有%这个操作，取余操作，机智啊
            这一步的意思就是，把分别对于0-5的对应的rest取出来，也就是公式1
            除了当前的这个指定的local，然后把其他的全部加起来然后除以5（统一尺寸）
            这一步就是构建Ri
            值得注意的是，这一块是属于构建的那个6分块的部分
            下面部分其实是一样的，只是变成了构建4块和2块的部分
            他是把2 4 6全部做好rest，存list，然后要用就直接取
            而不是一个一个做
            '''
            
        for i in range(4):
            local_4_feat = F.max_pool2d(
                feat[:, :, i * stripe_h_4: (i + 1) * stripe_h_4, :],
                (stripe_h_4, feat.size(-1)))
            #print(local_4_feat.shape)
            
            local_4_feat_list.append(local_4_feat)
        # 算这四块每一块具体是什么，然后存在local_4_feat_list里面
        
        
            
        global_4_max_feat = F.max_pool2d(feat, (feat.size(2), feat.size(3)))
        #print(global_4_max_feat.shape) # 8 2048 1 1
        global_4_rest_feat = (local_4_feat_list[0] + local_4_feat_list[1] + local_4_feat_list[2] 
                              + local_4_feat_list[3] - global_4_max_feat)/3
        #print(global_4_rest_feat.shape)                     
        global_4_max_feat = self.global_4_max_conv_list[0](global_4_max_feat) # 8 256 1 1
        #print(global_4_max_feat.shape)
        global_4_rest_feat = self.global_4_rest_conv_list[0](global_4_rest_feat)  # 8 256 1 1
        #print(global_4_rest_feat.shape)
        global_4_max_rest_feat = self.global_4_pooling_conv_list[0](torch.cat((global_4_max_feat, global_4_rest_feat), 1))
        #print(global_4_max_rest_feat.shape) # 8 256 1 1
        global_4_feat = (global_4_max_feat + global_4_max_rest_feat).squeeze(3).squeeze(2)
        #print(global_4_feat.shape) # 依旧是16 256
        # 和上面那个大模块完全一致

        for i in range(4):
            
            rest_4_feat_list.append((local_4_feat_list[(i+1)%4] 
                                   + local_4_feat_list[(i+2)%4]
                                   + local_4_feat_list[(i+3)%4])/3)

        # 和上面大模块的操作完全一致，就是改了改数量而已
        for i in range(2):
            local_2_feat = F.max_pool2d(
                feat[:, :, i * stripe_h_2: (i + 1) * stripe_h_2, :],
                (stripe_h_2, feat.size(-1)))
            #print(local_2_feat.shape)
            local_2_feat_list.append(local_2_feat)
        
        
            
        global_2_max_feat = F.max_pool2d(feat, (feat.size(2), feat.size(3)))
        #print(global_2_max_feat.shape)
        global_2_rest_feat = (local_2_feat_list[0] + local_2_feat_list[1] - global_2_max_feat)
        #print(global_2_rest_feat.shape)
        global_2_max_feat = self.global_2_max_conv_list[0](global_2_max_feat)
        #print(global_2_max_feat.shape)
        global_2_rest_feat = self.global_2_rest_conv_list[0](global_2_rest_feat)
        #print(global_2_rest_feat.shape)
        global_2_max_rest_feat = self.global_2_pooling_conv_list[0](torch.cat((global_2_max_feat, global_2_rest_feat), 1))
        #print(global_2_max_rest_feat.shape)
        global_2_feat = (global_2_max_feat + global_2_max_rest_feat).squeeze(3).squeeze(2)
        #print(global_2_feat.shape)
        for i in range(2):
            
            rest_2_feat_list.append((local_2_feat_list[(i+1)%2]))
        # 和上面两个大模块完全一致

        '''
        前面等于全是准备工作准备好了R1
        下面才是核心要走ONE VS REST的核心部分
        '''
            
        for i in range(self.num_stripes):
        # 这个比较麻烦一点，因为要算P1和所有rest，P2和所有rest等等，所以直接用遍历的操作比较好

            local_6_feat = self.local_6_conv_list[i](local_6_feat_list[i]).squeeze(3).squeeze(2)#pi
            # 首先拿到当前的local特征（local_6_conv_list）传进来一个i，然后进行卷积，算出P1（P上多一横）转化为一个256维的特征
            #print(local_6_feat.shape)
            input_rest_6_feat = self.rest_6_conv_list[i](rest_6_feat_list[i]).squeeze(3).squeeze(2)#ri
            # 也是一样的，和上面的Pi一样，卷积，然后得到一个256维的特征
            # 值得注意的是rest_6_feat_list里面存有除了当前特征之外的每一个特征
            # 我多说一句，这个卷积的规格都是一样的，都是把2048降维到256的
            # print(input_rest_6_feat.shape)
            input_local_rest_6_feat = torch.cat((local_6_feat, input_rest_6_feat), 1).unsqueeze(2).unsqueeze(3)
            # ok开始拼接，把上面两个拼一起，得到一个512维的特征
            # print(input_local_rest_6_feat.shape) # 8 512 1 1
            local_rest_6_feat = self.relation_6_conv_list[i](input_local_rest_6_feat)
            # 要做加法操作，但是上面的分支下来的是256的，你这个是512的
            # 这一步就是说，要做一个卷积的操作，把这个512的降到256，方便可以做对应的事情
            #print(local_rest_6_feat.shape) # 8 256 1 1
            local_rest_6_feat = (local_rest_6_feat 
                               + local_6_feat.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)
            # 然后把两个做一个加法操作，把两个特征加起来得到最终特征
            # print(local_rest_6_feat.shape)# 16 256
            final_feat_list.append(local_rest_6_feat)
            # 最后一步的结果，存起来，待会用
            

            
            if self.num_classes > 0:
                logits_local_rest_list.append(self.fc_local_rest_6_list[i](local_rest_6_feat))
                # 当前local和rest的分类结果 i是每一份都是自己的，需要分开
                logits_local_list.append(self.fc_local_6_list[i](local_6_feat))
                # 当前local的分类结果
                logits_rest_list.append(self.fc_rest_6_list[i](input_rest_6_feat))
                # 当前rest的分类结果
        #print(np.array(logits_local_rest_list).shape)实际上做了6次
        # 分别用这几个特征来识别人物，判断具体每个分支的效果（参悟不够）
        # 大概意思就是，每一步我都需要让他很强- -大概就这么个意思

        # 下面是分四份，一模一样

        for i in range(4):
            
            local_4_feat = self.local_4_conv_list[i](local_4_feat_list[i]).squeeze(3).squeeze(2)
            #print(local_4_feat.shape)
            input_rest_4_feat = self.rest_4_conv_list[i](rest_4_feat_list[i]).squeeze(3).squeeze(2)
            #print(input_rest_4_feat.shape)
            input_local_rest_4_feat = torch.cat((local_4_feat, input_rest_4_feat), 1).unsqueeze(2).unsqueeze(3)
            #print(input_local_rest_4_feat.shape)
            local_rest_4_feat = self.relation_4_conv_list[i](input_local_rest_4_feat)
            #print(local_rest_4_feat.shape)
            local_rest_4_feat = (local_rest_4_feat 
                               + local_4_feat.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)
            #print(local_rest_4_feat.shape)
            final_feat_list.append(local_rest_4_feat)

            
            if self.num_classes > 0:
                logits_local_rest_list.append(self.fc_local_rest_4_list[i](local_rest_4_feat))
                logits_local_list.append(self.fc_local_4_list[i](local_4_feat))
                logits_rest_list.append(self.fc_rest_4_list[i](input_rest_4_feat))
        # 必须要注意到的是，这里面突然变成了一个append操作，这个是一个往里面添加的操作，所以他的shape会突然变成10，不是错了
        # 他就是4+6就是10
        #print(np.array(logits_local_rest_list).shape)
                
        for i in range(2):

            local_2_feat = self.local_2_conv_list[i](local_2_feat_list[i]).squeeze(3).squeeze(2)
            #print(local_2_feat.shape)
            input_rest_2_feat = self.rest_2_conv_list[i](rest_2_feat_list[i]).squeeze(3).squeeze(2)
            #print(input_rest_2_feat.shape)
            input_local_rest_2_feat = torch.cat((local_2_feat, input_rest_2_feat), 1).unsqueeze(2).unsqueeze(3)
            #print(input_local_rest_2_feat.shape)
            local_rest_2_feat = self.relation_2_conv_list[i](input_local_rest_2_feat)
            #print(local_rest_2_feat.shape)
            local_rest_2_feat = (local_rest_2_feat 
                               + local_2_feat.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)
            #print(local_rest_2_feat.shape)
            final_feat_list.append(local_rest_2_feat)
            

            
            if self.num_classes > 0:
                logits_local_rest_list.append(self.fc_local_rest_2_list[i](local_rest_2_feat))
                logits_local_list.append(self.fc_local_2_list[i](local_2_feat))
                logits_rest_list.append(self.fc_rest_2_list[i](input_rest_2_feat))
        # 基础操作和前面完全一样
        # 需要注意的是，这里同样也是append操作，意思就是说，这里应该是属于4+6+2的
        # 也就是说这里的shape是12
        #print(np.array(logits_local_rest_list).shape)
                
                
                
            
        final_feat_list.append(global_6_feat)
        final_feat_list.append(global_4_feat)
        final_feat_list.append(global_2_feat)
        # 这里的意思是定义一些最终特征的list
        # 但是现在只是简单把ONE VS REST的最终特征存进去了，还没有包括GCP的
        # 所以这一步的操作就是把之前我们已经做好的GCP的最终特征也放进去
        # print(np.array(logits_local_rest_list).shape) 其实也是12

        if self.num_classes > 0:
            # 一共九种分类，分别是6 4 2，每一种三次
            # 每个都是751个分类任务
            
            logits_global_list.append(self.fc_global_6_list[0](global_6_feat))
            logits_global_list.append(self.fc_global_max_6_list[0](global_6_max_feat.squeeze(3).squeeze(2)))
            logits_global_list.append(self.fc_global_rest_6_list[0](global_6_rest_feat.squeeze(3).squeeze(2)))
            logits_global_list.append(self.fc_global_4_list[0](global_4_feat))
            logits_global_list.append(self.fc_global_max_4_list[0](global_4_max_feat.squeeze(3).squeeze(2)))
            logits_global_list.append(self.fc_global_rest_4_list[0](global_4_rest_feat.squeeze(3).squeeze(2)))
            logits_global_list.append(self.fc_global_2_list[0](global_2_feat))
            logits_global_list.append(self.fc_global_max_2_list[0](global_2_max_feat.squeeze(3).squeeze(2)))
            logits_global_list.append(self.fc_global_rest_2_list[0](global_2_rest_feat.squeeze(3).squeeze(2)))
            #print(np.array(logits_global_list).shape) 最后是9
            return final_feat_list, logits_local_rest_list, logits_local_list, logits_rest_list, logits_global_list
            # 这里是几个损失函数
            # 最后final_feat_list里面的是2，4,6的，加一起，就是12个特征，把所有的结果返回去就行了，要得就是这个东西
        
        
        return final_feat_list
    
        