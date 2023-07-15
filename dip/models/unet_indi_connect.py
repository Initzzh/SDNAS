
# 将二维不定长的list平铺为一维list
def flatten(input_list):
    output_list = []
    while True:
        if input_list == []:
            break
        for index, value in enumerate(input_list):

            if type(value) == list:
                input_list = value + input_list[index + 1:]
                break
            else:
                output_list.append(value)
                input_list.pop(index)
                break
    return output_list

def check_active2(nl_num, connect_gene):
    nl_active = [None for _ in range(nl_num)]
    nl_connect = [] # 节点2，3的连接结构[[1],[0,1]]
    start_index = 0
    end_index = 1 # 用来划分connect_gene
    # 划分2，3节点的连接结构, nl_connect 中只有2，3的连接结构，1号节点无
    for _ in range(nl_num-1):
        nl_connect.append(connect_gene[start_index: start_index + end_index]) 
        start_index += end_index
        end_index += 1
    
    # 根据connect判断前驱节点是否激活使用
    for nl_con in nl_connect:
        for nl_index, nl_c in enumerate(nl_con):
            if nl_c==1:
                nl_active[nl_index] = True
    # 根据connect判断当前节点是否激活
    for nl_index, nl_con in enumerate(nl_connect):
        if sum(nl_con) >= 1:
            nl_active[nl_index+1] = True
    
    # 前驱节点list
    pre_index = []
    for nl_index in range(nl_num):
        if(nl_active[nl_index]):
            if nl_index == 0:
                # 1号节点
                pre_index.append([0]) # 上一层输出作为1号节点的输入
            else:
                # 2,3号节点 ， 无入边，有出边,则节点输入为上一层输出，即0号节点
                if sum(nl_connect[nl_index-1]) == 0:
                    pre_index.append([0])
                # 有入边，获取其前驱节点
                else:
                    tmp_index = [] # 前驱节点
                    for index, nl_con in enumerate(nl_connect[nl_index-1]):
                        if nl_con == 1:
                            tmp_index.append(index + 1)
                    pre_index.append(tmp_index)
        else:
            # 节点没有被连接
            pre_index.append([None])

    # 块最终输出边相连的节点
    out_index = []
    for nl_index in range(nl_num):
        # 将2，3号节点的前驱展开
        pre_index_ = flatten(pre_index[nl_index+1:])
        if nl_active[nl_index] and nl_index not in pre_index_:
            # nl激活，且不是其他节点的前驱，作为block的输出
            out_index.append(nl_index+1)
    # 当connect_gene
    if sum(connect_gene) == 0:
        # [0,0,0]
        nl_active = [True,None,None]
        pre_index = [[0],[None],[None]]
        out_index = [1]
    print("connect_arcitecture", nl_active, pre_index,out_index)
    return nl_active, pre_index, out_index


check_active2(3, [0,1,1])  


# unit 中节点的连接结构
def check_active(nl_num, connect_gene):
    """
        connect_gene:[0,0,0],[0,1,1]  000,001,010,011
    """
    # nl_num: block nl数
    nl_active=[None for _ in range(nl_num)]
    nl_connect = [] # [[0],[1,1]] # 连接
    i = 0  #
    j = 1
    # 将connect_gene 划分, such as [0,1,1]-->[[0],[1,1]]
    for _ in range(nl_num-1):
        nl_connect.append(connect_gene[i:i+j])
        i=i+j
        j+=1
    
    # 确定2，3号卷积节点是否使用，节点编号1，2，3 在nl_active的索引为0，1，2
    for p,nl in enumerate(nl_connect):
        # p:0,1 --> 对应的是2，3 号节点，对应nl_active的索引为1，2
        if sum(nl)>=1:
            nl_active[p+1]=True
    
    # 通过遍历前驱方法，来判断此前驱节点是否使用
    for k in range(nl_num-1):
        for nl in nl_connect:
            if k < len(nl) and nl[k]==1:
                # nl[k]在连接结构中的的第k个节点
                nl_active[k] = True

    # 最后一个节点连接
    if sum(connect_gene) >= 1:
        nl_active[nl_num-1] = True

    # 有边节点的前驱节点list
    pre_index = []
    for m in range(nl_num):
        if nl_active[m]:
            if m==0:
                pre_index.append([0])
            else:
                # 节点m有出节点，无入节点，则m的入节点为整个块的输入
                if sum(nl_connect[m - 1])==0:
                    pre_index.append([0])
                # 有入节点
                else:
                    tmp_index = []
                    for index,con in enumerate(nl_connect[m - 1]):
                        if con == 1:
                            tmp_index.append(index+1)
                    pre_index.append(tmp_index)
                    
        # 当这个节点没有被连接，pre_index 处补[]
        else:
            pre_index.append([])

    # 块最终输出边相连的节点list
    out_index = []
    for t in range(nl_num):
        pre_index_ = flatten(pre_index[t+1:])
        # t+1: 表示有1-3编号节点的真实索引
        if nl_active[t] and t+1 not in pre_index_:
            out_index.append(t + 1)
    # 当块中所有节点都没有边，则3个节点并行与输入节点相连，输出为3个节点值
    # 设定第一个
    if sum([1 for act in nl_active if act is not None]) == 0:
        nl_active = [True for _ in range(nl_num)]
        pre_index = [[0] for _ in range(nl_num)]
        out_index = [1,2,3]
        # out_index = [0]
    # print("nl_active,pre_index,out_index",nl_active,pre_index,out_index)
    return nl_active, pre_index, out_index


class UnetConnect(nn.Module):
    def __init__(self, indi, in_channels=3, features=48):
        super(UnetConnect, self).__init__()
        self.concat = Concat_layer(dim = 1)
        self.level_amount = indi.level_amount
        self.encoder_units = indi.encoder_units # indi 的encoder_units
        self.decoder_units = indi.decoder_units 

        self.encoder_conv_blocks = nn.ModuleList([None for _ in range(self.level_amount)]) 
        self.dsl_blocks = nn.ModuleList([None for _ in range(self.level_amount)])

        self.decoder_conv_blocks =  nn.ModuleList([None for _ in range(self.level_amount)])
        self.usl_blocks = nn.ModuleList([None for _ in range(self.level_amount)])
        # self.concat_bn_blocks = nn.ModuleList([None for _ in range(self.level_amount)])
        

        # self.first_conv = nn.Conv2d(in_channels=in_channels, out_channels= features, kernel_size=3, stride=1, padding=1)

        # encoder
        for level in range(self.level_amount):

            # dsl
            if level == 0:
                dsl_in_features = in_channels # in_channels
            else:
                # print("level",level)
                dsl_in_features = self.encoder_units[level-1].features
            # print("level",level)
            self.dsl_blocks[level] = DownBlock(dsl_in_features, self.encoder_units[level].features , self.encoder_units[level].downsample_type)
            self.encoder_conv_blocks[level] = ConvBlock(self.encoder_units[level].block_amount, self.encoder_units[level].features, self.encoder_units[level].features, self.encoder_units[level].conv_type)
           

        for level in range(self.level_amount-1, -1, -1):

            if level == self.level_amount-1:
                conv_in_features = self.encoder_units[level].features
            else:
                conv_in_features = self.encoder_units[level].features + self.decoder_units[level].features
                # self.concat_bn_blocks[level] = OPS['bn'](conv_in_features)
            

            self.decoder_conv_blocks[level] = ConvBlock(self.decoder_units[level].block_amount, conv_in_features, self.decoder_units[level].features, self.decoder_units[level].conv_type)
            self.usl_blocks[level] = UpBlock(self.decoder_units[level].features, self.decoder_units[level].features, self.decoder_units[level].upsample_type)
         
        
        self.conv_last = OPS['conv_block_last'](self.decoder_units[self.level_amount-1].features + in_channels, in_channels)
        self.sig = nn.Sigmoid()

    def forward(self, x):
      
        origin_x =  x
        # x = self.first_conv(x)
        encoder_conv_outputs = [None for _ in range(self.level_amount)] # 存放encoder的ConvBlock部分的结果，之后在decoder中进行残差连接

        # encoder
        for level in range(self.level_amount):
            x = self.dsl_blocks[level](x)
            x = self.encoder_conv_blocks[level](x)
            encoder_conv_outputs[level] = x
        
        # decoder
        for level in range(self.level_amount-1,-1,-1):
            if level != self.level_amount-1:
                x = self.concat([x, encoder_conv_outputs[level]]) # concat
                # x = self.concat_bn_blocks[level](x) # bn
            x = self.decoder_conv_blocks[level](x)
            x = self.usl_blocks[level](x)
        
        # last_conv
        x = self.concat([x, origin_x])
        x = self.conv_last(x)
        out = self.sig(x)

        return out
        