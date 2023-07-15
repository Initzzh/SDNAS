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
    if sum(connect_gene) >= 1:
        nl_active[nl_num-1] = True
    
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
            pre_index.append([])

    # 块最终输出边相连的节点
    out_index = []
    for nl_index in range(nl_num):
        # 将2，3号节点的前驱展开
        pre_index_ = flatten(pre_index[nl_index+1:])
        
        if nl_active[nl_index] and nl_index+1 not in pre_index_:
            # nl激活，且不是其他节点的前驱，作为block的输出
            out_index.append(nl_index+1)
    # 当connect_gene
    if sum(connect_gene) == 0:
        # [0,0,0]
        nl_active = [True,None,None]
        pre_index = [[0],[],[]]
        out_index = [1]
    print("connect_arcitecture", nl_active, pre_index,out_index)
    return nl_active, pre_index, out_index
 


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
    print("connect_arcitecture", nl_active, pre_index,out_index)
    return nl_active, pre_index, out_index
 

def check_active3(node_num, connect_gene):
    active = [None for _ in range(node_num)]
    node_connect = []
    j = 1
    i = 0
    for _ in range(node_num - 1):
        node_connect.append(connect_gene[i:i + j])
        i = i + j
        j += 1
    for p, node in enumerate(node_connect):
        if p != node_num - 2:
            if sum(node) >= 1:
                active[p + 1] = True
    for k in range(node_num):
        for node in node_connect:
            if k < len(node) and k != node_num - 1:
                if node[k] == 1:
                    active[k] = True

            elif k == node_num - 1:
                if sum(node) >= 1:
                    active[k] = True

    pre_index = [None for _ in range(node_num)]
    for m in range(node_num):
        if active[m]:
            if m == 0:
                pre_index[m] = [m]
            else:
                p_index = []
                if sum(node_connect[m - 1]) == 0:
                    pre_index[m] = [0]
                else:
                    for index, con in enumerate(node_connect[m - 1]):
                        if con == 1:
                            p_index.append(index + 1)
                    if len(p_index) > 0:
                        pre_index[m] = p_index
    out_index = []
    for t in range(node_num):
        pre_index_ = flatten(pre_index[t + 1:])
        if active[t] and t + 1 not in pre_index_:
            out_index.append(t + 1)
    if sum([1 for act in active if act is not None]) == 0:
        out_index = [0]
    print("connect_arcitecture", active, pre_index,out_index)
    return active, pre_index, out_index


# check_active2(3, [0,0,1])  
# check_active(3, [0,0,1])  
# check_active3(3, [0,0,1])  



connect_type = 7 
val = connect_type
connect_gene = [0,0,0] # 010,001效果一样
index = len(connect_gene) - 1
while val > 0:
    rem = val % 2
    connect_gene[index] = rem
    index -= 1
    val = val // 2

print(connect_gene)








    




