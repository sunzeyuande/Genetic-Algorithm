import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import random

matplotlib.rcParams['font.family'] = 'STSong'  # 字体为华文宋体

# 载入数据
site_name = []
site_coordinate = []
with open('11.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split('\n')[0]
        line = line.split(' ')
        site_name.append(line[0])
        site_coordinate.append([float(line[2]), float(line[1])])
site_coordinate = np.array(site_coordinate)

# 两两地点之间的距离矩阵
site_count = len(site_name)
Distance = np.zeros([site_count, site_count])
for i in range(site_count):
    for j in range(site_count):
        Distance[i][j] = math.sqrt(
            (site_coordinate[i][0] - site_coordinate[j][0]) ** 2 + (site_coordinate[i][1] - site_coordinate[j][1]) ** 2)

# 种群数
count = 200
# 进化次数
itter_time = 600
# 变异率
mutation_rate = 0.1

# 51个地点与数字对应的字典   地点1：0 .......
site_index_dict = {}
i = 0
for site in site_name:
    site_index_dict[site] = i
    i += 1

# 51个数字与地点对应的字典   0：地点1
index_site_dict = {}
i = 0
for site in site_name:
    index_site_dict[i] = site
    i += 1


# 获得起点
def get_origin(select_site):
    global site_index_dict
    origin = site_index_dict[select_site[0]]  # 将选择的地点中的第一个设为起点地点
    select_site_index = []
    for site in select_site:
        select_site_index.append(site_index_dict[site])  # 将选择的地点转换为地点对应的序号
    select_site_index.remove(origin)
    return origin, select_site_index


# 一个个体的总距离
def get_total_distance(x, origin):
    distance = 0
    distance += Distance[origin][x[0]]
    for i in range(len(x)):
        if i == len(x) - 1:
            distance += Distance[origin][x[i]]

            break
        else:
            distance += Distance[x[i]][x[i + 1]]
    return distance


# 初始化种群
def generate_population(select_site_index):
    population = []
    for i in range(count):
        # 随机生成个体
        x = select_site_index.copy()
        random.shuffle(x)  # 随机排序
        population.append(x)
    return population


# 自然选择    轮盘赌算法
def selection(population, origin):
    graded = [[get_total_distance(x, origin), x] for x in population]
    # 计算适应度
    fit_value = []  # 存储每个个体的适应度
    for i in range(len(graded)):
        fit_value.append(1 / graded[i][0] ** 15)
    # 适应度总和
    total_fit = 0
    for i in range(len(fit_value)):
        total_fit += fit_value[i]

    # 计算每个适应度占适应度总和的比例
    newfit_value = []  # 储存每个个体轮盘选择的概率
    for i in range(len(fit_value)):
        newfit_value.append(fit_value[i] / total_fit)

    # 计算累计概率
    t = 0
    for i in range(len(newfit_value)):
        t = t + newfit_value[i]
        newfit_value[i] = t

    # 生成随机数序列用于选择和比较
    ms = []  # 随机数序列
    for i in range(len(population)):
        ms.append(random.random())
    ms.sort()

    # 轮盘赌选择法
    i = 0
    j = 0
    parents = []
    while i < len(population):
        # 选择--累积概率大于随机概率
        if ms[i] < newfit_value[j]:
            if population[j] not in parents:
                parents.append(population[j])
            i = i + 1
        # 不选择--累积概率小于随机概率
        else:
            j = j + 1

    return parents


# 交叉繁殖
def crossover(parents):
    # 生成子代的个数,以此保证种群稳定
    child_count = count - len(parents)
    # 孩子列表
    children = []
    while len(children) < child_count:
        # 随机选择父母
        mother_index = random.randint(0, len(parents) - 1)
        father_index = random.randint(0, len(parents) - 1)
        if mother_index != father_index:
            mother = parents[mother_index]
            father = parents[father_index]

            # 随机选择交叉点
            left = random.randint(0, len(mother) - 2)
            right = random.randint(left + 1, len(mother) - 1)

            # 交叉片段
            gene1 = mother[left:right]
            gene2 = father[left:right]

            child1_c = mother[right:] + mother[:right]
            child2_c = father[right:] + father[:right]
            child1 = child1_c.copy()
            child2 = child2_c.copy()

            for o in gene2:
                child1_c.remove(o)

            for o in gene1:
                child2_c.remove(o)

            child1[left:right] = gene2
            child2[left:right] = gene1

            child1[right:] = child1_c[0:len(child1) - right]
            child1[:left] = child1_c[len(child1) - right:]

            child2[right:] = child2_c[0:len(child1) - right]
            child2[:left] = child2_c[len(child1) - right:]

            children.append(child1)
            children.append(child2)

    return children


# 变异    基因次序片段交换
def mutation(children):
    for i in range(len(children)):
        if random.random() < mutation_rate:
            child = children[i]
            u = random.randint(0, len(child) - 2)
            v = random.randint(u + 1, len(child) - 1)

            child_x = child[u + 1:v]
            child_x.reverse()
            child = child[0:u + 1] + child_x + child[v:]


# 得到最佳纯输出结果
def get_result(population, origin):
    graded = [[get_total_distance(x, origin), x] for x in population]
    graded = sorted(graded)
    return graded


# 画图
def draw(origin, result_path, distance):
    global site_coordinate
    global site_name
    # 34个地点散点图
    plt.scatter(site_coordinate[:, 0], site_coordinate[:, 1])

    site_name1 = []
    with open('22.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            site_name1.append(line)

    for i in range(51):
        plt.text(site_coordinate[i, 0], site_coordinate[i, 1], site_name1[i], fontsize="8")
    X = []
    Y = []
    X.append(site_coordinate[origin, 0])
    Y.append(site_coordinate[origin, 1])
    i = 0
    for index in result_path:
        X.append(site_coordinate[index, 0])
        Y.append(site_coordinate[index, 1])
        plt.plot(X, Y, '-')
        plt.text((X[0] + X[1]) / 2, (Y[0] + Y[1]) / 2, i, fontsize='small')

        plt.title("distance = " + str(distance))
        del (X[0])
        del (Y[0])
        i += 1
    X.append(site_coordinate[origin, 0])
    Y.append(site_coordinate[origin, 1])
    plt.text((X[0] + X[1]) / 2, (Y[0] + Y[1]) / 2, i, fontdict={"size": 12})  # 给这个线段表上序号
    plt.plot(X, Y, '-')
    # 起点特别标注
    plt.scatter(site_coordinate[origin, 0], site_coordinate[origin, 1], s=150)
