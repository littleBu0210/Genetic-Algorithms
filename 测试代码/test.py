import numpy as np
import time
import random
import matplotlib.pyplot as plt
import os

def read(data_file,i):
    """
    读取txt文件
    :param data_file:输入测试样例文件路径
    :param i:样例i
    :return: W：背包最大容量(int)；N：物品数量(int)；w：每件物品的重量(list);v：每件物品的价值(list)
    """
    W = 0
    N = 0
    w = []
    v = []
    with open(data_file,'r') as f:
        string = f.readlines()
        for j in range(len(string)):
            if string[j] == '(' + str(i) + ')' + ' \n':
                W = int(string[j+2].split(' ')[0])
                N = int(string[j+2].split(' ')[1])
                for k in range(1,N+1):
                    w.append(int(string[j+k+2].split(' ')[0]))
                    v.append(int(string[j+k+2].split(' ')[1]))
    return W, N, w, v

class MyGA(object):
    #类属性
    cross_p = 0.6
    variation_p = 0.2
    pop_num = 100               #种群内个体数目
    cross_num = int(pop_num/2)  #交叉的对数
    #精英解参数
    elite_num = 4               #精英解池的个数
    elite_th = 0                #精英解池的最低门限
    elite_min_index = 0         #精英解池 最小价值精英解的索引
    #自适应参数
    avg_fitness = 0
    max_fitness = 0

    #类方法
    def __init__(self,W,N,w,v):
        """
        初始化类的函数
        :param W: 背包最大承重
        :param N: 物品总数
        :param w: 每件物品的重量
        :param v: 每件物品的价值
        """
        self.bag_max_w = W
        self.N = N
        self.w = w
        self.v = v
        #迭代的代数采用十倍的物品数量
        self.generation = self.N*10
        #定义一个元组，便于表达数组大小
        self.shape =(self.pop_num,N)
        #初始化种群，0的概率为0.9，1的概率为0.1，防止重量过大适应度减少为0
        self.population = np.random.choice([0, 1], size=self.shape, p=[.9, .1])
        #初始化价值存储数组
        self.max_value_pre_gen = np.zeros(self.generation,dtype=np.int32)
        #初始化精英解池
        self.elite = np.zeros(shape=(self.elite_num,N),dtype=np.uint32)
        #初始化精英解池的价值
        self.elite_value = np.zeros(shape=self.elite_num,dtype=np.uint32)
        #记录每一轮精英解池的最大值
        self.max_value_elite = np.zeros(self.generation,dtype=np.int32)




    def fitnessF(self):
        """
        计算适应度的函数
        :param self.population  :所有个体
        :param self.v           :每件物品的价值
        :param self.w           :每件物品的重量
        :param self.bag_max_w   :背包最大承重
        :param self.value       :当前每个个体背包内的价值
        :param self.fitness     :当前每个个体的适应度
        """
        #计算每个个体的价值
        self.value = np.dot(self.population,self.v)
        #计算每个个体的所带物品的重量
        self.weight = self.population.dot(self.w)
        self.fitness =self.value
        #大于背包最大值的个体适应度清零
        for i in range(self.pop_num):
            if(self.weight[i]>=self.bag_max_w):
                self.fitness[i] = 0
        #计算平均适应度
        self.avg_fitness = np.average(self.fitness)
        self.max_fitness = np.max(self.fitness)



    def select(self):
        """
        用轮盘赌的方法确定子代个体
        :param self.fitness     :当前每个个体的适应度
        :param fitness_sum      :所有适应度的总和
        :param cumulative_p     :累计概率
        :param new_population   :新选出来的子代
        引入精英解来共同产生子代
        """
        #总和归一化
        fitness_sum = np.sum(self.fitness)
        select_p = self.fitness/fitness_sum
        #计算累计概率
        cumulative_p = np.zeros(shape=(self.pop_num,1),dtype=np.float64)
        cumulative_p[0,0] = select_p[0]
        for i in range(1,self.pop_num):
            cumulative_p[i] = cumulative_p[i-1] + select_p[i]
        #生成新的子代
        new_population = np.zeros_like(self.population)

        #子代个体生成的顺序
        order = random.sample(range(self.pop_num),self.pop_num)
        From_parent = order[0:self.pop_num-self.elite_num]
        From_elite = order[self.pop_num-self.elite_num:]
        for i in From_parent:
            rand_num = np.random.uniform()
            for j,cum_val in enumerate(cumulative_p):
                if(cum_val>rand_num):
                    break
            new_population[i,:] = self.population[j,:]

        for index,value in enumerate(From_elite):
            new_population[value,:] = self.elite[index,:]

        self.population = new_population

    def cross(self):
        """
        进行交叉的函数
        采用随机两个交叉位点 将中间段进行互换
        """
        for i in range(self.cross_num):
            #先判断要不要进行交叉
            rand_num = np.random.uniform()
            if(rand_num>self.cross_p):
                continue
            #产生两个随机的位置
            cross_pos = np.sort(random.sample(range(0, self.N), 2))
            section = self.population[2*i,cross_pos[0]:cross_pos[1]].copy()
            self.population[2*i,cross_pos[0]:cross_pos[1]] = self.population[2*i+1,cross_pos[0]:cross_pos[1]]
            self.population[2*i+1,cross_pos[0]:cross_pos[1]] = section

    def variation(self):
        '''
        进行变异的函数
        01翻转
        '''
        for i in range(self.pop_num):
            #先判断要不要进行变异
            rand_num = np.random.uniform()
            if(rand_num>self.variation_p):
                continue
            variation_pos = np.random.randint(self.N,dtype=np.uint8)
            self.population[i,variation_pos] = np.mod(self.population[i,variation_pos]+1,2)

    def record_best_individual(self,times):
        #保存最大价值
        self.max_value_pre_gen[times] = np.max(self.fitness)

        #保存精英解中最大值
        self.max_value_elite[times] =  np.max(self.elite_value)
        # #保存最佳个体
        # self.best_individual[times,:] = self.population[index,:]

    def elite_renew(self):
        for i in range(self.pop_num):
            if(self.fitness[i]>self.elite_th):
                #更新精英解池和精英解价值
                self.elite[self.elite_min_index,:] = self.population[i,:]
                self.elite_value[self.elite_min_index] = self.fitness[i]
                #更新精英解池最小值的索引
                index = np.argmin(self.elite_value)
                #更新精英解池的最低门限的索引
                self.elite_min_index = index
                #更新精英解池的最低门限
                self.elite_th = self.elite_value[index]
    def F_adapt(self,k,f_avg,f_max,x):
        return k if x<=f_avg else k*(f_max-x)/(f_max-f_avg)



def GA(W,N,w,v,data_select,res_value):
    """
    遗传算法解决0-1背包问题主函数
    :param W: 背包最大承重
    :param N: 物品总数
    :param w: 每件物品的重量
    :param v: 每件物品的价值
    :param save_fig_path: 样例i的收敛曲线存储路径
    :return: max_value:求解的放入背包的物品最大价值(int)；best_solu：放入背包的物品序号(list)
    """
    #-----------------请同学们完成遗传算法-----------
    ga = MyGA(W,N,w,v)

    for i in range(ga.generation):
        #计算适应度函数
        ga.fitnessF()
        #更新精英解池
        ga.elite_renew()
        #交叉
        ga.cross()
        #变异
        ga.variation()
        #计算适应度函数
        ga.fitnessF()
        #更新精英解池
        ga.elite_renew()
        # #保留最大的价值
        ga.record_best_individual(i)
        #选择
        ga.select()

    ga.fitnessF()
    max_value = np.max(ga.fitness)
    best_solu = ga.population[np.argmax(ga.fitness),:]

    if(max_value>res_value):
    #绘图
        plt.clf()
        plt.plot(ga.max_value_pre_gen)
        plt.savefig("测试结果/result"+str(data_select)+".png")
















    #-----------------以上由同学完成---------------



    return max_value,best_solu



if __name__ == '__main__':
    data_file = "实验代码/data.txt"

    select_data_all = [1,2,3,4,5]
    data_select = int(input("请输入读取的文件序号(只输入单个数字即可):"))
    times = np.uint64(input("请输入迭代的次数(只输入数字即可):"))
    if(data_select not in select_data_all):
        print("请输入 1 2 3 4 5 中的一个数字")
        exit(0)

    W,N,w,v = read(data_file,data_select)

    str1 = "测试结果/result"+str(data_select)+".txt"
    if(os.path.isfile(str1)):
        fileHandler  =  open  (str1,  "r",encoding ='utf-8')
        line  =  fileHandler.readlines()[5]
        res_value = int(line.split('：')[1])
    else:
        res_value = -1

    new_write = 0

    for j in range(times):
        #计时
        start_time = time.time()

        max_value,best_solu = GA(W,N,w,v,data_select,res_value)

        end_time = time.time()

        #判断结果准确性
        assert len(best_solu)==N,'物品的件数为%d，给出的方案长度应当与之匹配'%N
        assert best_solu.dot(v)==max_value,'最大价值与给出的方案不匹配'
        assert best_solu.dot(w)<W,'给出的方案超重'
        print("装入背包的最大价值：{}\n".format(max_value))

        #如果大于当前已知的值，则更新txt文档
        if(max_value>res_value):
            res_value = max_value
            res_solu = best_solu
            res_T = end_time-start_time
            #写入标志位
            new_write=1
            print("出现更大的值")
    if(new_write==1):
        with open("测试结果/result"+str(data_select)+".txt","w",encoding ='utf-8') as f:
            f.write("背包最大承重：{}\n".format(W))

            f.write("物品件数：{}\n".format(N))

            f.write("每件物品的重量：[")
            for i in range(N-1):
                f.write("{},".format(w[i]))
            f.write("{}]\n".format(w[N-1])) 

            f.write("每件物品的价值：[")
            for i in range(N-1):
                f.write("{},".format(v[i]))
            f.write("{}]\n".format(v[N-1])) 

            f.write("收敛曲线存储的文件名："+"result"+str(data_select)+".png"+"\n")

            f.write("装入背包的最大价值：{}\n".format(res_value))

            f.write("装入背包最大价值的最优物品组合：[")
            for i in range(N-1):
                f.write("{},".format(res_solu[i]))
            f.write("{}]\n".format(res_solu[N-1])) 

            f.write("测试用时：{:.6f}s\n".format(res_T))        


