import numpy as np
from pyspark import SparkConf, SparkContext
from pprint import pprint

def Dv1(step_h,Dimension):
    C = 1/(step_h*step_h)
    B = 0.5/step_h

    return lambda x:((x[0],x[1],x[2],x[3]),C *\
                    C * (x[4][3,1,2,2] + x[4][3,3,2,2] - 2*x[4][3,2,2,2]) *\
                    C * (x[4][3,2,1,2] + x[4][3,2,3,2] - 2*x[4][3,2,2,2]) *\
                    C * (x[4][3,2,2,1] + x[4][3,2,2,3] - 2*x[4][3,2,2,2]) + C -\
                    (x[0]+x[1]+x[2]+x[3]+8)/(B*x[4][3,2,2,2]*x[4][3,2,2,2]) +\
                    C * (x[4][1,3,2,2] + x[4][3,3,2,2] - 2*x[4][2,3,2,2]) *\
                    C *\
                    C * (x[4][2,3,1,2] + x[4][2,3,3,2] - 2*x[4][2,3,2,2]) *\
                    C * (x[4][2,3,2,1] + x[4][2,3,2,3] - 2*x[4][2,3,2,2]) + C -\
                    (x[0]+x[1]+x[2]+x[3]+8)/(B*x[4][2,3,2,2]*x[4][2,3,2,2]) +\
                    C * (x[4][1,2,3,2] + x[4][3,2,3,2] - 2*x[4][2,2,3,2]) *\
                    C * (x[4][2,1,3,2] + x[4][2,3,3,2] - 2*x[4][2,2,3,2]) *\
                    C *\
                    C * (x[4][2,2,3,1] + x[4][2,2,3,3] - 2*x[4][2,2,3,2]) + C -\
                    (x[0]+x[1]+x[2]+x[3]+8)/(B*x[4][2,2,3,2]*x[4][2,2,3,2]) +\
                    C * (x[4][1,2,2,3] + x[4][3,2,2,3] - 2*x[4][2,2,2,3]) *\
                    C * (x[4][2,1,2,3] + x[4][2,3,2,3] - 2*x[4][2,2,2,3]) *\
                    C * (x[4][2,2,1,3] + x[4][2,2,3,3] - 2*x[4][2,2,2,3]) *\
                    C + C -\
                    (x[0]+x[1]+x[2]+x[3]+8)/(B*x[4][2,2,2,3]*x[4][2,2,2,3]))

def Dv3(step_h,Dimension):
    C = 1/(step_h*step_h)
    B = 0.5/step_h

    return lambda x:((x[0],x[1],x[2],x[3]),C *\
                    C * (x[4][1,1,2,2] + x[4][1,3,2,2] - 2*x[4][1,2,2,2]) *\
                    C * (x[4][1,2,1,2] + x[4][1,2,3,2] - 2*x[4][1,2,2,2]) *\
                    C * (x[4][1,2,2,1] + x[4][1,2,2,3] - 2*x[4][1,2,2,2]) + C +\
                    (x[0]+x[1]+x[2]+x[3]+8)/(B*x[4][1,2,2,2]*x[4][1,2,2,2]) +\
                    C * (x[4][1,1,2,2] + x[4][3,1,2,2] - 2*x[4][2,1,2,2]) *\
                    C *\
                    C * (x[4][2,1,1,2] + x[4][2,1,3,2] - 2*x[4][2,1,2,2]) *\
                    C * (x[4][2,1,2,1] + x[4][2,1,2,3] - 2*x[4][2,1,2,2]) + C +\
                    (x[0]+x[1]+x[2]+x[3]+8)/(B*x[4][2,1,2,2]*x[4][2,1,2,2]) +\
                    C * (x[4][1,2,1,2] + x[4][3,2,1,2] - 2*x[4][2,2,1,2]) *\
                    C * (x[4][2,1,1,2] + x[4][2,3,1,2] - 2*x[4][2,2,1,2]) *\
                    C *\
                    C * (x[4][2,2,1,1] + x[4][2,2,1,3] - 2*x[4][2,2,1,2]) + C +\
                    (x[0]+x[1]+x[2]+x[3]+8)/(B*x[4][2,2,1,2]*x[4][2,2,1,2]) +\
                    C * (x[4][1,2,2,1] + x[4][3,2,2,1] - 2*x[4][2,2,2,1]) *\
                    C * (x[4][2,1,2,1] + x[4][2,3,2,1] - 2*x[4][2,2,2,1]) *\
                    C * (x[4][2,2,1,1] + x[4][2,2,3,1] - 2*x[4][2,2,2,1]) *\
                    C + C +\
                    (x[0]+x[1]+x[2]+x[3]+8)/(B*x[4][2,2,2,1]*x[4][2,2,2,1]))

def Dv2(step_h,Dimension):
    C = 1/(step_h*step_h)

    return lambda x:((x[0],x[1],x[2],x[3]),
                    C * (-2) *\
                    C * (x[4][2,1,2,2] + x[4][2,3,2,2] - 2*x[4][2,2,2,2]) *\
                    C * (x[4][2,2,1,2] + x[4][2,2,3,2] - 2*x[4][2,2,2,2]) *\
                    C * (x[4][2,2,2,1] + x[4][2,2,2,3] - 2*x[4][2,2,2,2]) +\
                    C * (x[4][1,2,2,2] + x[4][3,2,2,2] - 2*x[4][2,2,2,2]) *\
                    C * (-2) *\
                    C * (x[4][2,2,1,2] + x[4][2,2,3,2] - 2*x[4][2,2,2,2]) *\
                    C * (x[4][2,2,2,1] + x[4][2,2,2,3] - 2*x[4][2,2,2,2]) +\
                    C * (x[4][1,2,2,2] + x[4][3,2,2,2] - 2*x[4][2,2,2,2]) *\
                    C * (x[4][2,1,2,2] + x[4][2,3,2,2] - 2*x[4][2,2,2,2]) *\
                    C * (-2) *\
                    C * (x[4][2,2,2,1] + x[4][2,2,2,3] - 2*x[4][2,2,2,2]) +\
                    C * (x[4][1,2,2,2] + x[4][3,2,2,2] - 2*x[4][2,2,2,2]) *\
                    C * (x[4][2,1,2,2] + x[4][2,3,2,2] - 2*x[4][2,2,2,2]) *\
                    C * (x[4][2,2,1,2] + x[4][2,2,3,2] - 2*x[4][2,2,2,2]) *\
                    C * (-2) - 2*C*Dimension)

def loss(step_h):
    C = 1/(step_h*step_h)
    B = 0.5/step_h

    return lambda x: C * (x[4][1,2,2,2] + x[4][3,2,2,2] - 2*x[4][2,2,2,2]) *\
                     C * (x[4][2,1,2,2] + x[4][2,3,2,2] - 2*x[4][2,2,2,2]) *\
                     C * (x[4][2,2,1,2] + x[4][2,2,3,2] - 2*x[4][2,2,2,2]) *\
                     C * (x[4][2,2,2,1] + x[4][2,2,2,3] - 2*x[4][2,2,2,2]) +\
                     C * (x[4][1,2,2,2] + x[4][3,2,2,2] - 2*x[4][2,2,2,2]) +\
                     C * (x[4][2,1,2,2] + x[4][2,3,2,2] - 2*x[4][2,2,2,2]) +\
                     C * (x[4][2,2,1,2] + x[4][2,2,3,2] - 2*x[4][2,2,2,2]) +\
                     C * (x[4][2,2,2,1] + x[4][2,2,2,3] - 2*x[4][2,2,2,2]) -\
                     (x[0]+x[1]+x[2]+x[3]+8)/(B*(-x[4][1,2,2,2] + x[4][3,2,2,2] \
                                                 -x[4][2,1,2,2] + x[4][2,3,2,2] \
                                                 -x[4][2,2,1,2] + x[4][2,2,3,2] \
                                                 -x[4][2,2,2,1] + x[4][2,2,2,3]))

def getData(U,D_length):
    data = []
    for x1 in range(D_length):
        for x2 in range(D_length):
            for x3 in range(D_length):
                for x4 in range(D_length):
                    data.append((x1,x2,x3,x4,U[x1:x1+4,x2:x2+4,x3:x3+4,x4:x4+4]))

    return data

def getData_2(U,D_length):
    data = []
    for x1 in range(D_length):
        for x2 in range(D_length):
            data.append((x1,x2,U[x1:x1+4,x2:x2+4]))

    return data

def loss_2(step_h):
    C = 1/(step_h*step_h)
    B = 0.5/step_h

    return lambda x: C * (x[2][1,2] + x[2][3,2] - 2*x[2][2,2]) *\
                     C * (x[2][2,1] + x[2][2,3] - 2*x[2][2,2]) +\
                     C * (x[2][1,2] + x[2][3,2] - 2*x[2][2,2]) +\
                     C * (x[2][2,1] + x[2][2,3] - 2*x[2][2,2]) # -\
                     # (x[0]+x[1]+4)/(B*(-x[2][1,2] + x[2][3,2] -x[2][2,1] + x[2][2,3]))

def Dv2_2(step_h,Dimension=2):
    C = 1/(step_h*step_h)

    return lambda x:((x[0],x[1]),
                    C * (-2) *\
                    C * (x[Dimension][2,1] + x[Dimension][2,3] - 2*x[Dimension][2,2]) +\
                    C * (x[Dimension][1,2] + x[Dimension][3,2] - 2*x[Dimension][2,2]) *\
                    C * (-2) - 2*C*Dimension)

def Dv3_2(step_h,Dimension):
    C = 1/(step_h*step_h)
    B = 0.5/step_h

    # return lambda x:((x[0],x[1]),C *\
    #                 C * (x[Dimension][1,1] + x[Dimension][1,3] - 2*x[Dimension][1,2]) + C +\
    #                 (x[0]+x[1]+4)/(B*x[Dimension][1,2]*x[Dimension][1,2]) +\
    #                 C * (x[Dimension][1,1] + x[Dimension][3,1] - 2*x[Dimension][2,1]) *\
    #                 C + C +\
    #                 (x[0]+x[1]+4)/(B*x[Dimension][2,1]*x[Dimension][2,1]))
    return lambda x:((x[0],x[1]),C *\
                C * (x[Dimension][1,1] + x[Dimension][1,3] - 2*x[Dimension][1,2]) + C +\
                C * (x[Dimension][1,1] + x[Dimension][3,1] - 2*x[Dimension][2,1]) *\
                C + C)

def Dv1_2(step_h,Dimension):
    C = 1/(step_h*step_h)
    B = 0.5/step_h

    # return lambda x:((x[0],x[1]),C *\
    #                 C * (x[Dimension][3,1] + x[Dimension][3,3] - 2*x[Dimension][3,2]) + C -\
    #                 (x[0]+x[1]+4)/(B*x[Dimension][3,2]*x[Dimension][3,2]) +\
    #                 C * (x[Dimension][1,3] + x[Dimension][3,3] - 2*x[Dimension][2,3]) *\
    #                 C + C -\
    #                 (x[0]+x[1]+4)/(B*x[Dimension][2,3]*x[Dimension][2,3]))
    return lambda x:((x[0],x[1]),C *\
                C * (x[Dimension][3,1] + x[Dimension][3,3] - 2*x[Dimension][3,2]) + C +\
                C * (x[Dimension][1,3] + x[Dimension][3,3] - 2*x[Dimension][2,3]) *\
                C + C)


appname = "MAPDE"
master = "local"
myconf = SparkConf().setAppName(appname).setMaster(master)
sc = SparkContext(conf=myconf)
SolutionMax = 100
SolutionMin = 0
Step_h = 1
D_length = int((SolutionMax - SolutionMin)/Step_h)*100
learning_rate = 0.1


Dimension = 2
dataInit_Solution = np.random.rand(D_length+4,D_length+4)
# pprint(dataInit_Solution)

for iter in range(30000):
    spark_U = sc.parallelize(getData_2(dataInit_Solution,D_length),20)

    Drive2 = spark_U.map(Dv2_2(Step_h,Dimension))
    Drive1 = spark_U.map(Dv1_2(Step_h,Dimension))
    Drive3 = spark_U.map(Dv3_2(Step_h,Dimension))
    AllDrive = Drive2.join(Drive3).mapValues(lambda x: x[0]+x[1]).join(Drive1).mapValues(lambda x: x[0]+x[1])
    # pprint(AllDrive.collect())
    # pprint(AllDrive.filter(lambda x:x[0] == (0,0)).collect())
    AllDrive = AllDrive.sortBy(lambda x:x[0]).values().collect()
    # pprint(AllDrive[0:5])
    if iter % 2 == 0:
        U_loss = spark_U.map(loss_2(Step_h)).reduce(lambda x,y:abs(x)+abs(y))
        print('Iter:',iter,'PDE Loss:',U_loss,max(AllDrive))

    if max(AllDrive) < 0.02:
        learning_rate = 0.0001
    elif max(AllDrive) < 0.1:
        learning_rate = 0.001
    elif max(AllDrive) < 0.5:
        learning_rate = 0.01
    # pprint(AllDrive[0:10])
    for x1 in range(D_length):
        for x2 in range(D_length):
            index = x1 *D_length + x2
            # tempDrive = AllDrive.filter(lambda x:x[0] == (x1,x2)).values().collect()[0]
            # # pprint(tempDrive)

            dataInit_Solution[x1+2,x2+2] = dataInit_Solution[x1+2,x2+2] - learning_rate*AllDrive[index]/max(AllDrive)




# Dimension = 4
# SolutionMax = 10
# SolutionMin = 0
# Step_h = 1
# D_length = int((SolutionMax - SolutionMin)/Step_h)
#
# dataInit_Solution = np.random.rand(D_length+4,D_length+4,D_length+4,D_length+4)
#
#
# for iter in range(300):
#     spark_U = sc.parallelize(getData(dataInit_Solution,D_length),40)
#     U_loss = spark_U.map(loss(Step_h)).reduce(lambda x,y:abs(x)+abs(y))
#     if iter % 5 == 0:
#         print('Iter:',iter,'PDE Loss:',U_loss)
#
#     Drive2 = spark_U.map(Dv2(Step_h,Dimension))
#     Drive1 = spark_U.map(Dv1(Step_h,Dimension))
#     Drive3 = spark_U.map(Dv3(Step_h,Dimension))
#     AllDrive = Drive2.join(Drive3).mapValues(lambda x: x[0]+x[1]).join(Drive1).mapValues(lambda x: x[0]+x[1])
#     # pprint(AllDrive.take(5))
#     AllDrive = AllDrive.values().collect()
#
#     for x1 in range(D_length):
#         for x2 in range(D_length):
#             for x3 in range(D_length):
#                 for x4 in range(D_length):
#                     index = x1 *D_length*D_length*D_length + x2*D_length*D_length + x3 *D_length + x4
#                     dataInit_Solution[x1+2,x2+2,x3+2,x4+2] = dataInit_Solution[x1+2,x2+2,x3+2,x4+2] - 0.0000000000000001*AllDrive[index]




