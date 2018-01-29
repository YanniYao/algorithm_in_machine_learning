
# coding: utf-8

# In[46]:


from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

def file2matrix(fileName):
    file = open(fileName)
    arrayLines = file.readlines()    #返回一个列表，一行为列表中的一个元素
    numberOfLines = len(arrayLines)
    returnMat = zeros((numberOfLines, 3))    #训练数据矩阵(numberOfLines, 3)表示用零所填充的数据的形状，为zeros()的第一个入口参数
    classLabelVector = []    #对应的标签向量
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFormLine = line.split("\t")
        returnMat[index, :] = listFormLine[0:3]   #[index, :]将第index行用等号右边的内容填充，[:, index]填充index列
        classLabelVector.append(int(listFormLine[-1]))
        #classLabelVector.append(listFormLine[-1])
        index += 1
    return returnMat, classLabelVector

# 归一化特征值（约会元素中，第一个值基数过大，对计算结果的影响过大，而我们假定三个特征值的权重相同，即三个特征同等重要）
# 利用公式：newValue = (oldValue-min)/(max-min)，进行规约化处理
def autoNorm(dataSet):
    minValue = dataSet.min(0)    #0:从列中选取最小值
    maxValue = dataSet.max(0)
    ranges = maxValue - minValue
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 因为需要对每个元素进行处理，所以使用tile进行整个dataSet上的最大最小或差值的重复
    normDataSet = dataSet - tile(minValue, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minValue

# 执行kNN算法
def classify(intX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]    #shape函数是numpy.core.fromnumeric中的函数，它的功能是查看矩阵或者数组的维数
    diffMat = tile(intX, (dataSetSize, 1)) - dataSet    #tile(A, B)，将A按B的格式要求进行重复，B可以是int(此时在列上重复，行默认重复一次)
                                                       #若B为元祖（a,b），则在行上重复a次，列上重复b次
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)    #axis＝0表示按列相加，axis＝1表示按照行相加
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()    #得到数组值从小到大的索引值
    #print(sortedDistIndicies)
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1    #get()返回指定键的值， 如果指定键的值不存在时，返回默认值值（此处为0）
                                                                    #此处即为记录每个入选标签在所有入选标签中出现的次数
    # 按照label出现的次数进行倒序排列，得到最大的那个label作为判断结果
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)    #items() 函数以列表返回可遍历的(键,值)元组
    return sortedClassCount[0][0]

# 将数据集的一部分抽出来得到分类的误差大小
def datingClassTest(fileName):
    choosePer = 0.1    #定义选取为测试数据的百分率
    datingMat, datingLabels = file2matrix(fileName)
    normMat = autoNorm(datingMat)[0]    #进行特征值规约化处理
    m = normMat.shape[0]    #获取数据集的行数
    testNum = int(m * choosePer)    #真正用于测试的数据量
    errCount = 0.0
    for i in range(testNum):
        classifyResult = classify(normMat[i, :], normMat[testNum:m, :], datingLabels[testNum:m], 3)
        print("the classfier came back with : %d , the real answer is : %d " % (i, classifyResult))
        if classifyResult != datingLabels[i]:
            errCount += 1
    print("the total error rate is : %f" % (errCount/float(testNum)))

#实际使用这个分类器
def classifyPerson(fileName):
    resultList = ['not at all', 'in small does', 'in large does']
    percentTats = float(input("percentage of time spent playing video games: "))
    ffMiles = float(input("frequent flier miles earned per year: "))
    iceCream = float(input("liters of ice cream consumed per year: "))
    datingMat, datingLabels = file2matrix(fileName)
    normMat, ranges, minValue = autoNorm(datingMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifyResult = classify((inArr - minValue)/ranges, normMat, datingLabels, 3)
    print("you will probably like this person: ", resultList[classifyResult-1])
    
if __name__ == '__main__':
    fileName = "datingTestSet.txt"
    fileName1 = "datingTestSet1.txt"
    '''
    datingMat, labels = file2matrix(fileName)
    fig = plt.figure()    #得到一张图
    ax = fig.add_subplot(111)    #第一个子图（最后一个1）子图总行数（第一个1）子图总列数（第二个1）
    #第一个15*array(labels)表示按照labels的元素值分别乘以（绘图点）基础大小15，得到最终大小不一的坐标点
    #第二个15*array(labels)表示按照labels的元素值分别乘以（绘图点）基础数值15，得到颜色不一的坐标点
    ax.scatter(datingMat[:, 0], datingMat[:, 1], 15 * array(labels), 15 * array(labels))    #使用每一行的第二列和第三列作为横纵坐标
    plt.show()
    '''
    #datingClassTest(fileName)
    classifyPerson(fileName)

