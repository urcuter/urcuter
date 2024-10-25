# 开发者：卡卡不是是卡卡
# 开发时间： 2023年03月04日
import numpy as np
import matplotlib.pyplot as plt


class OneOLS_linear_regression:
    """一元线性回归的类"""

    def __init__(self, x, y):
        self.x = x
        self.y = y

        """判断 x 与 y 是否一致"""
        if len(x) != len(y):
            print('警告：x 与 y 数目 不一致！')
        """基本统计量"""
        self.n = len(x)
        self.x_mean = np.mean(x)
        self.y_mean = np.mean(y)
        self.x_std = np.std(x)
        self.y_std = np.std(y)
        corr = 0
        for i in range(len(x)):
            corr += (x[i] - self.x_mean) * (y[i] - self.y_mean)
        self.corr = corr / len(x)

        """代估计系数"""
        self.beta1 = None
        self.beta0 = None
        self.predict_list = []

    def statistics(self):
        """显示基本统计量"""
        print('n:        ' + str(self.n))
        print('x_mean:   ' + str(self.x_mean))
        print('x_std:    ' + str(self.x_std))
        print('y_mean:   ' + str(self.y_mean))
        print('y_std:    ' + str(self.y_std))
        print('x_y_corr: ' + str(self.corr))

    def fit(self):
        """模型拟合"""
        if len(self.x) == len(self.y):
            self.beta1 = self.corr / (self.x_std ** 2)
            self.beta0 = self.y_mean - self.beta1 * self.x_mean
        else:
            print('x 与 y 数目不相同， 无法进行OLS回归！')

    def predict(self):
        """模型预测结果"""
        for j in self.x:
            predict = self.beta0 + self.beta1 * j
            self.predict_list.append(predict)
        return self.predict_list

    def figure(self):
        plt.scatter(self.x, self.y, s=5, c='y')
        plt.plot(self.x, self.predict_list, c='b')
        plt.title('模型拟合情况')
        plt.xlabel('x')
        plt.ylabel('y/predict')
        plt.show()


if __name__ == '__main__':
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 8, 10, 12]
    fault = OneOLS_linear_regression(x, y)
    print(fault.beta1)
    print(fault.predict_list)
    fault.fit()
