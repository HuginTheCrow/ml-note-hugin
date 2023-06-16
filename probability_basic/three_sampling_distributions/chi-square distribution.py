import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def chi2_distribution(df=1):
    """
    卡方分布，在实际的定义中只有一个参数df，即定义中的n
    :param df: 自由度，也就是该分布中独立变量的个数
    :return:
    """

    fig, ax = plt.subplots(1, 1)

    # 直接传入参数, Display the probability density function (pdf)
    x = np.linspace(stats.chi2.ppf(0.001, df),
                    stats.chi2.ppf(0.999, df), 200)
    ax.plot(x, stats.chi2.pdf(x, df), 'r-',
            lw=5, alpha=0.6, label=r'$\chi^2$ pdf')

    # 从冻结的均匀分布取值, Freeze the distribution and display the frozen pdf
    chi2_dis = stats.chi2(df=df)
   