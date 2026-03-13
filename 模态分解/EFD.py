def EFD(S, T=None, fs=None):
    """
    经验傅里叶分解实现
    :param S: 信号 1D numpy数组
    :param T: 对应的时间序列 要求时间间隔恒定
    :return:
    """

    import numpy as np
    from scipy.signal import find_peaks
    from matplotlib import pyplot as plt

    N = len(S)

    # 处理T为None的情况
    if T is None:
        if fs is not None:
            # 通过采样频率创建时间序列
            dt = 1.0 / fs
            T = np.arange(N) * dt
        else:
            # 默认采样频率为1
            T = np.arange(N)
            print(f"警告: T 为 None，已使用默认时间序列 T = [0, 1, 2, ..., {N - 1}]")

    # 检查时间序列长度是否匹配
    if len(T) != N:
        raise ValueError(f"时间序列长度 ({len(T)}) 与信号长度 ({N}) 不匹配")

    # 检查时间间隔是否恒定
    dt = np.diff(T)
    if not np.allclose(dt, dt[0], rtol=1e-10, atol=1e-14):
        raise ValueError("时间序列必须具有恒定的时间间隔")

    plt.rc('font', size=16)  # 设置字体大小
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    # 获取傅里叶频谱
    fft = np.fft.fft
    F = fft(S)
    freq = np.fft.fftfreq(len(T), d=T[1]-T[0])

    f = np.abs(F)
    idx = np.where(freq >= 0)[0]
    freq_positive = freq[idx]
    spectrum = f[idx]

    # 频域图像
    # plt.plot(freq, np.abs(F))
    # plt.title("FFT 频谱")
    # plt.xlabel("频率 (Hz)")
    # plt.ylabel("幅值")
    # plt.show()

    # 构建边界
    part_max, _ = find_peaks(spectrum)
    part_min, _ = find_peaks(-spectrum)
    boundary = np.concatenate([np.array([0]), part_min, part_max, np.array([len(freq_positive) - 1])])
    boundary = np.unique(boundary)
    boundary = np.sort(boundary)  # 是freq_positive的下标

    # 构建零相位滤波器组
    # u = []
    # for b in range(1, len(boundary)):
    #     down_bound = boundary[b - 1]
    #     up_bound = boundary[b]
    #
    #     f = lambda x: 1 if down_bound <= x <= up_bound else 0
    #     u.append(f)

    # 获取子频带信号，构建零相位滤波器，提取IMF
    IMFs = []

    freq_set = []  # 不同区间的频率
    spectrum_set = []  # 不同区间的频谱

    for b in range(1, len(boundary)):
        down_bound = boundary[b - 1]
        up_bound = boundary[b]

        filter_mask = np.zeros(N)
        filter_mask[N // 2 + down_bound : N // 2 + up_bound + 1] = 1
        filter_mask[N // 2 - up_bound : N // 2 - down_bound + 1] = 1

        if down_bound == 0:
            filter_mask[N // 2 + 1 : N // 2 + up_bound + 1] = 1

        F_band = F * filter_mask

        IMF = np.fft.ifft(F_band)
        IMFs.append(np.real(IMF))  # 出现了小数值的虚部

    sum_IMFs = np.sum(IMFs, axis=0)
    Res = S - sum_IMFs

    return IMFs, Res

if __name__ == '__main__':
    def verify_efd(S, T, IMFs):
        """
        验证 EFD 分解的正确性
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # 1. 检查重构误差
        reconstructed = np.sum(IMFs, axis=0)
        error = np.max(np.abs(S - reconstructed))
        print(f"最大重构误差: {error:.2e}")

        # 2. 检查能量守恒
        energy_original = np.sum(S ** 2)
        # print(IMFs)
        energy_imfs = np.sum([np.sum(imf ** 2) for imf in IMFs])
        print(f"原始信号能量: {energy_original:.6f}")
        print(f"IMF能量和: {energy_imfs:.6f}")
        print(f"能量误差: {abs(energy_original - energy_imfs):.2e}")

        # 3. 可视化
        fig, axes = plt.subplots(len(IMFs) + 2, 1, figsize=(12, 2 * (len(IMFs) + 2)))

        # 原始信号
        axes[0].plot(T, S)
        axes[0].set_title("原始信号")
        axes[0].set_ylabel("幅值")

        # IMFs
        for i, imf in enumerate(IMFs):
            axes[i + 1].plot(T, imf)
            axes[i + 1].set_title(f"IMF {i + 1}")
            axes[i + 1].set_ylabel("幅值")

        # 重构信号
        axes[-1].plot(T, reconstructed, 'r-', label='重构')
        axes[-1].plot(T, S, 'b--', alpha=0.5, label='原始')
        axes[-1].set_title("重构信号 vs 原始信号")
        axes[-1].set_ylabel("幅值")
        axes[-1].set_xlabel("时间")
        axes[-1].legend()

        plt.tight_layout()
        plt.show()

        return error < 1e-10

    # 生成测试信号
    import numpy as np

    t = np.linspace(0, 1, 1000)  # 1000个点
    f1, f2, f3 = 5, 20, 50
    S = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t) + 0.2 * np.sin(2 * np.pi * f3 * t)

    # 应用 EFD
    IMFs = EFD(S, t)

    # 验证
    verify_efd(S, t, IMFs)