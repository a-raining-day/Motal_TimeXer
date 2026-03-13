import numpy as np
import antropy as ant
from 模态分解.EFD import EFD
from PyEMD import CEEMDAN

class CEEFD:
    """
    CEEFD模态分解
    """

    def __init__ \
        (
            self,
            trials=10,  # 集成次数: 整数 ≥ 1, 默认100
            noise_width=0.05,  # 噪声幅度: 浮点数, 通常0.05-0.3
            noise_seed=42,  # 随机种子: 整数或None
            spline_kind='cubic',  # 样条类型: 同EMD
            nbsym=2,  # 边界对称点数: 同EMD
            extrema_detection='parabol',  # 同EMD
            parallel=False,  # 是否并行计算: bool
            processes=None,  # 进程数: None=自动, 整数≥1
            random_state=42,  # 随机状态
            noise_scale=1.0,  # 噪声尺度因子
            noise_kind='normal',  # 噪声类型: 'normal', 'uniform'
            range_thr=0.01,  # 停止阈值
            total_power_thr=0.005
        ):

        self.trials = trials
        self.noise_width = noise_width
        self.noise_seed = noise_seed
        self.spline_kind = spline_kind
        self.nbsym = nbsym
        self.extrema_detection = extrema_detection
        self.parallel = parallel
        self.processes = processes
        self.random_state = random_state
        self.noise_scale = noise_scale
        self.noise_kind = noise_kind
        self.range_thr = range_thr
        self.total_power_thr = total_power_thr

    def give_ceemdan(self):
        return CEEMDAN \
            (
                trials=self.trials,
                noise_width=self.noise_width,
                noise_seed=self.noise_seed,
                spline_kind=self.spline_kind,
                nbsym=self.nbsym,
                extrema_detection=self.extrema_detection,
                parallel=self.parallel,
                processes=self.processes,
                random_state=self.random_state,
                noise_scale=self.noise_scale,
                noise_kind=self.noise_kind,
                range_thr=self.range_thr,
                total_power_thr=self.total_power_thr
            )

    def ceemdan(self, S, T=None, max_imf=-1):
        CEEMDAN = self.give_ceemdan()
        IMF_Residue = CEEMDAN.ceemdan(S, T, max_imf)

        IMFs = IMF_Residue[:-1, :]  # shape [n_imfs, len(S)]
        Res = IMF_Residue[-1, :]

        return IMFs, Res

    # 在 CEEFD.py 的 ceefd 方法中
    def ceefd(self, S, T=None, max_imf=-1):
        """
        返回: other_IMFs (ndarray), IMF_ (List[np.ndarray]), Res (ndarray), Res_ (ndarray)

        """
        CEEMDAN = self.give_ceemdan()
        IMF_Residue = CEEMDAN.ceemdan(S, T, max_imf)

        IMFs = IMF_Residue[:-1, :]  # shape [n_imfs, len(S)]
        Res = IMF_Residue[-1, :]

        # 计算每个 IMF 的样本熵
        Entropy = [ant.sample_entropy(IMF) for IMF in IMFs]
        max_entropy_mask = np.argmax(Entropy)
        maxIMF = IMFs[max_entropy_mask]

        # 对最大熵 IMF 进行 EFD 分解
        if T is None:
            T_efd = np.arange(len(maxIMF))  # 避免 EFD 内部警告
        else:
            T_efd = T
        IMF_, Res_ = EFD(maxIMF, T_efd)

        # 提取 other_IMFs（除去最大熵的 IMF）
        other_IMFs_list = [IMF for i, IMF in enumerate(IMFs) if i != max_entropy_mask]
        if other_IMFs_list:
            other_IMFs = np.stack(other_IMFs_list)
        else:
            other_IMFs = None

        return other_IMFs, IMF_, Res, Res_