import numpy as np
import pandas as pd
import MLKNN_index
import matplotlib.pyplot as plt
from skmultilearn.adapt import MLkNN
from sklearn import preprocessing
from sklearn.model_selection import KFold

def normal(data, interval_inf=0, interval_max=1):

    normall = preprocessing.MinMaxScaler((interval_inf, interval_max))
    data_normall = normall.fit_transform(data)
    return data_normall

class HSR_MLFS:
    def __init__(self, alpha=0.9, beta=0.5, delta_step=0.1, lammbda=0.5, sigma=1, delta=0.1):
        self.alpha = alpha
        self.beta = beta
        self.delta_step = delta_step
        self.X_train = None
        self.y_train = None
        self.judge = None
        self.lammbda = lammbda
        self.sigma = sigma
        self.judge_sparse = None
        self.judge_lower = None
        self.delta = delta
        self.ablation = None

    def fit(self, X_train, y_train, ablation=0, sparse=0):
        self.judge_sparse = sparse
        X1 = normal(np.array(X_train))
        self.X_train = np.array(X1, dtype=np.float16)
        self.y_train = y_train
        if np.mean(y_train) < 0.1:
            self.judge_lower = 1
        else:
            self.judge_lower = 0
        self.ablation = ablation

    def _calculate_distance_matrix(self):
        """
        :return: The distance matrix
        """
        data1 = np.array(self.X_train, dtype=np.float16)
        data2 = np.array(self.X_train, dtype=np.float16)
        m = len(data1)
        l = len(data1[0])
        if l != len(data2[0]):
            return False
        dis1 = np.sum(np.square(data1), axis=1)
        dis2 = np.sum(np.square(data2), axis=1)
        dis = np.add(np.add(-2 * np.dot(data1, data2.T), dis1.reshape(m, 1)), dis2)
        dis[dis < 0] = 0
        np.fill_diagonal(dis, 0)
        return np.sqrt(dis)

    def _PLNC3(self):
        n, o = self.y_train.shape
        dis = self._calculate_distance_matrix()
        np.fill_diagonal(dis, -1)
        dis_index = np.argsort(dis, axis=1)
        delta_num = int(0.9 / self.delta_step)
        PLNC_dic = {}
        label_set = set(np.arange(0, o, 1))
        x_set = set(np.arange(0, n, 1))
        label_cover_set = set()
        x_cover_set = set()
        
        label_have = np.zeros(o, dtype=np.float16)
        label_dict = {}
        for i in range(o):
            label_dict[i] = []

        for i in range(delta_num):
            delta = 0.1 + i * self.delta_step
            neighbor_num = np.sum(dis <= delta, axis=1)
            for j in range(n):
                neighbor = dis_index[j, :neighbor_num[j]]
                judge = np.sum(self.y_train[neighbor], axis=0) / len(neighbor)
                PL = np.where(judge >= self.alpha)[0]
                if len(PL) > 0:
                    if tuple(PL) in PLNC_dic:
                        PLNC_dic[tuple(PL)].append(neighbor)
                    else:
                        PLNC_dic[tuple(PL)] = [neighbor]
                    label_cover_set = label_cover_set | set(PL)
                    x_cover_set = x_cover_set | set(neighbor)
                
                index = np.where(judge >= label_have)[0]
                for k in index:
                    if judge[k] == label_have[k]:
                        label_dict[k].append(neighbor)
                    else:
                        label_dict[k] = [neighbor]
                        label_have[k] = judge[k]

        # Cover All labels
        label_remain_set = label_set - label_cover_set
        if len(label_remain_set) > 0:
            for i in label_remain_set:
                if label_have[i] >= self.beta:
                    current_neighbor = label_dict[i][0]
                    PLNC_dic[tuple([i])] = [current_neighbor]

        # Cover all samples 
        x_remain_set = x_set - x_cover_set
        label_have1 = np.zeros(len(x_remain_set), dtype=np.float16)
        x_dic = []
        for i in range(delta_num):
            delta = 0.1 + i * self.delta_step
            neighbor_num = np.sum(dis <= delta, axis=1)
            for index, j in enumerate(x_remain_set):
                x_dic.append([])
                if neighbor_num[j] <= 1:
                    break
                current_neighbor = dis_index[j, :neighbor_num[j]]
                judge = np.sum(self.y_train[current_neighbor], axis=0) / len(current_neighbor)
                maxl = np.max(judge)
                PL2 = np.where(judge == maxl)[0]
                if maxl > label_have1[index]:
                    x_dic[index] = [[PL2, current_neighbor]]
                    label_have1[index] = maxl

        compare = x_remain_set.copy()
        label_have2 = label_have1.copy()
        index1 = np.argsort(label_have2)[::-1]
        i = 0
        while len(compare) > 0 and i < len(index1):
            j = index1[i]
            if label_have2[j] <= self.beta:
                break
            current_x = set(x_dic[j][0][1])
            xxx = current_x & compare
            if len(xxx) == 0:
                i = i + 1
                continue
            else:
                compare = compare - current_x
                if tuple(x_dic[j][0][0]) in PLNC_dic:
                    PLNC_dic[tuple(x_dic[j][0][0])].append(x_dic[j][0][1])
                else:
                    PLNC_dic[tuple(x_dic[j][0][0])] = [x_dic[j][0][1]]
                i = i + 1

        return PLNC_dic

    def _PLNC1(self):
        n, o = self.y_train.shape
        dis = self._calculate_distance_matrix()
        np.fill_diagonal(dis, -1)
        dis_index = np.argsort(dis, axis=1)
        delta_num = int(0.9 / self.delta_step)
        PLNC_dic = {}
        label_set = set(np.arange(0, o, 1))
        x_set = set(np.arange(0, n, 1))
        label_cover_set = set()
        x_cover_set = set()
        
        label_have = np.zeros(o, dtype=np.float16)
        label_dict = {}
        for i in range(o):
            label_dict[i] = []
        
        for i in range(delta_num):
            delta = 0.1 + i * self.delta_step
            neighbor_num = np.sum(dis <= delta, axis=1)
            for j in range(n):
                neighbor = dis_index[j, :neighbor_num[j]]
                judge = np.sum(self.y_train[neighbor], axis=0) / len(neighbor)
                PL = np.where(judge >= self.alpha)[0]
                if len(PL) > 0:
                    if tuple(PL) in PLNC_dic:
                        PLNC_dic[tuple(PL)].append(neighbor)
                    else:
                        PLNC_dic[tuple(PL)] = [neighbor]
                    label_cover_set = label_cover_set | set(PL)
                    x_cover_set = x_cover_set | set(neighbor)
                
                index = np.where(judge >= label_have)[0]
                for k in index:
                    if judge[k] == label_have[k]:
                        label_dict[k].append(neighbor)
                    else:
                        label_dict[k] = [neighbor]
                        label_have[k] = judge[k]

        # Cover all labels
        label_remain_set = label_set - label_cover_set
        if len(label_remain_set) > 0:
            for i in label_remain_set:
                if label_have[i] >= self.beta:
                    current_neighbor = label_dict[i][0]
                    PLNC_dic[tuple([i])] = [current_neighbor]

        return PLNC_dic

    def _PLNC2(self):
        n, o = self.y_train.shape
        dis = self._calculate_distance_matrix()
        np.fill_diagonal(dis, -1)
        dis_index = np.argsort(dis, axis=1)
        delta_num = int(0.9 / self.delta_step)
        PLNC_dic = {}
        
        label_dict = {}
        for i in range(o):
            label_dict[i] = []
        
        for i in range(delta_num):
            delta = 0.1 + i * self.delta_step
            neighbor_num = np.sum(dis <= delta, axis=1)
            for j in range(n):
                neighbor = dis_index[j, :neighbor_num[j]]
                judge = np.sum(self.y_train[neighbor], axis=0) / len(neighbor)
                PL = np.where(judge >= self.alpha)[0]
                if len(PL) > 0:
                    if tuple(PL) in PLNC_dic:
                        PLNC_dic[tuple(PL)].append(neighbor)
                    else:
                        PLNC_dic[tuple(PL)] = [neighbor]

        return PLNC_dic

    def _PLNC(self):
        n, o = self.y_train.shape
        dis = self._calculate_distance_matrix()
        np.fill_diagonal(dis, -1)
        dis_index = np.argsort(dis, axis=1)
        delta_num = int(0.9 / self.delta_step)
        PLNC_dic = {}
        label_set = set(np.arange(0, o, 1))
        x_set = set(np.arange(0, n, 1))
        label_cover_set = set()
        x_cover_set = set()
        
        label_have = np.zeros(o, dtype=np.float16)
        label_dict = {}
        for i in range(o):
            label_dict[i] = []
        
        for i in range(delta_num):
            delta = 0.1 + i * self.delta_step
            neighbor_num = np.sum(dis <= delta, axis=1)
            for j in range(n):
                neighbor = dis_index[j, :neighbor_num[j]]
                judge = np.sum(self.y_train[neighbor], axis=0) / len(neighbor)
                PL = np.where(judge >= self.alpha)[0]
                if len(PL) > 0:
                    if tuple(PL) in PLNC_dic:
                        PLNC_dic[tuple(PL)].append(neighbor)
                    else:
                        PLNC_dic[tuple(PL)] = [neighbor]
                    label_cover_set = label_cover_set | set(PL)
                    x_cover_set = x_cover_set | set(neighbor)
                
                index = np.where(judge >= label_have)[0]
                for k in index:
                    if judge[k] == label_have[k]:
                        label_dict[k].append(neighbor)
                    else:
                        label_dict[k] = [neighbor]
                        label_have[k] = judge[k]

        # Cover all labels
        x_remain_set = x_set - x_cover_set
        label_have1 = np.zeros(len(x_remain_set), dtype=np.float16)
        x_dic = []
        for i in range(delta_num):
            delta = 0.1 + i * self.delta_step
            neighbor_num = np.sum(dis <= delta, axis=1)
            for index, j in enumerate(x_remain_set):
                x_dic.append([])
                if neighbor_num[j] <= 1:
                    break
                current_neighbor = dis_index[j, :neighbor_num[j]]
                judge = np.sum(self.y_train[current_neighbor], axis=0) / len(current_neighbor)
                maxl = np.max(judge)
                PL2 = np.where(judge == maxl)[0]
                if maxl > label_have1[index]:
                    x_dic[index] = [[PL2, current_neighbor]]
                    label_have1[index] = maxl

        
        compare = x_remain_set.copy()
        label_have2 = label_have1.copy()
        index1 = np.argsort(label_have2)[::-1]
        i = 0
        while len(compare) > 0 and i < len(index1):
            j = index1[i]
            if label_have2[j] <= self.beta:
                break
            current_x = set(x_dic[j][0][1])
            xxx = current_x & compare
            if len(xxx) == 0:
                i = i + 1
                continue
            else:
                compare = compare - current_x
                if tuple(x_dic[j][0][0]) in PLNC_dic:
                    PLNC_dic[tuple(x_dic[j][0][0])].append(x_dic[j][0][1])
                else:
                    PLNC_dic[tuple(x_dic[j][0][0])] = [x_dic[j][0][1]]
                i = i + 1

        return PLNC_dic

    def _LE(self, PLNC_dic):
        """
        Label Enhancement
        """
        n, o = self.y_train.shape
        x_LD = np.zeros((n, o), dtype=np.float16)
        x_LD1 = np.zeros((n, o), dtype=np.float16)
        x_LD2 = np.zeros((n, o), dtype=np.float16)
        for key, value in PLNC_dic.items():
            label_num = len(key)
            for i in range(label_num):
                mid = np.zeros(n, dtype=np.float16)
                n_num = len(value)
                for j in range(n_num):
                    mid[value[j]] = mid[value[j]] + 1
                x_LD[:, key[i]] = x_LD[:, key[i]] + mid

        mid_judge = np.sum(x_LD, axis=1)
        x_LD[mid_judge > 0] = (x_LD[mid_judge > 0].T / mid_judge[mid_judge > 0]).T
        mid_judge = np.sum(self.y_train, axis=1)
        x_LD1[mid_judge > 0] = (self.y_train[mid_judge > 0].T / mid_judge[mid_judge > 0]).T
        x_LD2[mid_judge > 0] = (1 - self.lammbda) * x_LD[mid_judge > 0] + self.lammbda * x_LD1[mid_judge > 0]
        x_LD2[mid_judge <= 0] = x_LD[mid_judge <= 0]
        return x_LD2

    def _R_a(self):
        
        n, m = self.X_train.shape
        R = np.zeros((m, n, n), dtype=np.float16)
        for i in range(m):
            current_x = self.X_train[:, i]
            M1 = np.repeat(current_x.reshape(n, 1), n, axis=1)
            M2 = np.repeat(current_x.reshape(1, n), n, axis=0)
            # R[i] = np.exp(-np.square(M1 - M2)/(2*self.sigma*self.sigma))
            R[i] = 1 - np.abs(M1 - M2)

        return R

    def _calculate_RB(self, B, R):
        """
        :param B:Attribute set
        :return: fuzzy similarity matrix 
        """
        current_R = R[B]
        R_B = np.min(current_R, axis=0)

        return R_B

    def _calculate_D_mu(self, fuzzyDC, LD):
        n = len(self.X_train)
        Mu_matrix = np.zeros((len(fuzzyDC), n), dtype=np.float16)
        for i_index, i in enumerate(fuzzyDC):
            label_index = np.array(i)
            D_space = LD[:, label_index]
            D_mu = np.sum(D_space, axis=1)
            if (np.max(D_mu) - np.min(D_mu)) == 0:
                if np.max(D_mu) != 0:
                    Mu_matrix[i_index] = D_mu / np.max(D_mu)
            else:
                Mu_matrix[i_index] = (D_mu - np.min(D_mu)) / (np.max(D_mu) - np.min(D_mu))
        return Mu_matrix

    def _calculate_pos(self, B, R1, Mu_matrix):
        """
        
        :return: Positive
        """
        if len(B) == 1:
            R = R1[B]
        else:
            R = self._calculate_RB(B, R1)
        num = len(Mu_matrix)
        pos = 0
        for i in range(num):
            D_mu = Mu_matrix[i]
            n = len(D_mu)

            D_Mu = np.repeat(D_mu.reshape(1, n), n, axis=0)
            judge1 = np.concatenate(((1 - R).reshape(1, n, n), D_Mu.reshape(1, n, n)), axis=0)
            lower_appro = np.min(np.max(judge1, axis=0), axis=1)
            pos = pos + np.sum(lower_appro)

        return pos

    def train(self):
        """
        :return: The ranking of attributes
        """
        n, m = self.X_train.shape
        if self.ablation == 0:
            PLNC = self._PLNC3()
        elif self.ablation == 1:
            PLNC = self._PLNC1()
        elif self.ablation == 2:
            PLNC = self._PLNC2()
        else:
            PLNC = self._PLNC()
        all_PLNL = PLNC.keys()
        fuzzyDC = []
        for i in all_PLNL:
            if len(i) == 1:
                fuzzyDC.append(i)
            elif len(PLNC[i]) >= 0.1 * n:
                fuzzyDC.append(i)
        LD = self._LE(PLNC)
        pos_record = np.zeros(m, dtype=np.float16)
        R2 = self._R_a()
        Mu_matrix = self._calculate_D_mu(fuzzyDC, LD)
        for i in range(m):
            pos_record[i] = self._calculate_pos([i], R2, Mu_matrix)
        selected = np.argsort(pos_record)[::-1]
        return selected
