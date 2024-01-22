import sys
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import csv
import random as rnd
import numpy as np
from numba import njit
from scipy.stats import truncnorm

# import warnings
# warnings.filterwarnings('ignore')


def run():

    # Global変数-------------------------------------------------------
    BEN_CA = 4  # 1,2,4 集合行動利益
    COST_CA = 1  # 集合行動のコスト
    GROUP_SIZE = 16  # 8,16,24 グループサイズ
    HALF_EFFORT = 0.5*GROUP_SIZE  # 0.25,0.5,1.0 half-success parameter 環境の厳しさ
    COST_PUNISHING = 0.5*GROUP_SIZE/8  # δ　0.25,0.5,1.0  罰コスト
    COST_PUNISHED = 3*COST_PUNISHING  # K 2,3,4 罰効率
    COST_MONITORING = 0.05*GROUP_SIZE/8  # フリーライダーを特定するために支払わなければならないコスト
    GROUP_NUM = 500  # 500 グループ数
    POP_SIZE = GROUP_NUM * GROUP_SIZE
    ROUND_NUM = 40  # 40 1世代のラウンド数
    MUT_OCCUR = 10000  # 10000
    MAX_GEN = 30000  # 10000
    # Global変数-------------------------------------------------------

    # 関数-------------------------------------------------------

    @njit
    def calcPayoff(str_X, str_Y, num_X, num_Y):
        """AI is creating summary for calcPayoff

        Args:
            str_X ([type]): [description]
            str_Y ([type]): [description]
            num_X ([type]): [description]
            num_Y ([type]): [description]

        Returns:
            [type]: [description]
        """
        pi_CA = BEN_CA * num_X / (num_X + HALF_EFFORT)
        Coop_Cost = str_X * COST_CA
        Punish_Cost = str_Y * ((GROUP_SIZE - 1 - num_X + str_X) * COST_PUNISHING / (GROUP_SIZE - 1) + COST_MONITORING)
        Punished_Cost = (1 - str_X) * COST_PUNISHED * (num_Y - str_Y) / (GROUP_SIZE - 1)
        return (pi_CA - Coop_Cost - Punish_Cost - Punished_Cost)

    @njit
    def oneNormPsych_utility_max(str_X, str_Y, inj_normPsych, num_X, num_Y):
        """AI is creating summary for oneNormPsych_utility_max

        Args:
            str_X ([type]): [description]
            str_Y ([type]): [description]
            inj_normPsych ([type]): [description]
            num_X ([type]): [description]
            num_Y ([type]): [description]

        Returns:
            [type]: [description]
        """
        if rnd.random() <= 0.95:
            wtPayoff = (1 - inj_normPsych)
            wtInj = inj_normPsych
            num_Others = GROUP_SIZE - 1
            utility_ls = [
                # if choose DN(0, 0)
                wtPayoff * calcPayoff(str_X=0, str_Y=0, num_X=num_X - str_X, num_Y=num_Y - str_Y) +
                wtInj * (0 * NORM_VALUE_X + 0 * NORM_VALUE_Y),

                # if choose DP(0, 1)
                wtPayoff * calcPayoff(str_X=0, str_Y=1, num_X=num_X - str_X, num_Y=num_Y - str_Y + 1) +
                wtInj * (NORM_VALUE_Y),

                # if choose CN(1, 0)
                wtPayoff * calcPayoff(str_X=1, str_Y=0, num_X=num_X - str_X + 1, num_Y=num_Y - str_Y) +
                wtInj * (NORM_VALUE_X),

                # if choose CP(1, 1)
                wtPayoff * calcPayoff(str_X=1, str_Y=1, num_X=num_X - str_X + 1, num_Y=num_Y - str_Y + 1) +
                wtInj * (NORM_VALUE_X + NORM_VALUE_Y)]
            # print(utility_ls)

            if utility_ls.index(max(utility_ls)) == 0:
                return 0, 0  # DN(0,0)
            elif utility_ls.index(max(utility_ls)) == 1:
                return 0, 1  # DP(0,1)
            elif utility_ls.index(max(utility_ls)) == 2:
                return 1, 0  # CN(1,0)
            elif utility_ls.index(max(utility_ls)) == 3:
                return 1, 1  # CP(1,1)
        else:
            return rnd.getrandbits(1), rnd.getrandbits(1)

    @njit
    def Mutation(normPsych):
        """AI is creating summary for Mutation

        Args:
            normPsych ([type]): [description]

        Returns:
            [type]: [description]
        """
        while True:
            mutation = np.random.normal(loc=0, scale=0.1, size=1)
            new_normPsych = normPsych + mutation
            if 0 <= new_normPsych <= 1:
                break
        return new_normPsych[0]
    # 関数-------------------------------------------------------

    @njit
    # @overload(np.asarray)
    def main():
        # # Global変数-------------------------------------------------------
        # 出力用リスト
        sum_strDP_ls = np.zeros(MAX_GEN, dtype="int")
        sum_strCN_ls = np.zeros(MAX_GEN, dtype="int")
        sum_strCP_ls = np.zeros(MAX_GEN, dtype="int")
        mean_fitness_ls = np.zeros(MAX_GEN, dtype="float")
        mean_inj_normPsych_ls = np.zeros(MAX_GEN, dtype="float")

        # # ここからが本番
        # # 規範内面化形質の生成
        inj_normPsych_arr = np.random.rand(GROUP_NUM, GROUP_SIZE) * 0.05
        # # Global変数-------------------------------------------------------

        for generation in range(MAX_GEN):
            # numpyの2次元配列－x戦略、y戦略, payoff, fitness
            str_X_arr = np.random.randint(0, high=2, size=(GROUP_NUM, GROUP_SIZE))
            str_Y_arr = np.random.randint(0, high=2, size=(GROUP_NUM, GROUP_SIZE))
            cum_Payoff_arr = np.zeros((GROUP_NUM, GROUP_SIZE), dtype="float")
            cum_grp_Success_arr = np.zeros(GROUP_NUM, dtype="float")

            # group毎に1世代開始
            for group_num in range(GROUP_NUM):
                # まず1ラウンド目の戦略をランダムで割り当てる
                tgt_grp_str_X_arr = str_X_arr[group_num, :]
                tgt_grp_str_Y_arr = str_Y_arr[group_num, :]
                tgt_grp_inj_normPsych_arr = inj_normPsych_arr[group_num, :]

                # 全ラウンド分の協力者の人数を入れるリスト　グループごとの集合行動の成功率を後で計算するために
                # cum_grp_Success = 0
                for round_num in range(ROUND_NUM):
                    num_X = np.sum(tgt_grp_str_X_arr)
                    num_Y = np.sum(tgt_grp_str_Y_arr)

                    # group selection用の累積成功度配列
                    cum_grp_Success_arr[group_num] += num_X / (num_X + HALF_EFFORT)

                    # グループのメンバー全員の戦略と自分の戦略から　利得計算
                    for mem in range(GROUP_SIZE):
                        cum_Payoff_arr[group_num, mem] += calcPayoff(str_X=tgt_grp_str_X_arr[mem], str_Y=tgt_grp_str_Y_arr[mem], num_X=num_X, num_Y=num_Y)

                    # 最後のラウンドに、戦略を集計して、次の処理（戦略修正）をスキップ
                    if round_num == (ROUND_NUM - 1):
                        for mem in range(GROUP_SIZE):
                            if (tgt_grp_str_X_arr[mem] == 0) & (tgt_grp_str_Y_arr[mem] == 0):
                                pass
                            elif (tgt_grp_str_X_arr[mem] == 0) & (tgt_grp_str_Y_arr[mem] == 1):
                                sum_strDP_ls[generation] += 1
                            elif (tgt_grp_str_X_arr[mem] == 1) & (tgt_grp_str_Y_arr[mem] == 0):
                                sum_strCN_ls[generation] += 1
                            elif (tgt_grp_str_X_arr[mem] == 1) & (tgt_grp_str_Y_arr[mem] == 1):
                                sum_strCP_ls[generation] += 1
                        continue

                    # 4分の1の確率で、効用を最大化する戦略に修正する
                    mask_isRevised = np.random.rand(GROUP_SIZE) <= 0.25
                    if mask_isRevised.any():
                        mask_isRevised_idxes = [
                            i for i, x in enumerate(mask_isRevised) if x]
                        for idx in mask_isRevised_idxes:
                            tgt_grp_str_X_arr[idx], tgt_grp_str_Y_arr[idx] = oneNormPsych_utility_max(str_X=tgt_grp_str_X_arr[idx], str_Y=tgt_grp_str_Y_arr[idx], inj_normPsych=tgt_grp_inj_normPsych_arr[idx], num_X=num_X, num_Y=num_Y)

            # fitnessの計算 格納　0以上1になるように
            Fitness_arr = np.zeros((GROUP_NUM, GROUP_SIZE), dtype="float")
            for group_num in range(GROUP_NUM):
                Fitness_arr[group_num, :] = 1 + \
                    (cum_Payoff_arr[group_num, :] / ROUND_NUM)
                Fitness_arr[group_num, :] = np.maximum(Fitness_arr[group_num, :], 0.00001)

            # グループの成功率(ラウンド分)の計算
            Grp_Success_arr = cum_grp_Success_arr / ROUND_NUM
            # 下駄履かせる
            Grp_Success_arr = np.maximum(Grp_Success_arr, 0.00001)

            # CSVに書き出すパラメタを記録 ---------------------------------------
            mean_inj_normPsych_ls[generation] = np.mean(inj_normPsych_arr)
            mean_fitness_ls[generation] = np.mean(Fitness_arr)
            # CSVに書き出すパラメタを記録 ---------------------------------------

            # selection for group ---------------------------------------
            grp_success_wt_arr = Grp_Success_arr / np.sum(Grp_Success_arr) if np.sum(Grp_Success_arr) != 0 else np.zeros(GROUP_NUM)
            grp_survival_arr = np.random.multinomial(n=GROUP_NUM, pvals=grp_success_wt_arr, size=1).flatten()
            # グループの多項分布のreturnをもとにグループ複製
            offspring_group_number = 0
            prev_grp_inj_normPsych_arr = np.copy(inj_normPsych_arr)
            prev_grp_Fitness_arr = np.copy(Fitness_arr)
            for grp_ID in range(GROUP_NUM):
                offspring_group_count = grp_survival_arr[grp_ID]
                if offspring_group_count == 0:
                    continue
                else:
                    new_grp_inj_normPsych = prev_grp_inj_normPsych_arr[grp_ID, :]
                    new_grp_fitness = prev_grp_Fitness_arr[grp_ID, :]
                    for _ in range(offspring_group_count):
                        inj_normPsych_arr[offspring_group_number, :] = new_grp_inj_normPsych
                        Fitness_arr[offspring_group_number, :] = new_grp_fitness
                        offspring_group_number += 1
            # selection for group ---------------------------------------

            # selection for member ---------------------------------------
            for grp_ID in range(GROUP_NUM):
                fitness_wt_arr = Fitness_arr[grp_ID, :] / np.sum(Fitness_arr[grp_ID, :]) if np.sum(Fitness_arr[grp_ID, :]) != 0 else np.repeat(1 / GROUP_SIZE, repeats=GROUP_SIZE)
                ind_survival_arr = np.random.multinomial(n=GROUP_SIZE, pvals=fitness_wt_arr, size=1).flatten()
                # グループ内メンバーの多項分布のreturnをもとにメンバー複製
                offspring_number = 0
                prev_ind_inj_normPsych = np.copy(inj_normPsych_arr[grp_ID, ])
                for group_member_id in range(GROUP_SIZE):
                    offspring_count = ind_survival_arr[group_member_id]
                    if offspring_count == 0:
                        continue
                    else:
                        new_ind_inj_normPsych = prev_ind_inj_normPsych[group_member_id]
                        for _ in range(offspring_count):
                            inj_normPsych_arr[grp_ID][offspring_number] = new_ind_inj_normPsych
                            offspring_number += 1
            # selection for member ---------------------------------------

            # mutation ---------------------------------------
            mask_inj_isMutated = np.random.rand(
                GROUP_NUM, GROUP_SIZE) <= (1 / MUT_OCCUR)
            if mask_inj_isMutated.any():
                for mutated_mem in range(mask_inj_isMutated.sum()):
                    idx_grp_arr = mask_inj_isMutated.nonzero()[0]
                    idx_mem_arr = mask_inj_isMutated.nonzero()[1]
                    inj_normPsych_arr[idx_grp_arr[mutated_mem]][idx_mem_arr[mutated_mem]] = Mutation(inj_normPsych_arr[idx_grp_arr[mutated_mem]][idx_mem_arr[mutated_mem]])
            # mutation ---------------------------------------

            # migration ---------------------------------------
            # step1 migrate_ls:グループの数×グループサイズの半分 エージェントの遺伝的に継承する形質のリスト
            #       index_ls:各グループ、どのインデックスの人（の遺伝的形質）がdisperseするのかを格納するリスト
            migrate_inj_arr = np.zeros((GROUP_NUM, GROUP_SIZE//2), dtype="float")
            migrate_index_arr = np.zeros((GROUP_NUM, GROUP_SIZE//2), dtype="int")

            # step2 group毎に移住する人（index_ls）を取り出して、形質をそれぞれmigrate_lsに格納
            for group_num in range(GROUP_NUM):
                migrate_index_arr[group_num, :] = np.random.choice(np.arange(GROUP_SIZE), size=GROUP_SIZE//2, replace=False)
                migrate_inj_arr[group_num, :] = inj_normPsych_arr[group_num][migrate_index_arr[group_num, :]].flatten()

            # step3 移住する人の形質がグループ0から順番に並んでいるので、シャッフルする
            migrate_inj_arr = migrate_inj_arr.flatten()
            np.random.shuffle(migrate_inj_arr)

            # step4 移住した人の所（index_ls）の形質を空けて、シャッフルした形質を順に入れていく
            for group_num in range(GROUP_NUM):
                inj_normPsych_arr[group_num, :] = np.append(np.delete(inj_normPsych_arr[group_num], migrate_index_arr[group_num, :]),
                                                            migrate_inj_arr[group_num * GROUP_SIZE // 2:(group_num + 1) * GROUP_SIZE // 2])
            # migration ---------------------------------------

        return sum_strDP_ls, sum_strCN_ls, sum_strCP_ls, mean_fitness_ls, mean_inj_normPsych_ls

    # --- 実行 --- #
    sum_strDP_ls, sum_strCN_ls, sum_strCP_ls, mean_fitness_ls, mean_inj_normPsych_ls = main()

    # --- 実行後のデータ保存 --- #
    # ディレクトリの名前
    head_chr = sys.argv[0][:-3]
    dir_name_list = ["NormX" + f"{NORM_VALUE_X}",
                     "NormY" + f"{NORM_VALUE_Y}"]
    dir_name = "_".join(dir_name_list)

    # 保存用のディレクトリを作成する
    if os.path.exists(f"{head_chr}_{dir_name}"):
        pass
    else:
        print(f"make new directory: {head_chr}_{dir_name}")
        os.makedirs(f"{head_chr}_{dir_name}")

    # CSV書き出し
    Res_dir = {"DP": sum_strDP_ls,
               "CN": sum_strCN_ls,
               "CP": sum_strCP_ls,
               "Fitness": mean_fitness_ls,
               "Inj": mean_inj_normPsych_ls}

    for Res_label, Res in Res_dir.items():
        np.savetxt(
            fname=f"./{head_chr}_{dir_name}/{Res_label}_{dir_name}_{RUN}.csv", X=Res, delimiter=",", fmt='%.5f')


# コマンドライン引数を受け取る
RUN = int(sys.argv[1])

# 実行
for NORM_VALUE_X in np.round(np.linspace(0, 1, 11), 1):
    for NORM_VALUE_Y in np.round(np.linspace(0, 1, 11), 1):
        run()
