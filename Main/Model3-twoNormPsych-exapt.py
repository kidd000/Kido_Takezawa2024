import sys, os, csv
import random as rnd
import numpy as np
from numba import njit
from scipy.stats import truncnorm

# import warnings
# warnings.filterwarnings('ignore')

def run():
    """
    Main function to run the simulation.
    """

    # Global variables-------------------------------------------------------
    DES_NORM_PSYCH_INIT_SIGMA = 0.25  # Descriptive norm psychology initial sigma
    BEN_CA = 4  # Benefit of collective action
    COST_CA = 1  # Cost of collective action
    GROUP_SIZE = 16  # Size of the group *In SI, we set this variable in {8, 16, 24}
    HALF_EFFORT = 0.5*GROUP_SIZE  # Half-success parameter (environmental harshness)
    COST_PUNISHING = 0.5*GROUP_SIZE/8  # Cost of punishing
    COST_PUNISHED = 3*COST_PUNISHING  # Cost of being punished
    COST_MONITORING = 0.05*GROUP_SIZE/8  # Cost of monitoring for free-riders
    GROUP_NUM = 500  # Number of groups
    POP_SIZE = GROUP_NUM * GROUP_SIZE  # Population size
    ROUND_NUM = 40  # Number of rounds per generation
    MUT_OCCUR = 10000  # Mutation occurrence
    MAX_GEN = 30000  # Maximum number of generations
    MIGRATION_RATE = 1/2 # Migarationn rate *In SI, we set this variable in {1/4, 1/2, 3/4}
    MIGRATION_NUM = int(MIGRATION_RATE*GROUP_SIZE) # Number of migrants in each group
    # Global variables-------------------------------------------------------

    # Functions-------------------------------------------------------
    @njit
    def calcPayoff(str_X, str_Y, num_X, num_Y):
        """
        Calculate the payoff of an agent based on its strategies and the number of X and Y strategies in the group.

        Args:
            str_X (int): The X strategy of the agent.
            str_Y (int): The Y strategy of the agent.
            num_X (int): The number of X strategies in the group.
            num_Y (int): The number of Y strategies in the group.

        Returns:
            float: The payoff of the agent.
        """
        pi_CA = BEN_CA * num_X / (num_X + HALF_EFFORT)
        Coop_Cost = str_X * COST_CA
        Punish_Cost = str_Y * ((GROUP_SIZE - 1 - num_X + str_X) * COST_PUNISHING / (GROUP_SIZE - 1) + COST_MONITORING)
        Punished_Cost = (1 - str_X) * COST_PUNISHED * (num_Y - str_Y) / (GROUP_SIZE - 1)
        return (pi_CA - Coop_Cost - Punish_Cost - Punished_Cost)

    @njit
    def twoNormPsych_utility_max(str_X, str_Y, inj_normPsych, des_normPsych, num_X, num_Y):
        """
        Function to calculate the utility maximization based on two norm psychology.

        Args:
            str_X (int): The current X strategy of the agent.
            str_Y (int): The current Y strategy of the agent.
            inj_normPsych (float): Injunctive norm psychology.
            des_normPsych (float): Descriptive norm psychology.
            num_X (int): The number of X strategies in the group.
            num_Y (int): The number of Y strategies in the group.

        Returns:
            tuple: The new X and Y strategies of the agent.
        """
        if rnd.random() <= 0.95:
            Std = (1 - inj_normPsych) * (1 - des_normPsych) + inj_normPsych + des_normPsych
            wtPayoff = (1 - inj_normPsych) * (1 - des_normPsych) / Std
            wtInj = inj_normPsych / Std
            wtDes = des_normPsych / Std
            num_Others = GROUP_SIZE - 1
            des_Norm_X = (num_X - str_X) / (num_Others)
            des_Norm_not_X = (num_Others - (num_X - str_X)) / (num_Others)
            des_Norm_Y = (num_Y - str_Y) / (num_Others)
            des_Norm_not_Y = (num_Others - (num_Y - str_Y)) / (num_Others)
            utility_ls = [
                # if choose DN(0, 0)
                wtPayoff * calcPayoff(str_X=0, str_Y=0, num_X=num_X - str_X, num_Y=num_Y - str_Y) +
                wtDes * (des_Norm_not_X + des_Norm_not_Y),

                # if choose DP(0, 1)
                wtPayoff * calcPayoff(str_X=0, str_Y=1, num_X=num_X - str_X, num_Y=num_Y - str_Y + 1) +
                wtInj * (NORM_VALUE_Y) +
                wtDes * (des_Norm_not_X + des_Norm_Y),

                # if choose CN(1, 0)
                wtPayoff * calcPayoff(str_X=1, str_Y=0, num_X=num_X - str_X + 1, num_Y=num_Y - str_Y) +
                wtInj * (NORM_VALUE_X) +
                wtDes * (des_Norm_X + des_Norm_not_Y),

                # if choose CP(1, 1)
                wtPayoff * calcPayoff(str_X=1, str_Y=1, num_X=num_X - str_X + 1, num_Y=num_Y - str_Y + 1) +
                wtInj * (NORM_VALUE_X + NORM_VALUE_Y) +
                wtDes * (des_Norm_X + des_Norm_Y)]

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
        """
        Perform mutation on the norm psychology of an agent.

        Args:
            normPsych (float): The current norm psychology of the agent.

        Returns:
            float: The new norm psychology of the agent after mutation.
        """
        while True:
            mutation = np.random.normal(loc=0, scale=0.1, size=1)
            new_normPsych = normPsych + mutation
            if 0 <= new_normPsych <= 1:
                break
        return new_normPsych[0]
    # Functions-------------------------------------------------------

    @njit
    def main():
        """
        Main function to run the simulation.
        """

        # Global variables-------------------------------------------------------
        # Output lists
        sum_strDP_ls = np.zeros(MAX_GEN, dtype="int")  # Sum of DP strategies per generation
        sum_strCN_ls = np.zeros(MAX_GEN, dtype="int")  # Sum of CN strategies per generation
        sum_strCP_ls = np.zeros(MAX_GEN, dtype="int")  # Sum of CP strategies per generation
        mean_fitness_ls = np.zeros(MAX_GEN, dtype="float")  # Mean fitness per generation
        mean_inj_normPsych_ls = np.zeros(MAX_GEN, dtype="float")  # Mean injunctive norm psychology per generation
        mean_des_normPsych_ls = np.zeros(MAX_GEN, dtype="float")  # Mean descriptive norm psychology per generation

        # Norm internalization traits generation
        inj_normPsych_arr = np.random.rand(GROUP_NUM, GROUP_SIZE) * 0.05  # Injunctive norm psychology array
        des_normPsych_arr = np.copy(global_des_normPsych_arr)  # Descriptive norm psychology array
        # Global variables-------------------------------------------------------

        for generation in range(MAX_GEN):
            # 2D numpy array - x strategy, y strategy, payoff, fitness
            str_X_arr = np.random.randint(0, high=2, size=(GROUP_NUM, GROUP_SIZE))  # X strategy array
            str_Y_arr = np.random.randint(0, high=2, size=(GROUP_NUM, GROUP_SIZE))  # Y strategy array
            cum_Payoff_arr = np.zeros((GROUP_NUM, GROUP_SIZE), dtype="float")  # Cumulative payoff array
            cum_grp_Success_arr = np.zeros(GROUP_NUM, dtype="float")  # Cumulative group success array

            # Start of one generation per group
            for group_num in range(GROUP_NUM):
                # First, assign strategies randomly for the first round
                tgt_grp_str_X_arr = str_X_arr[group_num, :]  # Target group X strategy array
                tgt_grp_str_Y_arr = str_Y_arr[group_num, :]  # Target group Y strategy array
                tgt_grp_inj_normPsych_arr = inj_normPsych_arr[group_num, :]  # Target group injunctive norm psychology array
                tgt_grp_des_normPsych_arr = des_normPsych_arr[group_num, :]  # Target group descriptive norm psychology array

                # List to store the number of cooperators for all rounds. Used to calculate the success rate of collective action for each group later.
                for round_num in range(ROUND_NUM):
                    num_X = np.sum(tgt_grp_str_X_arr)  # Number of X strategies
                    num_Y = np.sum(tgt_grp_str_Y_arr)  # Number of Y strategies

                    # Cumulative success array for group selection
                    cum_grp_Success_arr[group_num] += num_X / (num_X + HALF_EFFORT)

                    # Calculate payoff from the strategies of all group members and own strategy
                    for mem in range(GROUP_SIZE):
                        cum_Payoff_arr[group_num, mem] += calcPayoff(
                            str_X=tgt_grp_str_X_arr[mem], str_Y=tgt_grp_str_Y_arr[mem], num_X=num_X, num_Y=num_Y)

                    # At the end of the last round, tally the strategies and skip the next process (strategy revision)
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

                    # With a 1/4 probability, revise the strategy to maximize utility
                    mask_isRevised = np.random.rand(GROUP_SIZE) <= 0.25
                    if mask_isRevised.any():
                        mask_isRevised_idxes = [
                            i for i, x in enumerate(mask_isRevised) if x]
                        for idx in mask_isRevised_idxes:
                            tgt_grp_str_X_arr[idx], tgt_grp_str_Y_arr[idx] = twoNormPsych_utility_max(
                                str_X=tgt_grp_str_X_arr[idx], str_Y=tgt_grp_str_Y_arr[idx], inj_normPsych=tgt_grp_inj_normPsych_arr[idx], des_normPsych=tgt_grp_des_normPsych_arr[idx], num_X=num_X, num_Y=num_Y)

            # Calculate fitness and store it. Make sure it's between 0 and 1.
            Fitness_arr = np.zeros((GROUP_NUM, GROUP_SIZE), dtype="float")
            for group_num in range(GROUP_NUM):
                Fitness_arr[group_num, :] = 1 + \
                    (cum_Payoff_arr[group_num, :] / ROUND_NUM)  # Calculate fitness
                Fitness_arr[group_num, :] = np.maximum(Fitness_arr[group_num, :], 0.00001)  # Ensure fitness is at least 0.00001

            # Calculate the success rate of the group (per round)
            Grp_Success_arr = cum_grp_Success_arr / ROUND_NUM  # Calculate group success rate
            Grp_Success_arr = np.maximum(Grp_Success_arr, 0.00001)  # Ensure success rate is at least 0.00001

            # Record parameters to be written to CSV ---------------------------------------
            mean_inj_normPsych_ls[generation] = np.mean(inj_normPsych_arr)  # Mean injunctive norm psychology per generation
            mean_des_normPsych_ls[generation] = np.mean(des_normPsych_arr)  # Mean descriptive norm psychology per generation
            mean_fitness_ls[generation] = np.mean(Fitness_arr)  # Mean fitness per generation
            # Record parameters to be written to CSV ---------------------------------------

            # Selection for group ---------------------------------------
            grp_success_wt_arr = Grp_Success_arr / \
                np.sum(Grp_Success_arr) if np.sum(
                    Grp_Success_arr) != 0 else np.zeros(GROUP_NUM)  # Calculate group success weight array
            grp_survival_arr = np.random.multinomial(
                n=GROUP_NUM, pvals=grp_success_wt_arr, size=1).flatten()  # Calculate group survival array
            # Based on the return of the group multinomial distribution, replicate the group
            offspring_group_number = 0
            prev_grp_inj_normPsych_arr = np.copy(inj_normPsych_arr)
            prev_grp_des_normPsych_arr = np.copy(des_normPsych_arr)
            prev_grp_Fitness_arr = np.copy(Fitness_arr)
            for grp_ID in range(GROUP_NUM):
                offspring_group_count = grp_survival_arr[grp_ID]
                if offspring_group_count == 0:
                    continue
                else:
                    new_grp_inj_normPsych = prev_grp_inj_normPsych_arr[grp_ID, :]
                    new_grp_des_normPsych = prev_grp_des_normPsych_arr[grp_ID, :]
                    new_grp_fitness = prev_grp_Fitness_arr[grp_ID, :]
                    for _ in range(offspring_group_count):
                        inj_normPsych_arr[offspring_group_number,
                                          :] = new_grp_inj_normPsych  # Set new group injunctive norm psychology
                        des_normPsych_arr[offspring_group_number,
                                          :] = new_grp_des_normPsych  # Set new group descriptive norm psychology
                        Fitness_arr[offspring_group_number, :] = new_grp_fitness  # Set new group fitness
                        offspring_group_number += 1
            # Selection for group ---------------------------------------

            # Selection for member ---------------------------------------
            # IndSelection(Fitness_arr, inj_normPsych_arr, des_normPsych_arr)
            for grp_ID in range(GROUP_NUM):
                fitness_wt_arr = Fitness_arr[grp_ID, :] / np.sum(Fitness_arr[grp_ID, :]) if np.sum(
                    Fitness_arr[grp_ID, :]) != 0 else np.repeat(1 / GROUP_SIZE, repeats=GROUP_SIZE)  # Calculate fitness weight array
                ind_survival_arr = np.random.multinomial(
                    n=GROUP_SIZE, pvals=fitness_wt_arr, size=1).flatten()  # Calculate individual survival array
                # Based on the return of the individual multinomial distribution, replicate the member
                offspring_number = 0
                prev_ind_inj_normPsych = np.copy(inj_normPsych_arr[grp_ID, ])  # Copy of previous individual injunctive norm psychology
                prev_ind_des_normPsych = np.copy(des_normPsych_arr[grp_ID, ])  # Copy of previous individual descriptive norm psychology
                for group_member_id in range(GROUP_SIZE):
                    offspring_count = ind_survival_arr[group_member_id]  # Number of offspring for the member
                    if offspring_count == 0:
                        continue
                    else:
                        new_ind_inj_normPsych = prev_ind_inj_normPsych[group_member_id]  # New individual injunctive norm psychology
                        new_ind_des_normPsych = prev_ind_des_normPsych[group_member_id]  # New individual descriptive norm psychology
                        for _ in range(offspring_count):
                            inj_normPsych_arr[grp_ID][offspring_number] = new_ind_inj_normPsych  # Set new individual injunctive norm psychology
                            des_normPsych_arr[grp_ID][offspring_number] = new_ind_des_normPsych  # Set new individual descriptive norm psychology
                            offspring_number += 1
            # Selection for member ---------------------------------------

            # Mutation ---------------------------------------
            mask_inj_isMutated = np.random.rand(
                GROUP_NUM, GROUP_SIZE) <= (1 / MUT_OCCUR)  # Mask for injunctive norm psychology mutation
            if mask_inj_isMutated.any():
                for mutated_mem in range(mask_inj_isMutated.sum()):
                    idx_grp_arr = mask_inj_isMutated.nonzero()[0]  # Group indices for mutation
                    idx_mem_arr = mask_inj_isMutated.nonzero()[1]  # Member indices for mutation
                    inj_normPsych_arr[idx_grp_arr[mutated_mem]][idx_mem_arr[mutated_mem]] = Mutation(
                        inj_normPsych_arr[idx_grp_arr[mutated_mem]][idx_mem_arr[mutated_mem]])  # Apply mutation
            mask_des_isMutated = np.random.rand(
                GROUP_NUM, GROUP_SIZE) <= (1 / MUT_OCCUR)  # Mask for descriptive norm psychology mutation
            if mask_des_isMutated.any():
                for mutated_mem in range(mask_des_isMutated.sum()):
                    idx_grp_arr = mask_des_isMutated.nonzero()[0]  # Group indices for mutation
                    idx_mem_arr = mask_des_isMutated.nonzero()[1]  # Member indices for mutation
                    des_normPsych_arr[idx_grp_arr[mutated_mem]][idx_mem_arr[mutated_mem]] = Mutation(
                        des_normPsych_arr[idx_grp_arr[mutated_mem]][idx_mem_arr[mutated_mem]])  # Apply mutation
            # Mutation ---------------------------------------

            # Migration ---------------------------------------
            # Step 1: Create arrays to store the traits of migrating agents and their indices
            migrate_inj_arr = np.zeros((GROUP_NUM, MIGRATION_NUM), dtype="float")  # Array to store the injunctive norm psychology of migrating agents
            migrate_des_arr = np.zeros((GROUP_NUM, MIGRATION_NUM), dtype="float")  # Array to store the descriptive norm psychology of migrating agents
            migrate_index_arr = np.zeros((GROUP_NUM, MIGRATION_NUM), dtype="int")  # Array to store the indices of migrating agents

            # Step 2: For each group, extract the migrating agents and store their traits in the arrays
            for group_num in range(GROUP_NUM):
                migrate_index_arr[group_num, :] = np.random.choice(
                    np.arange(GROUP_SIZE), size=MIGRATION_NUM, replace=False)  # Indices of migrating agents
                migrate_inj_arr[group_num,
                                :] = inj_normPsych_arr[group_num][migrate_index_arr[group_num, :]].flatten()  # Injunctive norm psychology of migrating agents
                migrate_des_arr[group_num,
                                :] = des_normPsych_arr[group_num][migrate_index_arr[group_num, :]].flatten()  # Descriptive norm psychology of migrating agents

            # Step 3: The traits of migrating agents are ordered from group 0, so shuffle them
            migrate_inj_arr = migrate_inj_arr.flatten()
            np.random.shuffle(migrate_inj_arr)  # Shuffle the injunctive norm psychology of migrating agents
            migrate_des_arr = migrate_des_arr.flatten()
            np.random.shuffle(migrate_des_arr)  # Shuffle the descriptive norm psychology of migrating agents

            # Step 4: Empty the traits of the migrated agents and sequentially insert the shuffled traits
            for group_num in range(GROUP_NUM):
                inj_normPsych_arr[group_num, :] = np.append(np.delete(
                    inj_normPsych_arr[group_num], migrate_index_arr[group_num, :]), migrate_inj_arr[group_num * GROUP_SIZE // 2:(group_num + 1) * GROUP_SIZE // 2])  # Update the injunctive norm psychology array
                des_normPsych_arr[group_num, :] = np.append(np.delete(
                    des_normPsych_arr[group_num], migrate_index_arr[group_num, :]), migrate_des_arr[group_num * GROUP_SIZE // 2:(group_num + 1) * GROUP_SIZE // 2])  # Update the descriptive norm psychology array
            # Migration ---------------------------------------

        return sum_strDP_ls, sum_strCN_ls, sum_strCP_ls, mean_fitness_ls, mean_inj_normPsych_ls, mean_des_normPsych_ls

    # Generate norm internalization traits
    rvs_a = (0 - DES_NORM_PSYCH_INIT_MU) / DES_NORM_PSYCH_INIT_SIGMA
    rvs_b = (1 - DES_NORM_PSYCH_INIT_MU) / DES_NORM_PSYCH_INIT_SIGMA
    global_des_normPsych_arr = np.array(truncnorm.rvs(rvs_a, rvs_b, loc=DES_NORM_PSYCH_INIT_MU, scale=DES_NORM_PSYCH_INIT_SIGMA, size=(GROUP_NUM, GROUP_SIZE)), dtype="float")

    # Execute the main function
    sum_strDP_ls, sum_strCN_ls, sum_strCP_ls, mean_fitness_ls, mean_inj_normPsych_ls, mean_des_normPsych_ls = main()

    # Save the data after execution
    # Directory name
    head_chr = sys.argv[0][:-3]
    dir_name_list = ["DesNormPsychInit" + f"{DES_NORM_PSYCH_INIT_MU}",
                     "NormX" + f"{NORM_VALUE_X}",
                     "NormY" + f"{NORM_VALUE_Y}"]
    dir_name = "_".join(dir_name_list)

    # Create a directory for saving
    if os.path.exists(f"{head_chr}_{dir_name}"):
        pass
    else:
        print(f"make new directory: {head_chr}_{dir_name}")
        os.makedirs(f"{head_chr}_{dir_name}")

    # Write to CSV
    Res_dir = {"DP": sum_strDP_ls,
               "CN": sum_strCN_ls,
               "CP": sum_strCP_ls,
               "Fitness": mean_fitness_ls,
               "Inj": mean_inj_normPsych_ls,
               "Des": mean_des_normPsych_ls}

    for Res_label, Res in Res_dir.items():
        np.savetxt(
            fname=f"./{head_chr}_{dir_name}/{Res_label}_{dir_name}_{RUN}.csv", X=Res, delimiter=",", fmt='%.5f')


# Receive command line arguments
DES_NORM_PSYCH_INIT_MU = float(sys.argv[1])
RUN = int(sys.argv[2])

# Execute
for NORM_VALUE_X in np.round(np.linspace(0, 1, 11), 1):
    for NORM_VALUE_Y in np.round(np.linspace(0, 1, 11), 1):
        run()
