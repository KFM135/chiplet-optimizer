from cost_function import *

parameter_space = np.array([[0, 2],[1, 64], [0, 62], [0, 3], [1, 50], [1, 100], [1, 10], [0, 3], [1, 50], [1, 100], [1, 10], [0, 3], [1, 50], [1, 100], [1, 10]])

best_parameter = parameter_space[:, 0] + rand(len(parameter_space)) * (parameter_space[:, 1] - parameter_space[:, 0])
best_parameter = best_parameter.round()
parameter_after = action_refined(best_parameter)
best_throughput, _, _, _, _, _ = throughput(parameter_after)
curr_param, curr_throughput = best_parameter, best_throughput

num_trials = 500000
tmp = 400
step_size = 10

for i in range(num_trials):
    candidate_param_before = curr_param + uniform(-1, 1, len(parameter_space)) * step_size
    candidate_param_before = [int(i) for i in candidate_param_before]
    for j in range(0, len(candidate_param_before)):
        if candidate_param_before[j] > parameter_space[j, 1]:
            candidate_param_before[j] = parameter_space[j, 1]
        if candidate_param_before[j] < parameter_space[j, 0]:
            candidate_param_before[j] = parameter_space[j, 0]
    candidate_param = action_refined(candidate_param_before)
    candidate_throughput, _, _, _, _, _ = throughput(candidate_param)

    if candidate_throughput > best_throughput:
        best_parameter, best_throughput = candidate_param, candidate_throughput

    t = tmp / float(i + 1)

    if candidate_throughput > curr_throughput or rand() < t:
        curr_param, curr_throughput = candidate_param, candidate_throughput

    # print(
    #     f'Iteration:{i}---best param:{best_parameter}, best thruput:{best_throughput}, candidate param:{candidate_param}, candidate throughput:{candidate_throughput}, curr_param:{curr_param}, curr_throughput:{curr_throughput}')

print(f'Best Parameter:{best_parameter}')