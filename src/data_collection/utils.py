from multiprocessing import Pool



def run_merge_responses(data, func, n_cpus_a = 48):
    """
    Runs a given function in parallel on subsets of the input data using a specified number of CPUs.

    Parameters:
        data (List): The input data that needs to be processed by the function.
        func (callable): The function that needs to be applied on the input data.
        n_cpus_a (int, optional): The number of CPUs to use for parallel processing. Defaults to 48.

    Returns:
        List: The final result after merging the results obtained from the function applied on subsets of the input data.
    """
    pool = Pool()
    len_items = len(data)
    results_cpu = []
    if len_items>=n_cpus_a:
        n_cpus = n_cpus_a
    else:
        n_cpus = len_items
    for i in range(n_cpus):
        result_i = pool.apply_async(func, [data[int((i)*len_items/n_cpus):int((i+1)*len_items/n_cpus)]])
        results_cpu.append(result_i)

    answers_cpu = []
    for result_i in results_cpu:
        answers_cpu.append(result_i.get())
    final_answer = []

    for answer_i in answers_cpu:
        final_answer += answer_i
    
    return final_answer

