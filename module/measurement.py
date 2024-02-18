import time

def time_function(function_to_time, *args, **kwargs):
    """
    Measures the execution time of a given function.

    :param function_to_time: The function to measure.
    :param args: Non-keyword arguments to pass to the function.
    :param kwargs: Keyword arguments to pass to the function.
    :return: The result of the function and the time taken to execute.
    """
    start_time = time.perf_counter()

    # Call the function with any arguments and keyword arguments
    result = function_to_time(*args, **kwargs)

    end_time = time.perf_counter()

    duration = end_time - start_time
    print(f"The function '{function_to_time.__name__}' took {duration} seconds to run.")

    return result