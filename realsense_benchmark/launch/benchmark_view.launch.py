from realsense_benchmark.launch_utils import generate_stream_benchmark_launch_description


def generate_launch_description():
    return generate_stream_benchmark_launch_description(
        aligned_depth=False,
        cpu_loops=0,
    )
