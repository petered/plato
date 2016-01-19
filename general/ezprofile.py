from time import time
__author__ = 'peter'


class EZProfiler(object):

    def __init__(self, print_result = True, profiler_name = 'Profile'):
        self.print_result = print_result
        self.start_time = None  # Hopefully this removes overhead of creating a parameter after the clock is running.
        self.profiler_name = profiler_name

    def __enter__(self):
        start_time = time()
        self.start_time = start_time
        return self

    def __exit__(self, *args):
        end_time = time()
        self.elapsed_time = end_time - self.start_time
        if self.print_result:
            self.print_elapsed()

    def print_elapsed(self):

        print '%s: Elapsed time is: %.4gs' % (self.profiler_name, self.elapsed_time, )
