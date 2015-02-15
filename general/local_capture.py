import sys


def execute_and_capture_locals(fcn, *args, **kwargs):
    """
    Execute the function with the provided arguments, and return both the output and
    the LOCAL VARIABLES of the function.  This is crazy.  It actually looks inside the
    function at the time that return is called and grabs all the local variables.
    """
    _locs = [None]

    def tracer(frame, event, arg):
        if event == 'return':
            # Note - this is called for every return of every function called within.  The only
            # reason it's ok is that the final return is always that of the decorated function.
            # Still, we're often doing thousands of unnecessary copies.
            local_variables = frame.f_locals.copy()
            _locs[0] = local_variables

    with _CaptureInnards(tracer):
        out = fcn(*args, **kwargs)

    local_vars, = _locs

    assert local_vars is not None, 'Failed to capture locals.   Why?'

    return out, local_vars


class _CaptureInnards(object):
    """
    Implementation detail that you shouldn't need to look at.  It's job in life is to make the local things work
    when execute_and_capture_locals is called from within another execute_and_capture_locals.
    """
    _profiler_stack = []

    def __init__(self, profile_fcn):
        self._profile_fcn = profile_fcn

    def __enter__(self):
        _CaptureInnards._profiler_stack.append(self._profile_fcn)
        sys.setprofile(self._profile_fcn)

    def __exit__(self, _, _1, _2):
        _CaptureInnards._profiler_stack.pop()
        new_profiler = None if len(_CaptureInnards._profiler_stack)==0 else _CaptureInnards._profiler_stack[-1]
        sys.setprofile(new_profiler)
