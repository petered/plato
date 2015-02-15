import sys


def execute_and_capture_locals(fcn, *args, **kwargs):

    _locs = [None]

    def tracer(frame, event, arg):
        # print event
        if event == 'return':
            # Note - this is called for every return of every function called within.  The only
            # reason it's ok is that the final return is always that of the decorated function.
            # Still, we're often doing thousands of unnecessary copies.
            local_variables = frame.f_locals.copy()
            # if 'wake_hidden' in local_vars:
                # print 'AAAAAAAA'
            # print local_vars.keys()
            _locs[0] = local_variables

    with CaptureInnards(tracer):
        out = fcn(*args, **kwargs)

    local_vars, = _locs

    assert local_vars is not None, 'Failed to capture locals.   Why?'

    return out, local_vars


# class capture_locals(fcn):



class CaptureInnards(object):
    """
    So.
    """
    _profiler_stack = []

    def __init__(self, profile_fcn):
        self._profile_fcn = profile_fcn

    def __enter__(self):
        CaptureInnards._profiler_stack.append(self._profile_fcn)
        sys.setprofile(self._profile_fcn)
        # print 'Depth: %s' % len(CaptureInnards._profiler_stack)

    def __exit__(self, _, _1, _2):
        CaptureInnards._profiler_stack.pop()
        new_profiler = None if len(CaptureInnards._profiler_stack)==0 else CaptureInnards._profiler_stack[-1]
        # print 'Depth: %s' % len(CaptureInnards._profiler_stack)
        sys.setprofile(new_profiler)
