import sys
from general.local_capture import execute_and_capture_locals

__author__ = 'peter'





# def test_local_capture():
#
#     # class naive_capture_innards(object):
#     #
#     #     def __init__(self, fcn):
#     #         self._fcn = fcn
#     #         self._locals = None
#     #
#     #     def locals(self):
#     #         return self._locals
#     #
#     #     def __call__(self, *args, **kwargs):
#     #
#     #         def tracer(frame, event, arg):
#     #             if event == 'return':
#     #                 self._locals = frame.f_locals.copy()
#     #
#     #         sys.setprofile(tracer)
#     #         try:
#     #             out = self._fcn(*args, **kwargs)
#     #         finally:
#     #             sys.setprofile(None)
#     #         return out
#
#     @naive_capture_innards
#     def _module_level_fcn():
#         a=3
#         return a+4
#
#     @naive_capture_innards
#     def _get_nested_fcn_with_innards():
#         b=4
#         @naive_capture_innards
#         def nested():
#             return _module_level_fcn()+b+5
#         return nested
#
#
#     fcn = _module_level_fcn
#     out = fcn()
#     assert out == 3+4
#     assert fcn.locals() == {'a': 3}
#
#     fcn = _get_nested_fcn_with_innards()
#     out = fcn()
#     assert out == 7+4+5
#     with pytest.raises(AssertionError):
#         assert fcn.locals() == {'b': 4}
#         # Why?  Because the inner call turned off the profiler!
#     assert fcn.locals() is None


def test_execute_and_capture_locals():

    def func():
        a=3
        return a+4

    def outer_func():
        b=4
        def nested():
            return func()+b+5
        return nested

    out, local_vars = execute_and_capture_locals(func)
    assert out == 3+4
    assert local_vars == {'a': 3}
    assert sys.getprofile() is None

    out, local_vars = execute_and_capture_locals(outer_func())
    assert out == 7+4+5
    assert local_vars == {'b': 4, 'func': func}
    assert sys.getprofile() is None


if __name__ == '__main__':

    test_execute_and_capture_locals()