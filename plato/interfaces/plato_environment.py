from IPython.core.magics import logging
import inspect
from plato.core import _SymbolicFunctionWrapper
from types import ModuleType


def get_locally_defined_things(module_dict):
    """
    Given a dict of modules, import all functions and classes defined within these modules.  This takes a little extra
    effort due to plato's decorator situation.

    to get the dict of modules, import them and then call locals().
    See plato.tools.all for an example.

    :param module_dict: A dict<str: module>
    :return: a dict<object_name: object> where object is a class or function defined within the module
        with name module_name
    """

    things = {}

    for module_name, module in module_dict.iteritems():

        if not isinstance(module, ModuleType):
            continue

        # Things will be imported under the following conditions:
        # 1) They are not private (do not begin with "_")
        # 2) They are defined withing the file.  Difficult exception here is symbolically decorated things,
        #    which, even if defined within the file, hace their actual class in the decorators file (its complicated).
        things[module_name] = {
            name: thing for name, thing in inspect.getmembers(module) if
            not name.startswith('_') and (
                (hasattr(thing, '__module__') and thing.__module__ == module_name) or
                isinstance(thing, _SymbolicFunctionWrapper) and thing.original.__module__ == module_name or
                isinstance(thing, type) and issubclass(thing, _SymbolicFunctionWrapper) and thing.fcn.__module__ == module_name
                )
            }

    # Warn of name conflicts here.
    used_names = {}
    items = []
    for module_name, contents in things.iteritems():
        for thing_name, thing in contents.iteritems():
            if thing_name in used_names:
                logging.warn('%s from module %s just won a name conflict with %s from module %s.' % (thing_name, module_name, thing_name, used_names[thing_name]))
            used_names[thing_name] = module_name
            items.append((thing_name, thing))

    locally_defined_things = dict(items)
    return locally_defined_things
