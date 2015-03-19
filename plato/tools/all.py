module_dict = locals().copy()
from plato.interfaces.plato_environment import get_locally_defined_things
locally_defined_things = get_locally_defined_things(module_dict)
globals().update(locally_defined_things)
