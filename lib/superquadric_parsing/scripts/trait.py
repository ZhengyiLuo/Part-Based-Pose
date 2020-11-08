from traits.api import HasTraits, Str, Range, Enum
from traitsui.api import Item, RangeEditor, View
ETSConfig.toolkit = 'qt4'

person_view = View(
    Item('name'),
    Item('gender'),
    Item('age', editor=RangeEditor(mode='spinner', low=0, high=150)),
    buttons=['OK', 'Cancel'],
    resizable=True,
)

class Person(HasTraits):
    name = Str('Jane Doe')
    age = Range(low=0)
    gender = Enum('female', 'male')

person = Person(age=30)
person.configure_traits(view=person_view)