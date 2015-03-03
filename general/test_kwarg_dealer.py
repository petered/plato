from general.kwarg_dealer import KwargDealer

__author__ = 'peter'


def my_favourite_system(planet, **kwargs):
    kd = KwargDealer(**kwargs)

    if planet == 'Earth':
        params = kd({'moon': 'the_moon', 'person': 'Jesus'})
        favsys = 'My Faviourite planet is %s, my favourite moon is %s, and my faviourite person is %s' %(planet, params['moon'], params['person'])
    elif planet == 'Mars':
        params = kd({'moon': 'Phobos'})
        favsys = 'My Favourite planet is %s, my favourite moon is %s' % (planet, params['moon'])
    elif planet == 'Venus':
        favsys == 'My favourite planet is Venus.'

    assert kd.is_empty(), 'Keyw'
    print favsys


def test_kwarg_dealer():



    for figure in figure_numbers:
        print 'Testing Figure %s ...' % (figure, )
        demo_create_figure(figure, test_mode = True)
        print '... Passed.'

if __name__ == '__main__':
    test_kwarg_dealer()
