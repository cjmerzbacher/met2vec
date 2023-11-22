
def boolean_string(s):
    if s.lower() not in {'true', 'false'}:
        raise ValueError('Not a valid boolean string.')
    return s.lower() == 'true'