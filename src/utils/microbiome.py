def trim_id(identifier):
    components = identifier.split(';')
    for comp in reversed(components):
        if 'uncultured' not in comp and comp != '__':
            return comp
    return identifier  # Return None if no meaningful group is found