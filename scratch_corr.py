import numpy as np

def val_to_color(val):
    if val < 0:
        r = 255
        g = int(255 * (1 + val))
        b = 0
    else:
        r = int(255 * (1 - val))
        g = 255
        b = 0
    return f"rgba({r}, {g}, {b}, 0.7)"

print(val_to_color(-1.0))
print(val_to_color(-0.5))
print(val_to_color(0.0))
print(val_to_color(0.5))
print(val_to_color(1.0))
