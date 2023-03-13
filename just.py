from math import floor


a = [23.987, 45.8999, 78.22]

print(floor(a[0]))
print("here", a[0::2])
print([a[0]])

units = [round(a[0]), round(a[1]), round(a[2])]

print(units)
size = [len(units), [round(a[0]), round(a[1]), round(a[2])]]

print(size[1][0])