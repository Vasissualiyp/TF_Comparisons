import numpy as np

next = 1024
nbuff = 64
ntile = 4
max_halosize = 40 # Mpc

nsub = ( next - 2 * nbuff ) / ntile
nmesh = nsub + 2 * nbuff
boxlength = max_halosize / nbuff * next

print(f"ntile: {ntile}")
print(f"nmesh: {nmesh}")
print(f"boxlength: {boxlength} Mpc")

print(f"next: {next}")
