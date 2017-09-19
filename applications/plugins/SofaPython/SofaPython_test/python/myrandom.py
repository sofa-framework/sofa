import random

# Simply generating a new random number.
# Note the number is only generated at the first module import
# while the already imported module would not execute this code
# and would not generate a new number.
random_number = random.random()
