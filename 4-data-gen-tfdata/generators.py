import tensorflow as tf


def gen_numbers(num):
    for i in range(num):
        yield i


for number in gen_numbers(10):
    print(number)
