import numpy as np
import itertools

X = [0, 3, 6, 7, 15, 10, 16, 5, 8, 1.5]
Y = [1, 2, 1, 4.5, -1, 2.5, 11, 6, 9, 12]
N = len(X)
#P=250
#n=0.8
#pm=0.2
#Tmax=1000

dist_matrix = []

def generate_population(P: int):
    return np.array([np.random.permutation(N) for _ in range(P)])

def calculate_cost(arr):
    sum = 0
    for k in range(N-1):
        x_diff = (X[arr[k]]-X[arr[k+1]])
        y_diff = (Y[arr[k]]-Y[arr[k+1]])
        sum += np.sqrt(x_diff**2 + y_diff**2)



def main():
    print(generate_population(10))


if __name__ == "__main__":
    main()
