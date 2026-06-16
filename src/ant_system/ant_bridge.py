import random

m = 1000
k = 20
d = 2

L_left = 5
L_right = 8

def symmetric_bridge(m, k, d):
    L_m = 0
    R_m = 0

    for ant in range(m):
        R_p = (R_m + k) ** d
        L_p = (L_m + k) ** d

        P_r = R_p / (R_p + L_p)

        r = random.uniform(0, 1)

        if r < P_r:
            R_m += 1
        else:
            L_m += 1

    print(f"Symmetric bridge: left = {L_m} times, right = {R_m} times.")

def asymmetric_bridge(m, k, d, l_len, r_len):
    L_m = 0
    R_m = 0

    for ant in range(m):
        R_p = ((R_m + k) ** d) / r_len
        L_p = ((L_m + k) ** d) / l_len

        P_r = R_p / (R_p + L_p)

        r = random.uniform(0, 1)

        if r < P_r:
            R_m += 1
        else:
            L_m += 1

    print(f"Asymmetric bridge: left = {L_m} times, right = {R_m} times.")

for i in range(10):
    symmetric_bridge(m, k, d)
for i in range(10):
    asymmetric_bridge(m, k, d, L_left, L_right)