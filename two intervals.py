from scipy import integrate
import numpy as np
import sympy as sp


#1/(2*(1-a))*(1-2*a+np.sqrt(np.square(a)+np.square(1-a))*(2*x-1)/np.sqrt(np.square(x)+np.square(1-x)))
#1/(2*(1-a))*(1-2*a+np.sqrt(np.square(a)+np.square(1-a))*(2*y-1)/np.sqrt(np.square(y)+np.square(1-y)))
def equilibrium1(a, b, c, d, gamma):
    delta2 = (1 - a * (1 - d) - d * (1 - a)) / ((1 - a) * (np.square(1 - d) + np.square(d)))
    theta_f = 1 / (2 * (1 - a))
    f = lambda x: 1/(2*(1-a))*(1-2*a+np.sqrt(np.square(a)+np.square(1-a))*(2*x-1)/np.sqrt(np.square(x)+np.square(1-x)))
    v, err = integrate.quad(f, c, gamma)
    F_b = 1/(2*(1-a))*(1-2*a+np.sqrt(np.square(a)+np.square(1-a))*(2*c-1)/np.sqrt(np.square(c)+np.square(1-c)))
    F_b_ = 1 - theta_f - (c-b)*F_b - v - (d-gamma)*(1-delta2) - (1-d)
    z = sp.symbols('z')
    delta1 = sp.solve((1 - b) * theta_f + (np.square(1 - b) + np.square(b)) * (F_b - z / 2) + (1 - 2 * b) * F_b_ - 0.5, z)[0]
    F_y = F_b - delta1
    if F_y <= 0:
        return 10
    else:
        gamma_ = sp.solve(
            (1 - z) * theta_f + (np.square(1 - z) + np.square(z)) * F_y + (1 - 2 * z) * (F_b_ - (b - z) * F_y) - 0.5,
            z)[0]
        if gamma_ < a or gamma_ > b:
            print('error:', a, b, c, d, gamma_)
            return 15
        else:
            # y:0-a, x:0-y
            v1 = 0.5 * np.square(a)

            # y:a-gamma_, x:0-a
            f2 = lambda x, y: np.square(1 - 1 / (2 * (1 - a)) * (
                    1 - 2 * a + np.sqrt(np.square(a) + np.square(1 - a)) * (2 * y - 1) / np.sqrt(
                np.square(y) + np.square(1 - y))))
            g2 = lambda y: 0
            h2 = lambda y: a
            v2, err2 = integrate.dblquad(f2, a, gamma_, g2, h2)

            # y:a-gamma_, x:a-y
            f3 = lambda x, y: np.square(1 + 1 / (2 * (1 - a)) * (
                    1 - 2 * a + np.sqrt(np.square(a) + np.square(1 - a)) * (2 * x - 1) / np.sqrt(
                np.square(x) + np.square(1 - x))) - 1 / (2 * (1 - a)) * (
                                                1 - 2 * a + np.sqrt(np.square(a) + np.square(1 - a)) * (
                                                2 * y - 1) / np.sqrt(
                                            np.square(y) + np.square(1 - y))))
            g3 = lambda y: a
            h3 = lambda y: y
            v3, err3 = integrate.dblquad(f3, a, gamma_, g3, h3)

            # y:gamma_-b, x:0-a
            f4 = lambda x, y: np.square(1 - F_y)
            g4 = lambda y: 0
            h4 = lambda y: a
            v4, err4 = integrate.dblquad(f4, gamma_, b, g4, h4)

            # y:gamma_-b, x:a-gamma_
            f5 = lambda x, y: np.square(1 + 1 / (2 * (1 - a)) * (
                    1 - 2 * a + np.sqrt(np.square(a) + np.square(1 - a)) * (2 * x - 1) / np.sqrt(
                np.square(x) + np.square(1 - x))) - F_y)
            g5 = lambda y: a
            h5 = lambda y: gamma_
            v5, err5 = integrate.dblquad(f5, gamma_, b, g5, h5)

            # y:gamma_-b, x:gamma_-y
            f6 = lambda x, y: 1
            g6 = lambda y: gamma_
            h6 = lambda y: y
            v6, err6 = integrate.dblquad(f6, gamma_, b, g6, h6)

            # y:b-c, x:0-a
            f7 = lambda x, y: np.square(1-F_b)
            g7 = lambda y: 0
            h7 = lambda y: a
            v7, err7 = integrate.dblquad(f7, b, c, g7, h7)

            # y:b-c, x:a-gamma_
            f8 = lambda x, y: np.square(1 + 1 / (2 * (1 - a)) * (
                    1 - 2 * a + np.sqrt(np.square(a) + np.square(1 - a)) * (2 * x - 1) / np.sqrt(
                np.square(x) + np.square(1 - x))) - F_b)
            g8 = lambda y: a
            h8 = lambda y: gamma_
            v8, err8 = integrate.dblquad(f8, b, c, g8, h8)

            # y:b-c, x:gamma_-b
            f9 = lambda x, y: np.square(1 + F_y - F_b)
            g9 = lambda y: gamma_
            h9 = lambda y: b
            v9, err9 = integrate.dblquad(f9, b, c, g9, h9)

            # y:b-c, x:b-y
            f10 = lambda x, y: 1
            g10 = lambda y: b
            h10 = lambda y: y
            v10, err10 = integrate.dblquad(f10, b, c, g10, h10)

            # y:c-gamma, x:0-a
            f11 = lambda x, y: np.square(1 - 1 / (2 * (1 - a)) * (
                    1 - 2 * a + np.sqrt(np.square(a) + np.square(1 - a)) * (2 * y - 1) / np.sqrt(
                np.square(y) + np.square(1 - y))))
            g11 = lambda y: 0
            h11 = lambda y: a
            v11, err11 = integrate.dblquad(f11, c, gamma, g11, h11)

            # y:c-gamma, x:a-gamma_
            f12 = lambda x, y: np.square(1 + 1 / (2 * (1 - a)) * (
                    1 - 2 * a + np.sqrt(np.square(a) + np.square(1 - a)) * (2 * x - 1) / np.sqrt(
                np.square(x) + np.square(1 - x))) - 1 / (2 * (1 - a)) * (
                                                1 - 2 * a + np.sqrt(np.square(a) + np.square(1 - a)) * (
                                                2 * y - 1) / np.sqrt(
                                            np.square(y) + np.square(1 - y))))
            g12 = lambda y: a
            h12 = lambda y: gamma_
            v12, err12 = integrate.dblquad(f12, c, gamma, g12, h12)

            # y:c-gamma, x:gamma_-b
            f13 = lambda x, y: np.square(1 + F_y - 1 / (2 * (1 - a)) * (
                    1 - 2 * a + np.sqrt(np.square(a) + np.square(1 - a)) * (2 * y - 1) / np.sqrt(
                np.square(y) + np.square(1 - y))))
            g13 = lambda y: gamma_
            h13 = lambda y: b
            v13, err13 = integrate.dblquad(f13, c, gamma, g13, h13)

            # y:c-gamma, x:b-c
            f14 = lambda x, y: np.square(1 + F_b - 1 / (2 * (1 - a)) * (
                    1 - 2 * a + np.sqrt(np.square(a) + np.square(1 - a)) * (2 * y - 1) / np.sqrt(
                np.square(y) + np.square(1 - y))))
            g14 = lambda y: b
            h14 = lambda y: c
            v14, err14 = integrate.dblquad(f14, c, gamma, g14, h14)

            # y:c-gamma, x:c-y
            f15 = lambda x, y: np.square(1 + 1 / (2 * (1 - a)) * (
                    1 - 2 * a + np.sqrt(np.square(a) + np.square(1 - a)) * (2 * x - 1) / np.sqrt(
                np.square(x) + np.square(1 - x))) - 1 / (2 * (1 - a)) * (
                                                1 - 2 * a + np.sqrt(np.square(a) + np.square(1 - a)) * (
                                                2 * y - 1) / np.sqrt(
                                            np.square(y) + np.square(1 - y))))
            g15 = lambda y: c
            h15 = lambda y: y
            v15, err15 = integrate.dblquad(f15, c, gamma, g15, h15)

            # y:gamma-d, x:0-a
            f16 = lambda x, y: np.square(delta2)
            g16 = lambda y: 0
            h16 = lambda y: a
            v16, err16 = integrate.dblquad(f16, gamma, d, g16, h16)

            # y:gamma-d, x:a-gamma_
            f17 = lambda x, y: np.square(1 + 1 / (2 * (1 - a)) * (
                    1 - 2 * a + np.sqrt(np.square(a) + np.square(1 - a)) * (2 * x - 1) / np.sqrt(
                np.square(x) + np.square(1 - x))) - (1 - delta2))
            g17 = lambda y: a
            h17 = lambda y: gamma_
            v17, err17 = integrate.dblquad(f17, gamma, d, g17, h17)

            # y:gamma-d, x:gamma_-b
            f18 = lambda x, y: np.square(1 + F_y - (1 - delta2))
            g18 = lambda y: gamma_
            h18 = lambda y: b
            v18, err18 = integrate.dblquad(f18, gamma, d, g18, h18)

            # y:gamma-d, x:b-c
            f19 = lambda x, y: np.square(1 + F_b - (1 - delta2))
            g19 = lambda y: b
            h19 = lambda y: c
            v19, err19 = integrate.dblquad(f19, gamma, d, g19, h19)

            # y:gamma-d, x:c-gamma
            f20 = lambda x, y: np.square(1 + 1 / (2 * (1 - a)) * (
                    1 - 2 * a + np.sqrt(np.square(a) + np.square(1 - a)) * (2 * x - 1) / np.sqrt(
                np.square(x) + np.square(1 - x))) - (1 - delta2))
            g20 = lambda y: c
            h20 = lambda y: gamma
            v20, err20 = integrate.dblquad(f20, gamma, d, g20, h20)

            # y:gamma-d, x:gamma-y
            f21 = lambda x, y: 1
            g21 = lambda y: gamma
            h21 = lambda y: y
            v21, err21 = integrate.dblquad(f21, gamma, d, g21, h21)

            # y:d-1, x:0-a
            v22 = 0

            # y:d-1, x:a-gamma_
            f23 = lambda x, y: np.square(1 / (2 * (1 - a)) * (
                    1 - 2 * a + np.sqrt(np.square(a) + np.square(1 - a)) * (2 * x - 1) / np.sqrt(
                np.square(x) + np.square(1 - x))))
            g23 = lambda y: a
            h23 = lambda y: gamma_
            v23, err23 = integrate.dblquad(f23, d, 1, g23, h23)

            # y:d-1, x:gamma_-b
            f24 = lambda x, y: np.square(F_y)
            g24 = lambda y: gamma_
            h24 = lambda y: b
            v24, err24 = integrate.dblquad(f24, d, 1, g24, h24)

            # y:d-1, x:b-c
            f25 = lambda x, y: np.square(F_b)
            g25 = lambda y: b
            h25 = lambda y: c
            v25, err25 = integrate.dblquad(f25, d, 1, g25, h25)

            # y:d-1, x:c-gamma
            f26 = lambda x, y: np.square(1 / (2 * (1 - a)) * (
                    1 - 2 * a + np.sqrt(np.square(a) + np.square(1 - a)) * (2 * x - 1) / np.sqrt(
                np.square(x) + np.square(1 - x))))
            g26 = lambda y: c
            h26 = lambda y: gamma
            v26, err26 = integrate.dblquad(f26, d, 1, g26, h26)

            # y:d-1, x:gamma-d
            f27 = lambda x, y: np.square(1-delta2)
            g27 = lambda y: gamma
            h27 = lambda y: d
            v27, err27 = integrate.dblquad(f27, d, 1, g27, h27)

            # y:d-1, x:d-y
            f28 = lambda x, y: 1
            g28 = lambda y: d
            h28 = lambda y: y
            v28, err28 = integrate.dblquad(f28, d, 1, g28, h28)

            return v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10 + v11 + v12 + v13 + v14 + v15 + v16 + v17 + v18 + v19 + v20 + v21 + v22 + v23 + v24 + v25 + v26 + v27 + v28

def equilibrium2(a, b, d):
    delta2 = (1-a*(1-d)-d*(1-a))/((1-a)*(np.square(1-d)+np.square(d)))
    theta_f = 1/(2*(1-a))
    F_b = 1 - delta2
    F_b_ = 1 - theta_f - (d-b)*F_b - (1-d)
    z = sp.symbols('z')
    delta1 = sp.solve((1-b)*theta_f + (np.square(1-b)+np.square(b))*(F_b-z/2) + (1-2*b)*F_b_ - 0.5, z)[0]
    if delta1 + delta2 >= 1:
        return 10
    else:
        F_y = 1 - delta2 - delta1
        gamma_ = sp.solve((1-z)*theta_f + (np.square(1-z)+np.square(z))*F_y + (1-2*z)*(F_b_-(b-z)*F_y) - 0.5, z)[0]
        if gamma_ < a or gamma_ > b:
            print('error:', a, b, d, gamma_)
            return 15
        else:
            # y:0-a, x:0-y
            v1 = 0.5*np.square(a)

            # y:a-gamma_, x:0-a
            f2 = lambda x, y: np.square(1 - 1 / (2 * (1 - a)) * (
                    1 - 2 * a + np.sqrt(np.square(a) + np.square(1 - a)) * (2 * y - 1) / np.sqrt(
                np.square(y) + np.square(1 - y))))
            g2 = lambda y: 0
            h2 = lambda y: a
            v2, err2 = integrate.dblquad(f2, a, gamma_, g2, h2)

            # y:a-gamma_, x:a-y
            f3 = lambda x, y: np.square(1 + 1 / (2 * (1 - a)) * (
                        1 - 2 * a + np.sqrt(np.square(a) + np.square(1 - a)) * (2 * x - 1) / np.sqrt(
                    np.square(x) + np.square(1 - x))) - 1 / (2 * (1 - a)) * (
                                                1 - 2 * a + np.sqrt(np.square(a) + np.square(1 - a)) * (
                                                    2 * y - 1) / np.sqrt(
                                            np.square(y) + np.square(1 - y))))
            g3 = lambda y: a
            h3 = lambda y: y
            v3, err3 = integrate.dblquad(f3, a, gamma_, g3, h3)

            # y:gamma_-b, x:0-a
            f4 = lambda x, y: np.square(1 - F_y)
            g4 = lambda y: 0
            h4 = lambda y: a
            v4, err4 = integrate.dblquad(f4, gamma_, b, g4, h4)

            # y:gamma_-b, x:a-gamma_
            f5 = lambda x, y: np.square(1 + 1 / (2 * (1 - a)) * (
                    1 - 2 * a + np.sqrt(np.square(a) + np.square(1 - a)) * (2 * x - 1) / np.sqrt(
                np.square(x) + np.square(1 - x))) - F_y)
            g5 = lambda y: a
            h5 = lambda y: gamma_
            v5, err5 = integrate.dblquad(f5, gamma_, b, g5, h5)

            # y:gamma_-b, x:gamma_-y
            f6 = lambda x, y: 1
            g6 = lambda y: gamma_
            h6 = lambda y: y
            v6, err6 = integrate.dblquad(f6, gamma_, b, g6, h6)

            # y:b-d, x:0-a
            f7 = lambda x, y: np.square(delta2)
            g7 = lambda y: 0
            h7 = lambda y: a
            v7, err7 = integrate.dblquad(f7, b, d, g7, h7)

            # y:b-d, x:a-gamma_
            f8 = lambda x, y: np.square(1 + 1 / (2 * (1 - a)) * (
                    1 - 2 * a + np.sqrt(np.square(a) + np.square(1 - a)) * (2 * x - 1) / np.sqrt(
                np.square(x) + np.square(1 - x))) - (1-delta2))
            g8 = lambda y: a
            h8 = lambda y: gamma_
            v8, err8 = integrate.dblquad(f8, b, d, g8, h8)

            # y:b-d, x:gamma_-b
            f9 = lambda x, y: np.square(1 + F_y - (1-delta2))
            g9 = lambda y: gamma_
            h9 = lambda y: b
            v9, err9 = integrate.dblquad(f9, b, d, g9, h9)

            # y:b-d, x:b-y
            f10 = lambda x, y: 1
            g10 = lambda y: b
            h10 = lambda y: y
            v10, err10 = integrate.dblquad(f10, b, d, g10, h10)

            # y:d-1, x:0-a
            v11 = 0

            # y:d-1, x:a-gamma_
            f12 = lambda x, y: np.square(1 / (2 * (1 - a)) * (
                    1 - 2 * a + np.sqrt(np.square(a) + np.square(1 - a)) * (2 * x - 1) / np.sqrt(
                np.square(x) + np.square(1 - x))))
            g12 = lambda y: a
            h12 = lambda y: gamma_
            v12, err12 = integrate.dblquad(f12, d, 1, g12, h12)

            # y:d-1, x:gamma_-b
            f13 = lambda x, y: np.square(F_y)
            g13 = lambda y: gamma_
            h13 = lambda y: b
            v13, err13 = integrate.dblquad(f13, d, 1, g13, h13)

            # y:d-1, x:b-d
            f14 = lambda x, y: np.square(1-delta2)
            g14 = lambda y: b
            h14 = lambda y: d
            v14, err14 = integrate.dblquad(f14, d, 1, g14, h14)

            # y:d-1, x:d-y
            f15 = lambda x, y: 1
            g15 = lambda y: d
            h15 = lambda y: y
            v15, err15 = integrate.dblquad(f15, d, 1, g15, h15)

            return v1+v2+v3+v4+v5+v6+v7+v8+v9+v10+v11+v12+v13+v14+v15


eq = []
m_inversion = 10
for a_ in range(0, 101):
    for b_ in range(0, 101):
        for c_ in range(0, 101):
            for d_ in range(0, 101):
                if a_ < b_ and b_ < c_ and c_ < d_:
                   a = a_/100
                   b = b_/100
                   c = c_/100
                   d = d_/100
                   if (1-a)*d > 0.5:
                       gamma = (1-a-2*d+4*a*d-2*a*d**2)/(1-4*(1-a)*d+2*(1-2*a)*d**2)
                       if b < gamma:
                           if c < gamma:
                               temp = equilibrium1(a, b, c, d, gamma)
                               if temp < m_inversion:
                                   m_inversion = temp
                                   eq = [a, b, c, d]
                           if c > gamma:
                               temp = equilibrium2(a, b, d)
                               if temp < m_inversion:
                                   m_inversion = temp
                                   eq = [a, b, c, d]

print(m_inversion)
print(eq)
